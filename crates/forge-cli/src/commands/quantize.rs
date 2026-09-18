use std::path::Path;
use anyhow::{Result, Context};
use forge_io::TensorStore;
use forge_quant::{JangQuantizer, Dynamic3Quantizer, ApexQuantizer, MixedPrecisionQuantizer, KvCacheOrganizer, GgufWriter, GGUFQuantType};
use forge_quant::mixed::MixedStrategy;
use bytemuck::cast_slice;

fn quantize_per_tensor<F>(store: &TensorStore, output: &Path, quantize_fn: F) -> Result<()>
where
    F: Fn(&[f32]) -> Result<(Vec<u8>, Vec<f32>)> + Sync,
{
    std::fs::create_dir_all(output)?;
    let names = store.tensor_names();
    let mut writer = forge_io::StreamingWriter::new(output, 5 * 1024 * 1024 * 1024)?;

    // M5c: same parallel pipeline as merge — workers quantize, main
    // thread writes (packed + scales) in arrival order. 4GB byte budget
    // caps in-flight tensors; FORGE_WORKERS overrides worker count.
    let workers = std::env::var("FORGE_WORKERS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
        .max(1)
        .min(names.len().max(1));
    {
        use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
        use std::sync::mpsc::sync_channel;
        const BUDGET: u64 = 4 * 1024 * 1024 * 1024;
        let next = AtomicUsize::new(0);
        let in_flight = AtomicU64::new(0);
        let (tx, rx) = sync_channel::<Option<(String, Vec<usize>, Vec<u8>, Vec<f32>)>>(workers * 2);
        let (next_r, flight_r, names_r, qf) = (&next, &in_flight, &names, &quantize_fn);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                let tx = tx.clone();
                scope.spawn(move || {
                    loop {
                        let i = next_r.fetch_add(1, Ordering::Relaxed);
                        let Some(name) = names_r.get(i) else { break };
                        let meta = match store.tensor_meta(name) {
                            Ok(m) => m,
                            Err(_) => {
                                let _ = tx.send(None);
                                continue;
                            }
                        };
                        let cost = (meta.size as u64).saturating_mul(2).max(1);
                        while flight_r.fetch_add(cost, Ordering::SeqCst) + cost > BUDGET {
                            flight_r.fetch_sub(cost, Ordering::SeqCst);
                            std::thread::yield_now();
                        }
                        let msg = (|| {
                            let tensor = store.tensor_f32(name).ok()?;
                            let (packed, scales) = qf(&tensor).ok()?;
                            Some((name.to_string(), meta.shape.clone(), packed, scales))
                        })();
                        flight_r.fetch_sub(cost, Ordering::SeqCst);
                        if tx.send(msg).is_err() {
                            break;
                        }
                    }
                });
            }
            drop(tx);
            let mut done = 0usize;
            for msg in rx {
                if let Some((name, shape, packed, scales)) = msg {
                    writer.write_tensor(&name, &packed, "U8", &shape)?;
                    writer.write_tensor(
                        &format!("{}_scales", name),
                        bytemuck::cast_slice(&scales),
                        "F16",
                        &[scales.len()],
                    )?;
                }
                done += 1;
                if done >= names.len() {
                    break;
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
    }

    writer.finalize("model")?;
    eprintln!("Quantized model written to {}", output.display());
    Ok(())
}

pub fn run(model: &str, method: &str, profile: Option<&str>, output: &Path, density: Option<f32>) -> Result<()> {
    // KV cache special case: `forge quant --method kv-cache --profile "32,32,128"` or just `forge info --kv-cache`
    if method == "kv-cache" || method == "kv" {
        let seq_len: usize = profile.and_then(|p| p.parse().ok()).unwrap_or(8192);
        let org = KvCacheOrganizer::new(32, 32, 128);
        let info = org.describe(seq_len);
        eprintln!("KV cache {} tokens: {} GB, {:?}", seq_len, info["memory_gb"], info["quant"]);
        std::fs::create_dir_all(output)?;
        std::fs::write(output.join("kv_cache.json"), serde_json::to_string_pretty(&info)?)?;
        return Ok(());
    }

    let model_path = std::path::Path::new(model);
    // TensorStore::open handles single-file + sharded dirs (merged 9B outputs are sharded)
    let store = TensorStore::open(model_path)
        .with_context(|| format!("opening model {}", model))?;
    std::fs::create_dir_all(output)?;
    eprintln!("Quantizing {} with method '{}'", model, method);
    match method {
        "jang" => {
            let profile_name = profile.unwrap_or("JANG_2L");
            let q = JangQuantizer::new(profile_name, forge_quant::jang::JangFormat::Mlx);
            q.quantize(&store, output)?;
        }
        "dynamic3"|"dynamic" => {
            let d = density.unwrap_or(0.5);
            let q = Dynamic3Quantizer::new(d, true);
            q.quantize(&store, output)?;
        }
        "apex" => {
            let tier = profile.unwrap_or("balanced");
            ApexQuantizer::new(tier).quantize(&store, output)?;
        }
        "btl4" => {
            MixedPrecisionQuantizer::new(MixedStrategy::Btl4Compact, density.unwrap_or(4.0)).quantize(&store, output, &[])?;
        }
        "mixed" => {
            // profile can be "apex" to use apex-style tiering, else generic
            let strat = if profile == Some("apex") { MixedStrategy::ApexStyle } else { MixedStrategy::Generic };
            MixedPrecisionQuantizer::new(strat, density.unwrap_or(4.0)).quantize(&store, output, &[])?;
        }
        "gguf" => {
            let qtype = profile.and_then(|p| GGUFQuantType::from_str(p)).unwrap_or(GGUFQuantType::Q4_K_M);
            // GgufWriter takes a FILE path; accept a dir and pick a filename.
            let file = if output.extension().map(|x| x == "gguf").unwrap_or(false) {
                output.to_path_buf()
            } else {
                output.join(format!("model-{}.gguf", qtype.name()))
            };
            let mut writer = GgufWriter::create(&file)?;
            writer.set_metadata("general.architecture", serde_json::Value::String("generic".into()));
            writer.set_metadata("general.name", serde_json::Value::String("forge-quantized".into()));
            writer.write_quantized(&store, qtype)?;
            eprintln!("GGUF written to {}", file.display());
        }
        "bsqat" => {
            let bits: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(4);
            let block: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::BsqatQuantizer::new(bits, block);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "onecomp" => {
            let hi_bits: u8 = profile.and_then(|p| p.split(':').next().and_then(|s| s.parse().ok())).unwrap_or(4);
            let lo_bits: u8 = profile.and_then(|p| p.split(':').nth(1).and_then(|s| s.parse().ok())).unwrap_or(2);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::OneCompQuantizer::new(hi_bits, lo_bits, group, u64::MAX);
            quantize_per_tensor(&store, output, |tensor| {
                let plan = q.plan_bits(&[tensor.len()]);
                q.quantize(tensor, plan[0])
            })?;
        }
        "quept" => {
            let width: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(4);
            let block: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::QueptQuantizer::new(vec![width], block, 2);
            quantize_per_tensor(&store, output, |tensor| {
                let calib = q.calibrate(tensor);
                q.quantize_at(tensor, &calib, width)
            })?;
        }
        "nanoquant"|"nano" => {
            let bits: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(1);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::NanoQuantQuantizer::new(bits, 50, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "arb" => {
            let bits: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(1);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::ArbQuantizer::new(bits, true, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "hbllm" => {
            let levels: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(3);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::HbllmQuantizer::new(levels, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "dbell"|"dbellquant" => {
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::DbellQuantizer::new(1, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "af1" => {
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::Af1Quantizer::new(1, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "btcllm"|"btc" => {
            // profile "bits:codebook", e.g. "1:2". codebook must equal 2^bits.
            let bits: u8 = profile.and_then(|p| p.split(':').next().and_then(|s| s.parse().ok())).unwrap_or(1);
            let codebook: usize = profile.and_then(|p| p.split(':').nth(1).and_then(|s| s.parse().ok())).unwrap_or(2);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::BtcQuantizer::new(bits, codebook, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        "littlebit" => {
            let bits: u8 = profile.and_then(|p| p.parse().ok()).unwrap_or(1);
            let group: usize = density.and_then(|d| Some(d as usize)).unwrap_or(128);
            let q = forge_quant::LittleBitQuantizer::new(bits, true, group);
            quantize_per_tensor(&store, output, |tensor| q.quantize(tensor))?;
        }
        _ => return Err(anyhow::anyhow!("Unknown quant method: {} (try jang, dynamic3, apex, btl4, mixed, gguf, bsqat, onecomp, quept, nanoquant, arb, hbllm, dbell, af1, btcllm, littlebit, kv-cache)", method)),
    }
    eprintln!("Quantized model written to {}", output.display());
    Ok(())
}
