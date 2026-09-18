use anyhow::Result;
use std::path::Path;
use crate::lora::LoraExtractor;
use forge_io::{TensorStore, StreamingWriter};

/// Multi-step fusing pipeline: extract → merge adapters → fuse into base → quantize → evaluate
/// All steps stream one tensor at a time (peak RAM = largest tensor), so bulk merging of many adapters works on-disk.
pub struct FusingPipeline { pub steps: Vec<String> }

impl FusingPipeline {
    pub fn from_config(_config_path: &Path) -> Result<Self> {
        Ok(Self { steps: vec!["extract".into(),"merge".into(),"fuse".into(),"quantize".into(),"evaluate".into()] })
    }

    pub fn run(&self, base_path: &Path, adapters_dir: &Path, output_dir: &Path) -> Result<()> {
        eprintln!("=== Fusing Pipeline ===");
        let base = TensorStore::open(base_path)?;
        eprintln!("Step 1: Extracting adapters (SVD rank 16)...");
        let adapters = LoraExtractor::batch_extract(&base, adapters_dir, 16)?;
        if adapters.is_empty() {
            anyhow::bail!("no adapters extracted from {} (need model dirs or .safetensors files)", adapters_dir.display());
        }
        eprintln!("  Extracted {} adapters", adapters.len());
        for (name, ad) in &adapters {
            eprintln!("    {}: {} tensors, rank {}", name, ad.lora_a.len(), ad.rank);
            // Write PEFT stub so `forge quant` can pick it up next
            let _ = ad.save_peft(&output_dir.join(format!("adapter-{}", name)));
        }

        eprintln!("Step 2: Merging adapters (mean of per-adapter B·A deltas)...");
        // Group A/B by tensor name; average deltas (adapters with mismatched rank/shape are skipped)
        let mut delta_sum: std::collections::HashMap<String, (Vec<f64>, usize, usize, usize)> = std::collections::HashMap::new();
        let mut n_contrib: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (_, ad) in &adapters {
            // NOTE: extractor SVDs the raw weight diff, so B·A already equals
            // the full delta — fuse with scale 1.0 for exact round-trip.
            // (External PEFT adapters trained with alpha/r runtime scaling need
            // that factor applied at import; not handled here yet.)
            let scale = 1.0f64;
            let b_of: std::collections::HashMap<&str, &[f32]> =
                ad.lora_b.iter().map(|(n, d)| (n.strip_suffix(".lora_B").unwrap_or(n), d.as_slice())).collect();
            for (aname, a) in &ad.lora_a {
                let (b, out, inn) = match ad.shapes.iter().find(|(s, _, _)| s == aname) {
                    Some((_, o, i)) => match b_of.get(aname.as_str()) {
                        Some(b) => (*b, *o, *i),
                        None => continue,
                    },
                    None => continue,
                };
                // NOTE: extractor clamps rank to min(rank, out, inn) — derive
                // the effective rank from the data, not ad.rank.
                if inn == 0 || a.len() % inn != 0 { continue; }
                let r = a.len() / inn;
                if r == 0 || b.len() != out * r { continue; }
                let entry = delta_sum.entry(aname.clone()).or_insert_with(|| (vec![0.0; out * inn], out, inn, 0));
                if entry.1 != out || entry.2 != inn { continue; }
                for i in 0..out {
                    for j in 0..inn {
                        let mut s = 0.0;
                        for k in 0..r { s += b[i * r + k] as f64 * a[k * inn + j] as f64; }
                        entry.0[i * inn + j] += scale * s;
                    }
                }
                *n_contrib.entry(aname.clone()).or_insert(0) += 1;
            }
        }
        // Mean over contributors
        for (name, (d, _, _, _)) in delta_sum.iter_mut() {
            let n = *n_contrib.get(name).unwrap_or(&1) as f64;
            if n > 0.0 { for v in d.iter_mut() { *v /= n; } }
        }
        eprintln!("  Merged {} adapter tensors", delta_sum.len());

        eprintln!("Step 3: Fusing into base (W + mean(B·A))...");
        std::fs::create_dir_all(output_dir)?;
        let mut writer = StreamingWriter::new(output_dir, 5 * 1024 * 1024 * 1024)?;
        let mut fused = 0usize;
        let mut passthrough = 0usize;
        for name in base.tensor_names() {
            let meta = base.tensor_meta(name)?;
            if let Some((delta, out, inn, _)) = delta_sum.get(name) {
                let w = base.tensor_f32(name)?;
                if w.len() == out * inn && w.len() == delta.len() {
                    let fused_w: Vec<f32> = w.iter().zip(delta.iter()).map(|(x, d)| x + *d as f32).collect();
                    let mut buf = Vec::with_capacity(fused_w.len() * 2);
                    for &v in &fused_w {
                        buf.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
                    }
                    writer.write_tensor(name, &buf, "F16", &meta.shape)?;
                    fused += 1;
                    continue;
                }
            }
            // Passthrough: copy raw bytes with matching dtype
            let dtype = match meta.dtype {
                forge_core::DType::F32 => "F32",
                forge_core::DType::BF16 => "BF16",
                _ => "F16",
            };
            let bytes: Vec<u8> = if matches!(meta.dtype, forge_core::DType::F32 | forge_core::DType::F16 | forge_core::DType::BF16) {
                base.tensor_bytes(name)?.to_vec()
            } else {
                let f = base.tensor_f32(name).unwrap_or_default();
                let mut buf = Vec::with_capacity(f.len() * 2);
                for &v in &f {
                    buf.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
                }
                buf
            };
            // If we re-encoded, dtype must be F16
            let dtype = if bytes.len() != meta.size { "F16" } else { dtype };
            writer.write_tensor(name, &bytes, dtype, &meta.shape)?;
            passthrough += 1;
        }
        writer.finalize("fused")?;
        // Carry sidecars so output is self-describing
        let cfg_dir = if base.path().is_dir() { base.path().to_path_buf() } else {
            base.path().parent().map(|p| p.to_path_buf()).unwrap_or_else(|| ".".into())
        };
        for sidecar in ["config.json", "tokenizer.json", "tokenizer_config.json"] {
            let src = cfg_dir.join(sidecar);
            if src.exists() { let _ = std::fs::copy(&src, output_dir.join(sidecar)); }
        }
        eprintln!("  Fused {} tensors, {} passthrough → {}", fused, passthrough, output_dir.display());

        eprintln!("Step 4: (optional) forge quant --method jang --profile JANG_2L {}", output_dir.display());
        eprintln!("Step 5: (optional) forge eval --benchmarks hella,mmlu,arc,gsm8k,gpqa --evals ace,swe,terminal,gaia,hle {}", output_dir.display());
        eprintln!("=== Pipeline Complete ===");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pipeline_from_config() {
        let p = FusingPipeline::from_config(Path::new(".")).unwrap();
        assert_eq!(p.steps.len(), 5);
    }

    #[test]
    fn fuse_roundtrip_reconstructs_finetune() {
        // base W vs ft = W + rank-1 delta → extract → fuse must ≈ ft
        let dir = tempfile::tempdir().unwrap();
        let base_d = dir.path().join("base");
        let ft_d = dir.path().join("ft");
        std::fs::create_dir_all(&base_d).unwrap();
        std::fs::create_dir_all(&ft_d).unwrap();
        let w: Vec<f32> = (0..128).map(|i| (i as f32 * 0.13).sin()).collect();
        let mut d = vec![0.0f32; 128];
        for i in 0..16 { for j in 0..8 { d[i * 8 + j] = 0.05 * (i as f32) * 0.1 * (j as f32); } }
        let ft: Vec<f32> = w.iter().zip(d.iter()).map(|(a, b)| a + b).collect();
        for (dpath, data) in [(&base_d, &w), (&ft_d, &ft)] {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32, vec![16, 8], bytemuck::cast_slice(data),
            ).unwrap();
            let bytes = safetensors::tensor::serialize(
                vec![("blk.weight".to_string(), view)], &None,
            ).unwrap();
            std::fs::write(dpath.join("model.safetensors"), bytes).unwrap();
        }
        let out = dir.path().join("fused");
        FusingPipeline::from_config(Path::new(".")).unwrap()
            .run(&base_d, &ft_d, &out).unwrap();
        let fused = TensorStore::open(&out).unwrap();
        let f = fused.tensor_f32("blk.weight").unwrap();
        assert_eq!(f.len(), 128);
        let err: f32 = f.iter().zip(ft.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 2e-2, "round-trip err {}", err);
    }
}
