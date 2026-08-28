use std::path::Path;
use anyhow::Result;
use forge_io::TensorStore;
use forge_quant::{JangQuantizer, Dynamic3Quantizer, ApexQuantizer, MixedPrecisionQuantizer, KvCacheOrganizer, GgufWriter, GGUFQuantType};
use forge_quant::mixed::MixedStrategy;

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

    let store = TensorStore::open(std::path::Path::new(model))?;
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
            let mut writer = GgufWriter::create(output)?;
            writer.set_metadata("general.architecture", serde_json::Value::String("generic".into()));
            writer.set_metadata("general.name", serde_json::Value::String("forge-quantized".into()));
            writer.write_quantized(&store, qtype)?;
        }
        _ => return Err(anyhow::anyhow!("Unknown quant method: {} (try jang, dynamic3, apex, btl4, mixed, gguf, kv-cache)", method)),
    }
    eprintln!("Quantized model written to {}", output.display());
    Ok(())
}
