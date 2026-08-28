use std::path::Path;
use anyhow::Result;
use forge_io::TensorStore;
use forge_quant::{JangQuantizer, Dynamic3Quantizer, ApexQuantizer};

pub fn run(model: &str, method: &str, profile: Option<&str>, output: &Path, density: Option<f32>) -> Result<()> {
    let store = TensorStore::open(std::path::Path::new(model))?;
    std::fs::create_dir_all(output)?;

    eprintln!("Quantizing {} with method '{}'", model, method);

    match method {
        "jang" => {
            let profile_name = profile.unwrap_or("JANG_2L");
            let quantizer = JangQuantizer::new(profile_name, forge_quant::jang::JangFormat::Mlx);
            quantizer.quantize(&store, output)?;
        }
        "dynamic3" => {
            let d = density.unwrap_or(0.5);
            let quantizer = Dynamic3Quantizer::new(d, true);
            quantizer.quantize(&store, output)?;
        }
        "apex" => {
            let tier = profile.unwrap_or("balanced");
            let quantizer = ApexQuantizer::new(tier);
            quantizer.quantize(&store, output)?;
        }
        _ => {
            return Err(anyhow::anyhow!("Unknown quant method: {}", method));
        }
    }

    eprintln!("Quantized model written to {}", output.display());
    Ok(())
}
