use std::path::Path;
use anyhow::{Context, Result};
use forge_io::TensorStore;
use forge_train::{LoraExtractor, DataRipper};
use forge_train::data_rip::ExtractionMethod;

pub fn run(
    model: &Path,
    base: &Path,
    output: &Path,
    rank: usize,
    method: Option<String>,
    calib: Option<String>,
    teacher: Option<String>,
) -> Result<()> {
    let base_store = TensorStore::open(base)?;
    let model_store = TensorStore::open(model)?;

    let method = method.unwrap_or_else(|| "lora".to_string());

    match method.as_str() {
        "lora" => {
            eprintln!("Extracting LoRA adapter (rank={})...", rank);
            let adapter = LoraExtractor::extract(&base_store, &model_store, rank)?;
            eprintln!("  LoRA A layers: {}", adapter.lora_a.len());
            eprintln!("  LoRA B layers: {}", adapter.lora_b.len());
            adapter.save_peft(output)?;
            eprintln!("  Saved PEFT to {}/adapter_config.json", output.display());
        }
        "weight-diff" => {
            eprintln!("Extracting weight diff (L1)...");
            let ripper = DataRipper::new(forge_train::data_rip::ExtractionMethod::WeightDiff);
            let data = ripper.extract(&base_store, &model_store)?;
            std::fs::create_dir_all(output)?;
            std::fs::write(output.join("weight_diff.json"), serde_json::to_string_pretty(&data)?)?;
            eprintln!("{}", data.summary);
        }
        "activation-probe" => {
            let calib_path = calib.context("activation-probe requires --calib")?;
            eprintln!("Extracting activation probe (L2) with calibration from {}...", calib_path);
            let calib_data = load_calibration(&calib_path)?;
            let ripper = DataRipper::new(forge_train::data_rip::ExtractionMethod::ActivationProbe)
                .with_calibration(calib_data);
            let data = ripper.extract(&base_store, &model_store)?;
            std::fs::create_dir_all(output)?;
            std::fs::write(output.join("activation_probe.json"), serde_json::to_string_pretty(&data)?)?;
            eprintln!("{}", data.summary);
        }
        "distill" => {
            let teacher_path = teacher.context("distill requires --teacher")?;
            let calib_path = calib.context("distill requires --calib")?;
            eprintln!("Running knowledge distillation (L3) with teacher {} and calibration {}...", teacher_path, calib_path);
            let calib_data = load_calibration(&calib_path)?;
            let ripper = DataRipper::new(forge_train::data_rip::ExtractionMethod::KnowledgeDistill)
                .with_calibration(load_calibration(&calib_path)?)
                .with_teacher(teacher_path.to_string());
            let data = ripper.extract(&base_store, &model_store)?;
            std::fs::create_dir_all(output)?;
            std::fs::write(output.join("distilled.json"), serde_json::to_string_pretty(&data)?)?;
            eprintln!("{}", data.summary);
        }
        _ => return Err(anyhow::anyhow!("Unknown method: {} (try lora, weight-diff, activation-probe, distill)", method)),
    }

    eprintln!("Extraction complete: {}", output.display());
    Ok(())
}

fn load_calibration(path: &str) -> anyhow::Result<Vec<forge_train::data_rip::CalibrationSample>> {
    let data = std::fs::read_to_string(path)?;
    let mut samples = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() { continue; }
        let v: serde_json::Value = serde_json::from_str(line)?;
        let input_ids = v.get("input_ids").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect())
            .unwrap_or_default();
        let attention_mask = v.get("attention_mask").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect());
        let labels = v.get("labels").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect());
        samples.push(forge_train::data_rip::CalibrationSample { input_ids, attention_mask, labels });
    }
    Ok(samples)
}