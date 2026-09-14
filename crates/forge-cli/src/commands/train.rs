use std::path::PathBuf;
use anyhow::Result;
use forge_train::{Trainer, TrainConfig, TrainMethod};

pub fn run(
    model: &str,
    dataset: &str,
    output: &PathBuf,
    method: &str,
    rank: usize,
    alpha: f32,
    lr: f32,
    epochs: usize,
    batch_size: usize,
    boundary: Option<f32>,
) -> Result<()> {
    let method = match method {
        "lora" => TrainMethod::LoRA,
        "qlora" => TrainMethod::QLoRA,
        "dora" => TrainMethod::DoRA,
        "grpo" => TrainMethod::GRPO,
        "dapo" => TrainMethod::DAPO,
        "diffusionblocks" => TrainMethod::DiffusionBlocks,
        "lopt" => TrainMethod::Lopt,
        "lls" => TrainMethod::Lls,
        _ => return Err(anyhow::anyhow!("Unknown method: {}", method)),
    };

    let config = TrainConfig {
        model_path: model.to_string(),
        dataset: dataset.to_string(),
        output: output.to_string_lossy().to_string(),
        rank,
        alpha,
        learning_rate: lr,
        epochs,
        batch_size,
        method,
        quant: None,
        boundary: boundary.unwrap_or(0.5),
    };

    let mut trainer = Trainer::new(config)?;
    trainer.train()?;
    Ok(())
}