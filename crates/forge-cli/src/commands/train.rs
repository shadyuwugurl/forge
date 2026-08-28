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
) -> Result<()> {
    let method = match method {
        "lora" => TrainMethod::LoRA,
        "qlora" => TrainMethod::QLoRA,
        "dora" => TrainMethod::DoRA,
        "grpo" => TrainMethod::GRPO,
        "dapo" => TrainMethod::DAPO,
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
    };

    let mut trainer = Trainer::new(config)?;
    trainer.train()?;
    Ok(())
}