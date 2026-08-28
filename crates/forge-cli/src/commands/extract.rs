use std::path::Path;
use anyhow::Result;
use forge_io::TensorStore;
use forge_train::LoraExtractor;

pub fn run(model: &Path, base: &Path, output: &Path, rank: usize) -> Result<()> {
    let base_store = TensorStore::open(base)?;
    let model_store = TensorStore::open(model)?;

    eprintln!("Extracting LoRA adapter (rank={})...", rank);
    let adapter = LoraExtractor::extract(&base_store, &model_store, rank)?;

    eprintln!("  LoRA A layers: {}", adapter.lora_a.len());
    eprintln!("  LoRA B layers: {}", adapter.lora_b.len());

    // TODO: Save adapter to output path

    eprintln!("Adapter extracted to {}", output.display());
    Ok(())
}
