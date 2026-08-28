use std::path::Path;
use anyhow::Result;
use forge_train::FusingPipeline;

pub fn run(base: &Path, adapters: &Path, output: &Path) -> Result<()> {
    let pipeline = FusingPipeline::from_config(&std::path::Path::new("."))?;
    pipeline.run(base, adapters, output)?;
    Ok(())
}
