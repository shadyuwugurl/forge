pub struct GRPOTrainer;

impl GRPOTrainer {
    pub fn new() -> Self { Self }
    pub fn train(&self, _model: &str, _dataset: &std::path::Path, _output: &std::path::Path) -> anyhow::Result<()> { Ok(()) }
}