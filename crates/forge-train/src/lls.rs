use anyhow::Result;

/// LLS: Log-Linear Attention for pretraining (Phase B, pretrain-only).
/// Provides the `--method lls` entry point. Streaming-safe per-shard
/// normalization; not a full attention kernel.
pub struct LlsTrainer {
    pub dim: usize,
}

impl LlsTrainer {
    pub fn new(dim: usize) -> Self {
        Self { dim: dim.max(8) }
    }
    /// Log-domain normalize one shard: y = log(1 + exp(x - max)) + max - log(n).
    pub fn normalize(&self, input: &[f32]) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        let max_v = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let n = input.len() as f32;
        Ok(input
            .iter()
            .map(|v| (1.0 + (v - max_v).exp()).ln() + max_v - n.ln())
            .collect())
    }
}
