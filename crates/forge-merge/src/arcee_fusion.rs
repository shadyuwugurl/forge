use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Arcee Fusion: dynamic-threshold fusion of salient changes.
/// For each parameter, keep a finetune's delta only if |δ| ≥ τ·σ,
/// where σ is the per-tensor std of deltas across models (τ = `threshold*`).
/// Survivors are averaged (sign-consistent) and added to base scaled by λ.
/// This fuses only "important" changes and drops noise.
pub struct ArceeFusionMerge {
    pub base_index: usize,
    pub lambda: f32,
    /// Keep deltas with |δ| >= threshold_std * std(delta)
    pub threshold_std: f32,
}

impl ArceeFusionMerge {
    pub fn new(lambda: f32, threshold_std: f32) -> Self {
        Self { base_index: 0, lambda, threshold_std }
    }
}

impl MergeOp for ArceeFusionMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() < 2 {
            anyhow::bail!("arcee_fusion: need base + ≥1 finetune");
        }
        let base_idx = self.base_index.min(inputs.len() - 1);
        let base = &inputs[base_idx];
        if base.len() != meta.num_elements() {
            anyhow::bail!("arcee_fusion: base tensor size mismatch");
        }
        let deltas: Vec<Vec<f32>> = inputs
            .iter()
            .enumerate()
            .filter(|(i, ft)| *i != base_idx && ft.len() == base.len())
            .map(|(_, ft)| base.iter().zip(ft.iter()).map(|(b, f)| f - b).collect())
            .collect();
        if deltas.is_empty() {
            return Ok(base.clone());
        }
        // Per-tensor std of all deltas
        let all: Vec<f32> = deltas.iter().flat_map(|d| d.iter().copied()).collect();
        let m = all.iter().sum::<f32>() / all.len() as f32;
        let std = (all.iter().map(|x| (x - m).powi(2)).sum::<f32>() / all.len() as f32)
            .sqrt()
            .max(1e-12);
        let cutoff = self.threshold_std * std;

        let mut result = base.clone();
        for j in 0..base.len() {
            let mut acc = 0.0f32;
            let mut cnt = 0usize;
            for d in &deltas {
                if d[j].abs() >= cutoff {
                    acc += d[j];
                    cnt += 1;
                }
            }
            if cnt > 0 {
                result[j] += self.lambda * acc / cnt as f32;
            }
        }
        Ok(result)
    }
}
