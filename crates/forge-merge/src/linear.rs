use anyhow::{bail, Result};
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Simple weighted averaging of N model tensors
pub struct LinearMerge<'a> {
    pub models: Vec<(&'a [f32], f32)>,  // (tensor_data, weight)
    pub normalize: bool,
}

impl<'a> LinearMerge<'a> {
    pub fn new(models: Vec<(&'a [f32], f32)>, normalize: bool) -> Self {
        Self { models, normalize }
    }
}

impl MergeOp for LinearMerge<'_> {
    /// Streaming entry point: average `inputs` elementwise. Uses
    /// `self.models` weights when the count matches, else uniform weights.
    /// Tensors whose length differs from the first are skipped (hetero
    /// shapes belong to Chimera/Hetero, not linear averaging).
    fn merge_tensors(&self, _name: &str, _meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            bail!("LinearMerge: no inputs");
        }
        let n = inputs[0].len();
        let weights: Vec<f32> = if self.models.len() == inputs.len() {
            self.models.iter().map(|(_, w)| *w).collect()
        } else {
            vec![1.0; inputs.len()]
        };
        let sum: f32 = weights.iter().sum();
        let inv = if self.normalize && sum != 0.0 { 1.0 / sum } else { 1.0 };
        let mut result = vec![0.0f32; n];
        for (data, w) in inputs.iter().zip(weights.iter()) {
            if data.len() != n {
                continue;
            }
            let w = w * inv;
            for (r, d) in result.iter_mut().zip(data.iter()) {
                *r += d * w;
            }
        }
        Ok(result)
    }

    fn merge_tensor(&self, _name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut result = vec![0.0f32; self.models[0].0.len()];

        let weight_sum: f32 = self.models.iter().map(|(_, w)| w).sum();
        let inv_sum = if self.normalize { 1.0 / weight_sum } else { 1.0 };

        for (data, weight) in &self.models {
            let w = weight * inv_sum;
            for (r, d) in result.iter_mut().zip(data.iter()) {
                *r += d * w;
            }
        }

        Ok(result)
    }
}
