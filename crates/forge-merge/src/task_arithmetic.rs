use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Task Arithmetic: base + λ · Σ wᵢ·(ftᵢ − base).
/// Convention: inputs[base_index] is the base model (default 0), rest are finetunes.
pub struct TaskArithmeticMerge {
    pub base_index: usize,
    pub lambda: f32,
    pub weights: Vec<f32>,
}

impl TaskArithmeticMerge {
    pub fn new(lambda: f32) -> Self {
        Self { base_index: 0, lambda, weights: vec![] }
    }

    pub fn with_weights(base_index: usize, lambda: f32, weights: Vec<f32>) -> Self {
        Self { base_index, lambda, weights }
    }
}

impl MergeOp for TaskArithmeticMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("task_arithmetic: no input tensors");
        }
        let base_idx = self.base_index.min(inputs.len() - 1);
        let base = &inputs[base_idx];
        if base.len() != meta.num_elements() {
            anyhow::bail!("task_arithmetic: base tensor size mismatch");
        }
        let mut result = base.clone();
        for (i, ft) in inputs.iter().enumerate() {
            if i == base_idx || ft.len() != base.len() {
                continue;
            }
            let w = self.weights.get(i).copied().unwrap_or(1.0);
            for (r, (b, f)) in result.iter_mut().zip(base.iter().zip(ft.iter())) {
                *r += self.lambda * w * (f - b);
            }
        }
        Ok(result)
    }
}
