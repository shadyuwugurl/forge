use anyhow::Result;
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
