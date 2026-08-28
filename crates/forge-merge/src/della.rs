use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// DELLA: Adaptive magnitude-based pruning + merge
pub struct DellaMerge<'a> {
    pub base: &'a [f32],
    pub models: Vec<(&'a [f32], f32, f32, f32)>,  // (tensor, weight, density, epsilon)
    pub seed: u64,
}

impl<'a> DellaMerge<'a> {
    pub fn new(base: &'a [f32], models: Vec<(&'a [f32], f32, f32, f32)>, seed: u64) -> Self {
        Self { base, models, seed }
    }
}

impl MergeOp for DellaMerge<'_> {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let len = self.base.len();
        let mut result = self.base.to_vec();

        let total_weight: f32 = self.models.iter().map(|(_, w, _, _)| w).sum();

        for (model, weight, density, epsilon) in &self.models {
            let delta: Vec<f32> = model.iter().zip(self.base.iter())
                .map(|(m, b)| m - b)
                .collect();

            // Compute magnitude threshold per tensor
            let mut magnitudes: Vec<f32> = delta.iter().map(|x| x.abs()).collect();
            magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let keep_count = (len as f32 * density).ceil() as usize;
            let mag_threshold = magnitudes.get(len.saturating_sub(keep_count))
                .copied()
                .unwrap_or(0.0);

            // Adaptive pruning: keep values above magnitude threshold
            // AND within epsilon of the maximum magnitude
            let max_mag = delta.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            for i in 0..len {
                let mag = delta[i].abs();
                if mag >= mag_threshold && mag >= max_mag * (1.0 - epsilon) {
                    result[i] += delta[i] * weight / total_weight;
                }
            }
        }

        Ok(result)
    }
}
