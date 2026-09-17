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
    fn merge_tensors(&self, _name: &str, _meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("della: no inputs");
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }
        let base = &inputs[0];
        let n = base.len();
        let mut out = base.clone();
        let total = (inputs.len() - 1) as f32;
        for model in inputs[1..].iter() {
            if model.len() != n {
                continue;
            }
            let delta: Vec<f32> = model.iter().zip(base.iter()).map(|(m, b)| m - b).collect();
            let mut mags: Vec<f32> = delta.iter().map(|x| x.abs()).collect();
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let keep = ((n as f32 * 0.5).ceil() as usize).max(1);
            let thresh = mags.get(n.saturating_sub(keep)).copied().unwrap_or(0.0);
            for i in 0..n {
                if delta[i].abs() >= thresh {
                    out[i] += delta[i] / total;
                }
            }
        }
        Ok(out)
    }

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
