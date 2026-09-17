use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// DARE: Random dropout + rescaled linear merge
pub struct DareMerge<'a> {
    pub base: &'a [f32],
    pub models: Vec<(&'a [f32], f32, f32)>,  // (tensor, weight, density)
    pub seed: u64,
}

impl<'a> DareMerge<'a> {
    pub fn new(base: &'a [f32], models: Vec<(&'a [f32], f32, f32)>, seed: u64) -> Self {
        Self { base, models, seed }
    }
}

impl MergeOp for DareMerge<'_> {
    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("dare: no inputs");
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }
        let base = &inputs[0];
        let n = base.len();
        let mut out = base.clone();
        let total = (inputs.len() - 1) as f32;
        let mut rng = self.seed.wrapping_add(meta.name.len() as u64);
        let density = 0.5f32;
        let scale = 1.0 / density;
        for model in inputs[1..].iter() {
            if model.len() != n {
                continue;
            }
            for i in 0..n {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let r = (rng >> 33) as f32 / (1u32 << 31) as f32;
                if r < density {
                    out[i] += (model[i] - base[i]) * scale / total;
                }
            }
        }
        Ok(out)
    }

    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let len = self.base.len();
        let mut result = self.base.to_vec();

        let total_weight: f32 = self.models.iter().map(|(_, w, _)| w).sum();

        // Simple LCG pseudo-random for dropout
        let mut rng_state = self.seed.wrapping_add(meta.name.len() as u64);

        for (model, weight, density) in &self.models {
            let delta: Vec<f32> = model.iter().zip(self.base.iter())
                .map(|(m, b)| m - b)
                .collect();

            // Random dropout + rescale
            let keep_prob = *density;
            let scale = 1.0 / keep_prob;

            for i in 0..len {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let random_val = (rng_state >> 33) as f32 / (1u32 << 31) as f32;

                if random_val < keep_prob {
                    result[i] += delta[i] * weight * scale / total_weight;
                }
            }
        }

        Ok(result)
    }
}
