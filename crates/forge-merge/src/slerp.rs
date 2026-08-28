use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Spherical linear interpolation between 2 models
pub struct SlerpMerge<'a> {
    pub model_a: &'a [f32],
    pub model_b: &'a [f32],
    pub t: f32,  // interpolation factor [0, 1]
}

impl<'a> SlerpMerge<'a> {
    pub fn new(model_a: &'a [f32], model_b: &'a [f32], t: f32) -> Self {
        Self { model_a, model_b, t }
    }
}

impl MergeOp for SlerpMerge<'_> {
    fn merge_tensor(&self, _name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        let len = self.model_a.len();
        let mut result = vec![0.0f32; len];

        // Compute cosine similarity
        let dot: f32 = self.model_a.iter().zip(self.model_b.iter())
            .map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.model_a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = self.model_b.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cos_theta = (dot / (norm_a * norm_b + 1e-8)).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();

        if theta.abs() < 1e-6 {
            // Vectors nearly parallel — fall back to lerp
            for i in 0..len {
                result[i] = (1.0 - self.t) * self.model_a[i] + self.t * self.model_b[i];
            }
        } else {
            let sin_theta = theta.sin();
            let w_a = ((1.0 - self.t) * theta).sin() / sin_theta;
            let w_b = (self.t * theta).sin() / sin_theta;

            for i in 0..len {
                result[i] = w_a * self.model_a[i] + w_b * self.model_b[i];
            }
        }

        Ok(result)
    }
}
