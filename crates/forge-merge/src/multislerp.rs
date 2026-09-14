use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::slerp_utils::{normalize_weights, slerp_pair};

/// Multi-SLERP: barycentric spherical interpolation over N models.
/// Sequentially folds each model into the running barycenter with its
/// normalized weight: acc = slerp(acc, mᵢ, wᵢ / Σ_{j≤i} wⱼ).
pub struct MultiSlerpMerge {
    pub weights: Vec<f32>,
}

impl MultiSlerpMerge {
    pub fn new(weights: Vec<f32>) -> Self {
        Self { weights }
    }
}

impl MergeOp for MultiSlerpMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("multislerp: no input tensors");
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }
        let n = meta.num_elements();
        for (i, t) in inputs.iter().enumerate() {
            if t.len() != n {
                anyhow::bail!("multislerp: input {} size mismatch", i);
            }
        }
        let w = normalize_weights(&self.weights, inputs.len());
        let mut acc = inputs[0].clone();
        let mut cum = w[0];
        for (i, t) in inputs.iter().enumerate().skip(1) {
            cum += w[i];
            acc = slerp_pair(&acc, t, w[i] / cum)?;
        }
        Ok(acc)
    }
}
