use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::slerp_utils::slerp_pair;

/// NuSLERP: SLERP with explicit per-model weighting on task vectors.
/// result = base + slerp direction scaled by t (mergekit nuslerp semantics:
/// interpolate the task vector along the spherical path).
pub struct NuSlerpMerge {
    pub base_index: usize,
    pub t: f32,
}

impl NuSlerpMerge {
    pub fn new(t: f32) -> Self {
        Self { base_index: 0, t }
    }
}

impl MergeOp for NuSlerpMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() < 2 {
            anyhow::bail!("nuslerp: need ≥2 models");
        }
        let base = &inputs[self.base_index.min(inputs.len() - 1)];
        if base.len() != meta.num_elements() {
            anyhow::bail!("nuslerp: tensor size mismatch");
        }
        // Sequential pairwise SLERP of task vectors around base
        let mut acc = vec![0.0f32; base.len()];
        let mut n = 0usize;
        for (i, ft) in inputs.iter().enumerate() {
            if i == self.base_index || ft.len() != base.len() {
                continue;
            }
            let dir: Vec<f32> = base.iter().zip(ft.iter()).map(|(b, f)| f - b).collect();
            for (a, d) in acc.iter_mut().zip(dir.iter()) {
                *a += d;
            }
            n += 1;
        }
        if n == 0 {
            return Ok(base.clone());
        }
        for a in acc.iter_mut() {
            *a /= n as f32;
        }
        // Spherical interpolation between zero-vector direction and mean task vector
        let target: Vec<f32> = base.iter().zip(acc.iter()).map(|(b, d)| b + d).collect();
        slerp_pair(base, &target, self.t)
    }
}
