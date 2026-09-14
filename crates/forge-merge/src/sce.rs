use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// SCE (Switch-Composition-Estimate / variance-weighted task arithmetic):
/// base + λ · Σᵢ αᵢ·τᵢ, where per-model weight αᵢ ∝ 1/(var(τᵢ)+ε).
/// Models whose task vectors vary wildly (noisy edits) get down-weighted;
/// consistent, low-variance editors dominate. Falls back to uniform weights
/// when all variances are ~0.
pub struct SceMerge {
    pub base_index: usize,
    pub lambda: f32,
    pub epsilon: f32,
}

impl SceMerge {
    pub fn new(lambda: f32) -> Self {
        Self { base_index: 0, lambda, epsilon: 1e-6 }
    }
}

impl MergeOp for SceMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() < 2 {
            anyhow::bail!("sce: need base + ≥1 finetune");
        }
        let base_idx = self.base_index.min(inputs.len() - 1);
        let base = &inputs[base_idx];
        if base.len() != meta.num_elements() {
            anyhow::bail!("sce: base tensor size mismatch");
        }
        // Task vectors + variances
        let mut taus: Vec<Vec<f32>> = Vec::new();
        for (i, ft) in inputs.iter().enumerate() {
            if i == base_idx || ft.len() != base.len() {
                continue;
            }
            taus.push(base.iter().zip(ft.iter()).map(|(b, f)| f - b).collect());
        }
        if taus.is_empty() {
            return Ok(base.clone());
        }
        let mut inv: Vec<f32> = taus
            .iter()
            .map(|t| {
                let m = t.iter().sum::<f32>() / t.len() as f32;
                let v = t.iter().map(|x| (x - m).powi(2)).sum::<f32>() / t.len() as f32;
                1.0 / (v + self.epsilon)
            })
            .collect();
        let sum: f32 = inv.iter().sum();
        if sum < 1e-12 {
            for x in inv.iter_mut() {
                *x = 1.0 / taus.len() as f32;
            }
        } else {
            for x in inv.iter_mut() {
                *x /= sum;
            }
        }
        let mut result = base.clone();
        for (tau, a) in taus.iter().zip(inv.iter()) {
            for (r, d) in result.iter_mut().zip(tau.iter()) {
                *r += self.lambda * a * d;
            }
        }
        Ok(result)
    }
}
