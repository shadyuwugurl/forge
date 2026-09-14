use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::slerp_utils::normalize_weights;

/// RAM (Randomized-weight Averaging Merge): stochastic linear merge.
/// Samples Dirichlet(1,…,1)-ish weights (via normalized exponential noise)
/// around the base weights, then returns the weighted average. Re-running
/// with a different seed explores nearby points on the simplex — useful for
/// evolutionary/population-based search over merge ratios.
pub struct RamMerge {
    pub seed: u64,
    pub weights: Vec<f32>,
    /// Noise scale applied to base weights before renormalization (0 = deterministic)
    pub temperature: f32,
}

impl RamMerge {
    pub fn new(seed: u64) -> Self {
        Self { seed, weights: vec![], temperature: 1.0 }
    }
}

fn next_f32(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
    (*state >> 33) as f32 / (1u32 << 31) as f32
}

impl MergeOp for RamMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("ram: no input tensors");
        }
        let n = meta.num_elements();
        for (i, t) in inputs.iter().enumerate() {
            if t.len() != n {
                anyhow::bail!("ram: input {} size mismatch", i);
            }
        }
        // Deterministic per-tensor seed so every tensor draws independently
        let mut state = self.seed.wrapping_add(name.len() as u64 * 0x9E3779B9);
        let base = normalize_weights(&self.weights, inputs.len());
        let mut w: Vec<f32> = base
            .iter()
            .map(|b| {
                let u = next_f32(&mut state).max(1e-6);
                (b.max(1e-6) + self.temperature * (-u.ln())).max(1e-9)
            })
            .collect();
        let s: f32 = w.iter().sum();
        for x in w.iter_mut() {
            *x /= s;
        }
        let mut result = vec![0.0f32; n];
        for (t, x) in inputs.iter().zip(w.iter()) {
            for (r, v) in result.iter_mut().zip(t.iter()) {
                *r += x * v;
            }
        }
        Ok(result)
    }
}
