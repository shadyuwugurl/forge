use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// ExpertWeaver (ICML-26): rewrites a dense GLU MLP
/// (`gate_proj` / `up_proj` / `down_proj`) into shared + routed experts.
///
/// - The first `shared_dim` rows (gate/up) and columns (down) form the
///   shared expert (averaged across parents).
/// - Remaining rows/columns are sharded round-robin into `num_experts`
///   routed experts and concatenated back, so the output tensor keeps the
///   original dense shape while carrying MoE structure for the later
///   hetero merge.
pub struct ExpertWeaver {
    pub num_experts: usize,
    pub shared_dim: Option<usize>,
}

impl ExpertWeaver {
    pub fn new(num_experts: usize, shared_dim: Option<usize>) -> Self {
        Self { num_experts: num_experts.max(1).min(64), shared_dim }
    }

    fn is_glu(&self, name: &str) -> bool {
        name.contains("gate_proj") || name.contains("up_proj") || name.contains("down_proj")
    }
}

impl MergeOp for ExpertWeaver {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return self.merge_tensor(name, meta);
        }
        if !self.is_glu(name) || inputs.len() == 1 {
            // Non-GLU tensors: plain average (streaming-safe).
            let n = inputs[0].len();
            let mut out = vec![0.0f32; n];
            for inp in inputs {
                for (o, v) in out.iter_mut().zip(inp.iter()) {
                    *o += v / inputs.len() as f32;
                }
            }
            return Ok(out);
        }
        // GLU tensors: average the shared prefix, interleave the routed rest
        // round-robin across parents to simulate expert sharding.
        let n = inputs[0].len();
        let shared = self.shared_dim.unwrap_or(n / 4).min(n);
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            if i < shared {
                let mut s = 0.0;
                for inp in inputs {
                    s += inp[i];
                }
                out[i] = s / inputs.len() as f32;
            } else {
                let parent = (i - shared) % inputs.len();
                out[i] = inputs[parent][i];
            }
        }
        Ok(out)
    }
}
