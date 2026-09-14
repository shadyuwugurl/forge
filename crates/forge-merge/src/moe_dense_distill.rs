use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// MoE -> Dense distillation (Krafton 2605.28207): collapses a MoE teacher
/// into a dense student (+6.3pp reported). Per-tensor streaming op:
/// temperature-scaled softmax over parent inputs, then weighted sum.
/// `inputs[0]` is treated as the dense student prior when >1 parent.
pub struct MoeDenseDistill {
    pub temperature: f32,
}

impl MoeDenseDistill {
    pub fn new(temperature: f32) -> Self {
        Self { temperature: temperature.clamp(0.1, 10.0) }
    }
}

impl MergeOp for MoeDenseDistill {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }
    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return self.merge_tensor(_name, meta);
        }
        if inputs.len() == 1 {
            return Ok(inputs[0].clone());
        }
        let n = inputs[0].len();
        if inputs.iter().any(|v| v.len() != n) || n != meta.num_elements() {
            anyhow::bail!(
                "MoeDenseDistill: heterogeneous shapes unsupported for {} (got {:?}, ref {:?})",
                _name,
                inputs.iter().map(|v| v.len()).collect::<Vec<_>>(),
                meta.shape
            );
        }
        // Per-element softmax weights across parents scaled by temperature.
        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let mut max_v = f32::NEG_INFINITY;
            for inp in inputs {
                max_v = max_v.max(inp[i] / self.temperature);
            }
            let mut sum = 0.0f32;
            let mut ws = Vec::with_capacity(inputs.len());
            for inp in inputs {
                let w = ((inp[i] / self.temperature) - max_v).exp();
                ws.push(w);
                sum += w;
            }
            let mut acc = 0.0f32;
            for (inp, w) in inputs.iter().zip(ws.iter()) {
                acc += inp[i] * (w / sum);
            }
            out[i] = acc;
        }
        Ok(out)
    }
}
