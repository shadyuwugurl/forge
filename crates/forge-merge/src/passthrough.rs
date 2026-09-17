use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Passthrough: Direct layer concatenation from different models (frankenmerge)
pub struct PassthroughMerge<'a> {
    pub slices: Vec<(&'a [f32], usize, usize)>,  // (tensor_data, layer_start, layer_end)
}

impl<'a> PassthroughMerge<'a> {
    pub fn new(slices: Vec<(&'a [f32], usize, usize)>) -> Self {
        Self { slices }
    }
}

impl MergeOp for PassthroughMerge<'_> {
    fn merge_tensors(&self, _name: &str, _meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("passthrough: no inputs");
        }
        // Streaming: keep first parent's tensor (layer routing uses merge_tensor path)
        Ok(inputs[0].clone())
    }

    fn merge_tensor(&self, _name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        // Concatenate data from all slices
        let mut result = Vec::new();
        for (data, _start, _end) in &self.slices {
            result.extend_from_slice(data);
        }
        Ok(result)
    }
}
