use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// FrankenMerge: Layer stacking with dimension adaptation
/// Supports merging models with different hidden sizes by padding/projection
pub struct FrankenMerge<'a> {
    pub slices: Vec<FrankenSlice<'a>>,
    pub dimension_adapter: DimensionAdapter,
}

pub struct FrankenSlice<'a> {
    pub data: &'a [f32],
    pub original_shape: Vec<usize>,
    pub target_shape: Vec<usize>,
}

pub enum DimensionAdapter {
    /// Zero-pad smaller tensors to match larger ones
    ZeroPad,
    /// Scale weights proportionally
    Scale,
    /// Skip incompatible tensors
    Skip,
}

impl<'a> FrankenMerge<'a> {
    pub fn new(slices: Vec<FrankenSlice<'a>>, adapter: DimensionAdapter) -> Self {
        Self { slices, dimension_adapter: adapter }
    }
}

impl MergeOp for FrankenMerge<'_> {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut result = Vec::new();

        for slice in &self.slices {
            match &self.dimension_adapter {
                DimensionAdapter::ZeroPad => {
                    // Pad slice data to target shape
                    let target_elements: usize = slice.target_shape.iter().product();
                    let source_elements = slice.data.len();

                    if source_elements <= target_elements {
                        let mut padded = slice.data.to_vec();
                        padded.resize(target_elements, 0.0);
                        result.extend_from_slice(&padded);
                    } else {
                        // Truncate if source is larger
                        result.extend_from_slice(&slice.data[..target_elements]);
                    }
                }
                DimensionAdapter::Scale => {
                    let target_elements: usize = slice.target_shape.iter().product();
                    let source_elements = slice.data.len();

                    if source_elements != target_elements {
                        let scale = (target_elements as f64 / source_elements as f64).sqrt() as f32;
                        for &val in slice.data.iter() {
                            result.push(val * scale);
                        }
                    } else {
                        result.extend_from_slice(slice.data);
                    }
                }
                DimensionAdapter::Skip => {
                    if slice.original_shape == slice.target_shape {
                        result.extend_from_slice(slice.data);
                    }
                    // Skip if dimensions don't match
                }
            }
        }

        Ok(result)
    }
}
