//! Aether-7B-5Attn Latin-square MoE layer remap.
//!
//! The Aether family arranges 5 heterogeneous attention types on a 7x7 Latin
//! square over 49 layers (see [`AetherLayout`](forge_core::tensor_map::AetherLayout)).
//! This op carries the grid-correct remap plan; on the weight path it is a
//! uniform average over equal-length parent inputs (attn-type alignment is
//! enforced at slot-plan time via `AetherLayout::align_ok`, never here).

use anyhow::{bail, Result};
use forge_core::tensor_map::AetherLayout;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Grid-correct layer-remap merge for Aether-family checkpoints.
pub struct AetherRemap {
    pub layout: AetherLayout,
}

impl AetherRemap {
    pub fn new(grid: usize) -> Self {
        Self { layout: AetherLayout { grid: grid.max(1), num_layers: grid * grid } }
    }

    pub fn aether49() -> Self {
        Self { layout: AetherLayout::aether49() }
    }

    /// `(layer, attn_type_idx)` remap plan over the layout's layers.
    pub fn remap_plan(&self) -> Vec<(usize, usize)> {
        self.layout.slot_map()
    }
}

impl MergeOp for AetherRemap {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            bail!("aether remap got zero inputs for tensor '{_name}'");
        }
        let n = inputs[0].len();
        if inputs.iter().any(|v| v.len() != n) || n != meta.num_elements() {
            bail!("aether remap needs equal-length inputs for tensor '{_name}'");
        }
        let mut out = vec![0.0f32; n];
        for inp in inputs {
            for (a, v) in out.iter_mut().zip(inp.iter()) {
                *a += v;
            }
        }
        let k = inputs.len() as f32;
        for a in out.iter_mut() {
            *a /= k;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::DType;

    fn meta(n: usize) -> TensorMeta {
        TensorMeta { name: "t".into(), shape: vec![n], dtype: DType::F32, offset: 0, size: n * 4 }
    }

    #[test]
    fn remap_covers_all_49_layers() {
        let r = AetherRemap::aether49();
        let plan = r.remap_plan();
        assert_eq!(plan.len(), 49);
        let mut layers: Vec<usize> = plan.iter().map(|(l, _)| *l).collect();
        layers.sort_unstable();
        assert_eq!(layers, (0..49).collect::<Vec<_>>());
    }

    #[test]
    fn remap_uses_all_five_attn_types() {
        let r = AetherRemap::aether49();
        let mut seen = std::collections::HashSet::new();
        for (_, t) in r.remap_plan() {
            seen.insert(t);
        }
        assert_eq!(seen.len(), 5);
    }

    #[test]
    fn average_math_is_uniform() {
        let r = AetherRemap::aether49();
        let out = r
            .merge_tensors("t", &meta(2), &[vec![1.0, 3.0], vec![3.0, 5.0]])
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn mismatched_lengths_bail() {
        let r = AetherRemap::aether49();
        assert!(r.merge_tensors("t", &meta(2), &[vec![1.0, 2.0], vec![1.0]]).is_err());
        assert!(r.merge_tensors("t", &meta(2), &[]).is_err());
    }
}
