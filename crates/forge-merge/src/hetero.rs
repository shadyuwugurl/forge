//! 0E keystone: heterogeneous merge over canonical slots.
//!
//! Cross-family / cross-shape merges (Qwen 27B dense + 35B MoE) align by
//! [`CanonicalSlot`](forge_core::tensor_map::CanonicalSlot), never by raw
//! layer index. Two coverage modes:
//!
//! - **Union**: every slot any parent provides is emitted (missing parents
//!   simply don't contribute). Maximizes coverage for hetero merges.
//! - **Intersection**: only slots present in *all* parents are emitted.
//!   Strict mode for same-family or validated merges.
//!
//! The merge itself is per-tensor and streaming-safe: the orchestrator loads
//! one tensor per parent store at a time, so we never hold two full >14B
//! models in memory. Slot-presence planning ([`HeteroMerge::plan`]) works on
//! [`TensorMap`](forge_core::tensor_map::TensorMap)s built from names alone
//! — zero weight I/O.

use anyhow::{bail, Result};
use forge_core::tensor_map::TensorMap;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Slot-coverage policy for heterogeneous merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeteroMode {
    /// Emit every slot any parent provides; average the holders present.
    Union,
    /// Emit only slots present in all parents; bail otherwise.
    Intersection,
}

impl HeteroMode {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "union" => Ok(Self::Union),
            "intersection" | "intersect" => Ok(Self::Intersection),
            other => bail!("unknown hetero mode '{other}' (expected union|intersection)"),
        }
    }
}

/// One planned slot: which parents hold it.
#[derive(Debug, Clone)]
pub struct SlotPlan {
    pub slot_key: String,
    pub holders: Vec<usize>,
}

/// Heterogeneous N-parent merge over canonical slots.
pub struct HeteroMerge {
    pub mode: HeteroMode,
    pub n_parents: usize,
    /// Convex combination weights (sums to 1, len == n_parents).
    pub weights: Vec<f32>,
}

impl HeteroMerge {
    pub fn new(mode: HeteroMode, n_parents: usize) -> Self {
        let n = n_parents.max(1);
        let w = 1.0 / n as f32;
        Self { mode, n_parents: n, weights: vec![w; n] }
    }

    pub fn with_weights(mode: HeteroMode, weights: Vec<f32>) -> Result<Self> {
        if weights.len() < 2 {
            bail!("hetero merge needs >= 2 parent weights, got {}", weights.len());
        }
        let sum: f32 = weights.iter().sum();
        if sum <= 1e-8 {
            bail!("hetero weights sum to ~0; cannot normalize");
        }
        let normed: Vec<f32> = weights.iter().map(|w| w / sum).collect();
        Ok(Self { mode, n_parents: normed.len(), weights: normed })
    }

    /// Plan slot coverage over one [`TensorMap`] per parent.
    /// Returns `(slot_key, holder_indices)` filtered by mode, sorted by key.
    pub fn plan(&self, maps: &[TensorMap]) -> Vec<SlotPlan> {
        use std::collections::HashMap;
        let mut holders: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, m) in maps.iter().enumerate() {
            for k in m.slots.keys() {
                holders.entry(k.clone()).or_default().push(i);
            }
        }
        let mut plans: Vec<SlotPlan> = holders
            .into_iter()
            .filter(|(_, h)| match self.mode {
                HeteroMode::Union => !h.is_empty(),
                HeteroMode::Intersection => h.len() == maps.len(),
            })
            .map(|(slot_key, holders)| SlotPlan { slot_key, holders })
            .collect();
        plans.sort_by(|a, b| a.slot_key.cmp(&b.slot_key));
        plans
    }

    /// Weighted convex combination over all parents.
    /// Requires exactly `n_parents` inputs of equal length.
    pub fn merge(&self, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() != self.n_parents {
            bail!(
                "hetero merge expects {} parent inputs, got {}",
                self.n_parents,
                inputs.len()
            );
        }
        let n = inputs[0].len();
        if inputs.iter().any(|v| v.len() != n) {
            bail!("hetero merge needs equal-length inputs for weighted combo");
        }
        let mut out = vec![0.0f32; n];
        for (inp, w) in inputs.iter().zip(self.weights.iter()) {
            for (a, v) in out.iter_mut().zip(inp.iter()) {
                *a += w * v;
            }
        }
        Ok(out)
    }
}

impl MergeOp for HeteroMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            bail!("hetero merge got zero inputs for tensor '{_name}'");
        }
        match self.mode {
            HeteroMode::Intersection => {
                // Strict: every parent must hold this slot. The orchestrator
                // passes one value per parent store, so a short list means a
                // parent is missing the slot.
                if inputs.len() != self.n_parents {
                    bail!(
                        "intersection: tensor '{_name}' held by {}/{} parents",
                        inputs.len(),
                        self.n_parents
                    );
                }
                self.merge(inputs)
            }
            HeteroMode::Union => {
                // Lenient: average whatever holders are present (uniform —
                // holder identities aren't visible at this level; use
                // `merge` with explicit weights when they are).
                let n = inputs[0].len();
                if inputs.iter().any(|v| v.len() != n) || n != meta.num_elements() {
                    bail!("union: heterogeneous lengths for tensor '{_name}' need slot planning");
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::introspect::{FamilyRegistry, ModelProfile};

    fn hetero_setup() -> (FamilyRegistry, ModelProfile) {
        let reg = FamilyRegistry::builtin();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 64,
            "hidden_size": 5120,
        });
        let tensors = vec![(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![5120, 5120],
        )];
        let p = ModelProfile::detect_from_parts(&config, &tensors, &reg).unwrap();
        (reg, p)
    }

    #[test]
    fn union_averages_present_inputs() {
        let h = HeteroMerge::new(HeteroMode::Union, 2);
        let meta = TensorMeta {
            name: "t".into(),
            shape: vec![2],
            dtype: forge_core::DType::F32,
            offset: 0,
            size: 8,
        };
        let out = h
            .merge_tensors("t", &meta, &[vec![1.0, 3.0], vec![3.0, 5.0]])
            .unwrap();
        assert_eq!(out, vec![2.0, 4.0]);
    }

    #[test]
    fn intersection_bails_when_parent_missing() {
        let h = HeteroMerge::new(HeteroMode::Intersection, 2);
        let meta = TensorMeta {
            name: "t".into(),
            shape: vec![2],
            dtype: forge_core::DType::F32,
            offset: 0,
            size: 8,
        };
        assert!(h.merge_tensors("t", &meta, &[vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn weighted_merge_honors_weights() {
        let h = HeteroMerge::with_weights(HeteroMode::Intersection, vec![0.25, 0.75]).unwrap();
        let out = h.merge(&[vec![4.0], vec![8.0]]).unwrap();
        assert!((out[0] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn with_weights_rejects_bad_input() {
        assert!(HeteroMerge::with_weights(HeteroMode::Union, vec![1.0]).is_err());
        assert!(HeteroMerge::with_weights(HeteroMode::Union, vec![0.0, 0.0]).is_err());
    }

    #[test]
    fn plan_union_vs_intersection() {
        let (reg, p) = hetero_setup();
        let a = TensorMap::build(
            &p,
            &[
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            &reg,
        )
        .unwrap();
        let b = TensorMap::build(&p, &["model.layers.0.self_attn.q_proj.weight".to_string()], &reg)
            .unwrap();
        let u = HeteroMerge::new(HeteroMode::Union, 2).plan(&[a.clone(), b.clone()]);
        assert_eq!(u.len(), 2);
        let head = u.iter().find(|s| s.slot_key == "global.lm_head").unwrap();
        assert_eq!(head.holders, vec![0]);
        let i = HeteroMerge::new(HeteroMode::Intersection, 2).plan(&[a, b]);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0].slot_key, "b0.attn.q_proj");
        assert_eq!(i[0].holders, vec![0, 1]);
    }

    #[test]
    fn empty_inputs_bail() {
        let h = HeteroMerge::new(HeteroMode::Union, 2);
        let meta = TensorMeta {
            name: "t".into(),
            shape: vec![2],
            dtype: forge_core::DType::F32,
            offset: 0,
            size: 8,
        };
        assert!(h.merge_tensors("t", &meta, &[]).is_err());
    }
}
