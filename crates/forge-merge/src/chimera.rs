//! Chimera cross-architecture merge: compat-score + freeze-parents + train-router-only.
//!
//! Per-tensor decision over two parents:
//!
//! - **Average (crossbreed)**: same shape and `score_compat >= threshold` —
//!   convex combination of the parents.
//! - **Transplant**: shape mismatch or low compat — whole-tensor copy from
//!   parent 0 (typically a whole-FFN/expert block graft).
//!
//! Both parents stay frozen; the only trainable part is the per-question
//! router ([`ChimeraRouter`]): small logits over parents added to the
//! compat scores, trained with a softmax cross-entropy step.

use anyhow::{bail, Result};
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Per-tensor merge decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeDecision {
    Average,
    Transplant,
}

/// Compat score in [0, 1] between two equal- or unequal-length weight slices.
///
/// Blends cosine alignment with mean/variance closeness so that a shifted or
/// rescaled copy still scores above an unrelated tensor. Identical inputs
/// score exactly 1.0.
pub fn score_compat(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len().min(b.len()) as f32;
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    let mut ma = 0.0f32;
    let mut mb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    let cos = if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        (dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0)
    };
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        va += (a[i] - ma) * (a[i] - ma);
        vb += (b[i] - mb) * (b[i] - mb);
    }
    va /= n;
    vb /= n;
    let mean_close = 1.0 / (1.0 + (ma - mb).abs() / (1.0 + ma.abs() + mb.abs()));
    let var_close = 1.0 / (1.0 + (va - vb).abs() / (1.0 + va + vb));
    // Length mismatch is itself incompatibility evidence.
    let len_factor = if a.len() == b.len() {
        1.0
    } else {
        (a.len().min(b.len()) as f32 / a.len().max(b.len()) as f32).sqrt()
    };
    len_factor * ((cos + 1.0) / 2.0 * 0.5 + mean_close * 0.25 + var_close * 0.25)
}

/// Route a compat score through the threshold.
pub fn decide(score: f32, threshold: f32) -> MergeDecision {
    if score >= threshold {
        MergeDecision::Average
    } else {
        MergeDecision::Transplant
    }
}

/// Compat-gated two-parent merge op.
pub struct ChimeraMerge {
    pub threshold: f32,
}

impl ChimeraMerge {
    pub fn new(threshold: f32) -> Self {
        Self { threshold: threshold.clamp(0.0, 1.0) }
    }

    /// Score a parent pair and return the routing decision.
    pub fn decide_tensors(&self, a: &[f32], b: &[f32]) -> (f32, MergeDecision) {
        let s = score_compat(a, b);
        (s, decide(s, self.threshold))
    }
}

impl MergeOp for ChimeraMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            bail!("chimera merge got zero inputs for tensor '{_name}'");
        }
        if inputs.len() == 1 || inputs[0].len() != inputs.get(1).map(|v| v.len()).unwrap_or(inputs[0].len()) {
            // Single parent or shape mismatch vs parent 1 → transplant parent 0.
            if inputs[0].len() != meta.num_elements() {
                bail!("chimera transplant length mismatch for tensor '{_name}'");
            }
            return Ok(inputs[0].clone());
        }
        let (_score, decision) = self.decide_tensors(&inputs[0], &inputs[1]);
        match decision {
            MergeDecision::Transplant => Ok(inputs[0].clone()),
            MergeDecision::Average => {
                let n = inputs[0].len();
                let mut out = vec![0.0f32; n];
                for inp in inputs.iter().take(2) {
                    for (a, v) in out.iter_mut().zip(inp.iter()) {
                        *a += 0.5 * v;
                    }
                }
                Ok(out)
            }
        }
    }
}

/// Frozen-parent per-question router: the ONLY trainable part of Chimera.
///
/// `forward` adds the learned logits to the per-parent compat scores and
/// returns a softmax distribution; `pick` returns the argmax parent.
#[derive(Debug, Clone)]
pub struct ChimeraRouter {
    pub logits: Vec<f32>,
}

impl ChimeraRouter {
    pub fn new(n_parents: usize) -> Self {
        Self { logits: vec![0.0; n_parents.max(1)] }
    }

    pub fn forward(&self, compat: &[f32]) -> Vec<f32> {
        let n = self.logits.len();
        let mut max_v = f32::NEG_INFINITY;
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let c = compat.get(i).copied().unwrap_or(0.0);
            let v = self.logits[i] + c;
            z.push(v);
            max_v = max_v.max(v);
        }
        let mut sum = 0.0f32;
        for v in z.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        for v in z.iter_mut() {
            *v /= sum.max(1e-12);
        }
        z
    }

    pub fn pick(&self, compat: &[f32]) -> usize {
        let p = self.forward(compat);
        let mut best = 0;
        for (i, v) in p.iter().enumerate() {
            if *v > p[best] {
                best = i;
            }
        }
        best
    }

    /// One softmax cross-entropy gradient step on the logits toward `target`.
    /// Returns the loss before the update.
    pub fn train_step(&mut self, compat: &[f32], target: usize, lr: f32) -> f32 {
        assert!(target < self.logits.len(), "router target out of range");
        let p = self.forward(compat);
        let loss = -p[target].max(1e-12).ln();
        for (i, l) in self.logits.iter_mut().enumerate() {
            let grad = p[i] - if i == target { 1.0 } else { 0.0 };
            *l -= lr * grad;
        }
        loss
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
    fn identical_tensors_score_one_and_average() {
        let a = vec![1.0, -2.0, 3.0, 0.5];
        assert!((score_compat(&a, &a) - 1.0).abs() < 1e-6);
        let m = ChimeraMerge::new(0.5);
        let out = m.merge_tensors("t", &meta(4), &[a.clone(), a.clone()]).unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn incompatible_shapes_transplant_parent_zero() {
        let m = ChimeraMerge::new(0.0); // even zero threshold can't avg mismatched lens
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![9.0, 9.0];
        let out = m.merge_tensors("t", &meta(4), &[a.clone(), b]).unwrap();
        assert_eq!(out, a);
    }

    #[test]
    fn low_compat_transplants_high_compat_averages() {
        let m = ChimeraMerge::new(0.9);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let neg: Vec<f32> = a.iter().map(|v| -v).collect();
        let (s, d) = m.decide_tensors(&a, &neg);
        assert!(s < 0.9);
        assert_eq!(d, MergeDecision::Transplant);
        let close: Vec<f32> = a.iter().map(|v| v * 1.01).collect();
        let (s2, d2) = m.decide_tensors(&a, &close);
        assert!(s2 >= 0.9);
        assert_eq!(d2, MergeDecision::Average);
    }

    #[test]
    fn router_trains_to_pick_correct_parent() {
        let mut r = ChimeraRouter::new(2);
        // Compat favors parent 0, but the label says parent 1.
        let compat = vec![0.9, 0.1];
        assert_eq!(r.pick(&compat), 0);
        let mut loss = f32::INFINITY;
        for _ in 0..200 {
            loss = r.train_step(&compat, 1, 0.5);
        }
        assert_eq!(r.pick(&compat), 1);
        assert!(loss < 0.1);
    }

    #[test]
    fn router_forward_is_valid_distribution() {
        let r = ChimeraRouter::new(3);
        let p = r.forward(&[0.2, 0.5, 0.3]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p.iter().all(|v| *v > 0.0));
    }

    #[test]
    fn empty_inputs_bail() {
        let m = ChimeraMerge::new(0.5);
        assert!(m.merge_tensors("t", &meta(2), &[]).is_err());
        assert_eq!(score_compat(&[], &[]), 0.0);
    }
}
