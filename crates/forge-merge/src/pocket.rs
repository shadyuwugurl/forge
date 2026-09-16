//! POCKET structured MoE prune: in-domain expert prune (e.g. 256→128) plus
//! MoE-aware mixed-precision planning.
//!
//! Expert scoring reuses the norm signal (experts whose weights carry more
//! energy are kept first); diversity comes from a greedy max-min pass over
//! downsampled expert signatures so the kept set is not a clique of
//! near-duplicates. Mixed precision assigns higher bits to language tags
//! marked sensitive (KO-hard/EN-hard style).

use anyhow::{bail, Result};
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// L2 energy of one expert's weight slice — the keep-first signal.
pub fn expert_score(w: &[f32]) -> f32 {
    w.iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// Score every expert vector.
pub fn score_experts(experts: &[Vec<f32>]) -> Vec<f32> {
    experts.iter().map(|e| expert_score(e)).collect()
}

/// Greedy diverse selection: seed with the highest score, then repeatedly add
/// the expert maximizing `score_share + min_distance_to_selected`.
///
/// `keep` is clamped to `[1, experts.len()]`. Returns kept indices sorted
/// ascending for stable downstream plans.
pub fn greedy_diverse_select(experts: &[Vec<f32>], scores: &[f32], keep: usize) -> Vec<usize> {
    let n = experts.len();
    assert_eq!(n, scores.len(), "experts/scores length mismatch");
    if n == 0 {
        return vec![];
    }
    let keep = keep.clamp(1, n);
    let max_score = scores.iter().cloned().fold(0.0f32, f32::max).max(1e-12);
    // Start from the highest-scoring expert.
    let mut selected = vec![scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()];
    while selected.len() < keep {
        let mut best = None;
        for (i, exp) in experts.iter().enumerate() {
            if selected.contains(&i) {
                continue;
            }
            let mut min_d = f32::INFINITY;
            for &s in &selected {
                min_d = min_d.min(euclid(exp, &experts[s]));
            }
            let norm_d = min_d / (exp.len() as f32).sqrt().max(1.0);
            let value = scores[i] / max_score + norm_d;
            if best.map(|(_, v)| value > v).unwrap_or(true) {
                best = Some((i, value));
            }
        }
        match best {
            Some((i, _)) => selected.push(i),
            None => break,
        }
    }
    selected.sort_unstable();
    selected
}

fn euclid(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut s = 0.0f32;
    for i in 0..n {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

/// MoE-aware mixed-precision plan: `sensitivity` in [0, 1] per language tag
/// lifts bits above `base_bits` (sensitive langs get higher bits).
/// Output bits are clamped to [1, 8].
pub fn mixed_precision_plan(lang_sensitivity: &[(&str, f32)], base_bits: u8) -> Vec<(String, u8)> {
    lang_sensitivity
        .iter()
        .map(|(lang, s)| {
            let bits = (base_bits as f32 + (s.clamp(0.0, 1.0) * 2.0).round()) as u8;
            (lang.to_string(), bits.clamp(1, 8))
        })
        .collect()
}

/// POCKET prune op. Pruning is a surgery-side plan (see `PocketPlan` users);
/// on the merge path this op is an identity copy of parent 0 so
/// `merge --method pocket` degrades to a documented no-op copy.
pub struct PocketPrune {
    pub keep: usize,
}

impl PocketPrune {
    pub fn new(keep: usize) -> Self {
        Self { keep: keep.max(1) }
    }
}

impl MergeOp for PocketPrune {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            bail!("pocket got zero inputs for tensor '{_name}'");
        }
        if inputs[0].len() != meta.num_elements() {
            bail!("pocket length mismatch for tensor '{_name}'");
        }
        Ok(inputs[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::DType;

    fn experts_4d() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.01, 0.01, 0.0, 0.0], // near-duplicate of expert 0
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.1], // weakest
        ]
    }

    #[test]
    fn prune_prefers_diverse_over_duplicates() {
        let experts = experts_4d();
        let scores = score_experts(&experts);
        let kept = greedy_diverse_select(&experts, &scores, 3);
        assert_eq!(kept.len(), 3);
        // Must not keep both 0 and its near-duplicate 1 while dropping a
        // genuinely different direction.
        assert!(!(kept.contains(&0) && kept.contains(&1)));
        // Weakest expert is dropped.
        assert!(!kept.contains(&4));
    }

    #[test]
    fn keep_count_and_bounds() {
        let experts = experts_4d();
        let scores = score_experts(&experts);
        assert_eq!(greedy_diverse_select(&experts, &scores, 2).len(), 2);
        assert_eq!(greedy_diverse_select(&experts, &scores, 99).len(), 5);
        assert_eq!(greedy_diverse_select(&experts, &scores, 0).len(), 1);
    }

    #[test]
    fn mixed_plan_respects_lang_tags() {
        let plan = mixed_precision_plan(&[("ko-hard", 1.0), ("en-hard", 0.8), ("en-easy", 0.0)], 2);
        let bits = |l: &str| plan.iter().find(|(t, _)| t == l).unwrap().1;
        assert!(bits("ko-hard") > bits("en-easy"));
        assert!(bits("en-hard") > bits("en-easy"));
        assert_eq!(bits("en-easy"), 2);
        assert!(plan.iter().all(|(_, b)| (1..=8).contains(b)));
    }

    #[test]
    fn expert_256_keep_128_fixture_shape() {
        // Synthetic 256-expert layer, keep 128: shape-correct end to end.
        let mut experts = Vec::new();
        for i in 0..256 {
            let mut e = vec![0.0f32; 16];
            e[i % 16] = 1.0 + (i as f32) * 0.001;
            e[(i * 7) % 16] += 0.5;
            experts.push(e);
        }
        let scores = score_experts(&experts);
        assert_eq!(scores.len(), 256);
        let kept = greedy_diverse_select(&experts, &scores, 128);
        assert_eq!(kept.len(), 128);
        let mut sorted = kept.clone();
        sorted.sort_unstable();
        assert_eq!(kept, sorted);
    }

    #[test]
    fn passthrough_copies_parent_zero() {
        let p = PocketPrune::new(128);
        let m = TensorMeta { name: "t".into(), shape: vec![3], dtype: DType::F32, offset: 0, size: 12 };
        let out = p.merge_tensors("t", &m, &[vec![1.0, 2.0, 3.0]]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
        assert!(p.merge_tensors("t", &m, &[]).is_err());
    }
}
