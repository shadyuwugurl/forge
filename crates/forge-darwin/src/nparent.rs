//! 0C — N-parent Darwin genome + MRI-Trust N-way fusion + slot-based ratios.
//!
//! The legacy [`crate::genome::DarwinGenome`] is a 2-parent genome: one
//! scalar ratio per component/block blended as `(1-r)*A + r*B`. This module
//! generalizes that to N parents:
//!
//! - [`NparentGenome`]: per-block weight rows over N parents, each row kept
//!   on the simplex (weights sum to 1). Constructed via
//!   [`NparentGenome::new_nparent`].
//! - [`crate::mri_trust::MriTrustFusion`] N-way fusion: `final_ratio_nway`
//!   blends an MRI-derived simplex (per-parent static scores) with the
//!   genome row via `tau`, mirroring the 2-parent
//!   `r_final = tau * r_MRI + (1 - tau) * r_genome` formula.
//! - Slot-based ratios: [`crate::genome::DarwinGenome::slot_ratio`] replaces
//!   the hardcoded `tensor_name.contains(...)` branches in `tensor_ratio`
//!   with canonical-slot role lookups (see `forge_core::tensor_map`).
//!   Canonical slots, NOT layer indices or raw names, drive the genome.

use forge_core::tensor_map::CanonicalSlot;

use crate::mri_trust::MriTrustFusion;
use crate::genome::DarwinGenome;

/// Genome for N-parent merges: one simplex weight row per block.
///
/// `weights[block][parent]`; every row sums to 1. Uniform rows mean "plain
/// average of all parents in that block".
#[derive(Debug, Clone)]
pub struct NparentGenome {
    weights: Vec<Vec<f32>>,
    /// MRI-vs-genome trust, same semantics as [`DarwinGenome::tau`].
    pub tau: f32,
}

impl NparentGenome {
    /// Create a uniform N-parent genome: `n_blocks` rows over `n_parents`.
    pub fn new_nparent(n_parents: usize, n_blocks: usize) -> Self {
        assert!(n_parents >= 2, "N-parent merge needs at least 2 parents");
        assert!(n_blocks >= 1, "genome needs at least 1 block");
        let uniform = vec![1.0 / n_parents as f32; n_parents];
        Self {
            weights: vec![uniform; n_blocks],
            tau: 0.5,
        }
    }

    /// Randomized genome from an LCG seed (matches [`DarwinGenome::random`]).
    pub fn random(seed: u64, n_parents: usize, n_blocks: usize) -> Self {
        assert!(n_parents >= 2, "N-parent merge needs at least 2 parents");
        assert!(n_blocks >= 1, "genome needs at least 1 block");
        let mut rng_state = seed;
        let mut next_f32 = |min: f32, max: f32| -> f32 {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let t = (rng_state >> 33) as f32 / (1u32 << 31) as f32;
            min + t * (max - min)
        };
        let mut weights = Vec::with_capacity(n_blocks);
        for _ in 0..n_blocks {
            let mut row: Vec<f32> = (0..n_parents).map(|_| next_f32(0.0, 1.0)).collect();
            simplex_normalize(&mut row);
            weights.push(row);
        }
        Self {
            weights,
            tau: next_f32(0.3, 0.6),
        }
    }

    pub fn n_parents(&self) -> usize {
        self.weights.first().map(Vec::len).unwrap_or(0)
    }

    pub fn n_blocks(&self) -> usize {
        self.weights.len()
    }

    /// Weight of `parent` in `block` (clamped indices).
    pub fn weight_for(&self, parent: usize, block: usize) -> f32 {
        let b = block.min(self.n_blocks().saturating_sub(1));
        let p = parent.min(self.n_parents().saturating_sub(1));
        self.weights[b][p]
    }

    /// Full weight row for `block` (sums to 1).
    pub fn row(&self, block: usize) -> &[f32] {
        let b = block.min(self.n_blocks().saturating_sub(1));
        &self.weights[b]
    }

    /// Uniform crossover: each row comes from one parent genome.
    pub fn crossover(a: &Self, b: &Self, rng_state: &mut u64) -> Self {
        assert_eq!(a.n_parents(), b.n_parents());
        assert_eq!(a.n_blocks(), b.n_blocks());
        let mut next_bool = || -> bool {
            *rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (*rng_state >> 33) & 1 == 0
        };
        let weights = a
            .weights
            .iter()
            .zip(b.weights.iter())
            .map(|(ra, rb)| if next_bool() { ra.clone() } else { rb.clone() })
            .collect();
        Self {
            weights,
            tau: if next_bool() { a.tau } else { b.tau },
        }
    }

    /// Gaussian-ish perturbation, rows renormalized to the simplex.
    pub fn mutate(&self, rate: f32, rng_state: &mut u64) -> Self {
        let mut next_f32 = || -> f32 {
            *rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            (*rng_state >> 33) as f32 / (1u32 << 31) as f32
        };
        let weights = self
            .weights
            .iter()
            .map(|row| {
                let mut out: Vec<f32> = row
                    .iter()
                    .map(|&w| {
                        if next_f32() < rate {
                            (w + (next_f32() - 0.5) * 0.2).max(0.0)
                        } else {
                            w
                        }
                    })
                    .collect();
                simplex_normalize(&mut out);
                out
            })
            .collect();
        let tau = if next_f32() < rate {
            (self.tau + (next_f32() - 0.5) * 0.1).clamp(0.1, 0.9)
        } else {
            self.tau
        };
        Self { weights, tau }
    }

    /// Blend N parent tensors for one slot: convex combination with the
    /// fused N-way ratio (MRI simplex + genome row via `tau`).
    ///
    /// `columns[p]` is parent `p`'s data; all columns must share length —
    /// the blend runs over the shortest column and extra tails are ignored.
    pub fn blend_slot(
        &self,
        block: usize,
        columns: &[&[f32]],
        mri_scores: &[f32],
        mri: &MriTrustFusion,
        slot_key: &str,
    ) -> Vec<f32> {
        let n = self.n_parents().min(columns.len()).min(mri_scores.len());
        if n == 0 {
            return Vec::new();
        }
        let ratios = mri.final_ratio_nway(slot_key, &mri_scores[..n], &self.row(block)[..n], self.tau);
        let len = columns[..n].iter().map(|c| c.len()).min().unwrap_or(0);
        let mut out = vec![0.0f32; len];
        for (p, col) in columns[..n].iter().enumerate() {
            let w = ratios[p];
            for (i, o) in out.iter_mut().enumerate() {
                *o += w * col[i];
            }
        }
        out
    }
}

/// Renormalize a row onto the simplex; falls back to uniform on degeneracy.
fn simplex_normalize(row: &mut [f32]) {
    let sum: f32 = row.iter().sum();
    if sum > 1e-8 {
        for w in row.iter_mut() {
            *w /= sum;
        }
    } else {
        let u = 1.0 / row.len().max(1) as f32;
        for w in row.iter_mut() {
            *w = u;
        }
    }
}

/// N-way MRI-Trust fusion (same crate, so the extension lives here next to
/// the N-parent genome instead of touching `mri_trust.rs`).
impl MriTrustFusion {
    /// Fuse per-parent MRI scores with a genome weight row.
    ///
    /// `r_MRI` is the score simplex (`scores / sum`); the result is
    /// `tau * r_MRI + (1 - tau) * genome`, renormalized. Mirrors the
    /// 2-parent `final_ratio = tau * r_MRI + (1 - tau) * r_genome`.
    pub fn final_ratio_nway(
        &self,
        _slot_key: &str,
        scores: &[f32],
        genome: &[f32],
        tau: f32,
    ) -> Vec<f32> {
        let n = scores.len().min(genome.len());
        if n == 0 {
            return Vec::new();
        }
        let sum: f32 = scores[..n].iter().sum();
        let mut out: Vec<f32> = (0..n)
            .map(|i| {
                let r_mri = if sum > 1e-8 {
                    scores[i] / sum
                } else {
                    1.0 / n as f32
                };
                tau * r_mri + (1.0 - tau) * genome[i]
            })
            .collect();
        simplex_normalize(&mut out);
        out
    }
}

/// Slot-driven ratios: canonical slots supersede the hardcoded
/// `tensor_name.contains(...)` branches in [`DarwinGenome::tensor_ratio`].
impl DarwinGenome {
    /// Ratio for a canonical slot. Role selects the component alpha
    /// (attn / ffn / embed, else 0.5); `slot.block` selects the depth
    /// block `r[]` exactly like `tensor_ratio`'s layer-index path.
    pub fn slot_ratio(&self, slot: &CanonicalSlot, total_blocks: usize) -> f32 {
        let role = slot.role.as_str();
        let component_ratio = if role.contains("attn")
            || role.contains("q_proj")
            || role.contains("k_proj")
            || role.contains("v_proj")
            || role.contains("o_proj")
            || role.contains("qkv")
            || role.contains("norm") && role.contains("attn")
        {
            self.alpha_attn
        } else if role.contains("ffn")
            || role.contains("mlp")
            || role.contains("gate")
            || role.contains("up_proj")
            || role.contains("down_proj")
            || role.contains("moe")
            || role.contains("expert")
        {
            self.alpha_ffn
        } else if role.contains("embed") || role.contains("lm_head") {
            self.alpha_emb
        } else {
            0.5
        };

        let layer_ratio = match slot.block {
            Some(b) => {
                let span = total_blocks.max(1);
                let block_idx = (b * 6 / span).min(5);
                self.r[block_idx]
            }
            None => 0.5,
        };

        (self.gamma * 0.4 + component_ratio * 0.3 + layer_ratio * 0.3).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe_genome() -> DarwinGenome {
        DarwinGenome {
            gamma: 0.0,
            alpha_attn: 1.0,
            alpha_ffn: 0.0,
            alpha_emb: 0.0,
            rho_a: 0.5,
            rho_b: 0.5,
            r: [0.0; 6],
            tau: 0.5,
            lambda: 0.1,
        }
    }

    #[test]
    fn new_nparent_rows_are_simplex() {
        let g = NparentGenome::new_nparent(4, 6);
        assert_eq!(g.n_parents(), 4);
        assert_eq!(g.n_blocks(), 6);
        for b in 0..6 {
            let sum: f32 = g.row(b).iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "row {b} sums to {sum}");
            assert!(g.row(b).iter().all(|&w| (w - 0.25).abs() < 1e-6));
        }
    }

    #[test]
    fn random_crossover_mutate_preserve_simplex() {
        let a = NparentGenome::random(1, 3, 4);
        let b = NparentGenome::random(2, 3, 4);
        let mut rng = 42u64;
        let c = NparentGenome::crossover(&a, &b, &mut rng);
        let m = a.mutate(1.0, &mut rng);
        for g in [&c, &m] {
            for blk in 0..4 {
                let sum: f32 = g.row(blk).iter().sum();
                assert!((sum - 1.0).abs() < 1e-5, "row sums to {sum}");
                assert!(g.row(blk).iter().all(|&w| w >= 0.0));
            }
        }
    }

    #[test]
    fn nway_tau_endpoints_match_formula() {
        let mri = MriTrustFusion::new(Default::default(), None);
        // tau = 1 -> pure MRI simplex
        let r = mri.final_ratio_nway("b0.attn.q_proj", &[1.0, 3.0], &[0.5, 0.5], 1.0);
        assert!((r[0] - 0.25).abs() < 1e-6);
        assert!((r[1] - 0.75).abs() < 1e-6);
        // tau = 0 -> pure genome row
        let r = mri.final_ratio_nway("b0.attn.q_proj", &[1.0, 3.0], &[0.9, 0.1], 0.0);
        assert!((r[0] - 0.9).abs() < 1e-6);
        assert!((r[1] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn nway_degenerate_scores_fall_back_to_uniform_mri() {
        let mri = MriTrustFusion::new(Default::default(), None);
        let r = mri.final_ratio_nway("b0.ffn.gate_proj", &[0.0, 0.0], &[0.5, 0.5], 1.0);
        assert!((r[0] - 0.5).abs() < 1e-6);
        assert!((r[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn slot_ratio_uses_role_not_raw_name() {
        let g = probe_genome();
        let attn = CanonicalSlot::named(Some(3), "attn.q_proj");
        // gamma=0, alpha_attn=1, r[3*6/24=0]=0, untrusted block term 0 -> 0.3
        assert!((g.slot_ratio(&attn, 24) - 0.3).abs() < 1e-6);
        let ffn = CanonicalSlot::named(Some(3), "ffn.gate_proj");
        // alpha_ffn=0, block 3 of 24 -> r[3*6/24=0]=0 -> 0.0
        assert!((g.slot_ratio(&ffn, 24) - 0.0).abs() < 1e-6);
        let global = CanonicalSlot::named(None, "model.embed_tokens");
        // alpha_emb=0 -> 0.15
        assert!((g.slot_ratio(&global, 24) - 0.15).abs() < 1e-6);
    }

    #[test]
    fn blend_slot_is_convex_combination() {
        let g = NparentGenome::new_nparent(2, 1);
        let mri = MriTrustFusion::new(Default::default(), None);
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];
        let out = g.blend_slot(0, &[&a, &b], &[1.0, 1.0], &mri, "b0.attn.q_proj");
        // uniform MRI + uniform genome, tau=0.5 -> [0.5, 0.5] -> midpoint
        assert!((out[0] - 2.0).abs() < 1e-6);
        assert!((out[1] - 3.0).abs() < 1e-6);
    }
}
