//! Cross-attention encoder fusion: graft encoder output streams into a decoder.
//!
//! Standard adapter-style approach: for selected (decoder block, encoder)
//! pairs we create *fresh* Q/K/V/O projection tensors initialized at
//! `gate_init` scale (near zero), so the decoder is initially unchanged and a
//! downstream trainer ([`DiffusionBlocksTrainer`](forge_train)) teaches the
//! cross-attention. Merge-time work is (a) [`EncoderFuser::attach`], a pure
//! [`TensorMap`] metadata pass that emits an [`EncoderFusePlan`], and
//! (b) per-pair [`EncoderFuser::init_pair_params`], so only one pair's
//! `4 * dim * dim` floats are ever resident — the two >14B source models are
//! never loaded together.

use anyhow::{bail, Result};
use forge_core::tensor_map::TensorMap;
use serde::{Deserialize, Serialize};

/// Configuration for [`EncoderFuser::attach`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderFuseConfig {
    /// Attention heads per fused cross-attention block.
    pub heads: usize,
    /// Model width. `0` = infer from `TensorMeta` shapes at apply time.
    pub dim: usize,
    /// Init scale for fresh projections (near zero preserves the decoder).
    pub gate_init: f32,
    /// Maximum number of (decoder block, encoder) pairs to fuse.
    pub max_pairs: usize,
    /// Fuse every `stride`-th decoder block (1 = densest).
    pub stride: usize,
    /// Encoders stay frozen; only fresh projections train (informational).
    pub freeze_encoders: bool,
}

impl Default for EncoderFuseConfig {
    fn default() -> Self {
        Self {
            heads: 8,
            dim: 0,
            gate_init: 1e-3,
            max_pairs: 8,
            stride: 1,
            freeze_encoders: true,
        }
    }
}

/// One fused (decoder block, encoder) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedPair {
    pub decoder_block: usize,
    pub encoder_idx: usize,
    pub encoder_block: usize,
    /// Gating value: 0 = pure decoder, 1 = pure cross-attention path.
    pub gate: f32,
}

/// Metadata-only fusion plan: no weights, safe to hold for huge models.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncoderFusePlan {
    pub pairs: Vec<FusedPair>,
    pub heads: usize,
    /// 0 = infer from shapes at apply time.
    pub dim: usize,
}

/// Fresh cross-attention projections for one [`FusedPair`], row-major
/// `[dim, dim]` each.
#[derive(Debug, Clone)]
pub struct CrossAttnParams {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub dim: usize,
}

pub struct EncoderFuser;

impl EncoderFuser {
    /// Build a fusion plan from aligned maps. Pure metadata: reads slot keys
    /// only (`b{idx}.{role}` / `global.{role}`), never weights.
    ///
    /// Decoder blocks are fused every `stride`-th block up to `max_pairs`,
    /// round-robin over encoders; each pair targets the encoder block with
    /// the closest ordinal (clamped), so depth ordering is preserved across
    /// heterogeneous depths.
    pub fn attach(
        decoder: &TensorMap,
        encoders: &[TensorMap],
        config: &EncoderFuseConfig,
    ) -> Result<EncoderFusePlan> {
        if encoders.is_empty() {
            bail!("encoder fusion needs at least one encoder map");
        }
        if config.heads == 0 {
            bail!("heads must be >= 1");
        }
        if config.stride == 0 {
            bail!("stride must be >= 1");
        }
        let mut dec_blocks = Self::block_ordinals(decoder);
        if dec_blocks.is_empty() {
            bail!("decoder map has no block slots to fuse into");
        }
        dec_blocks.sort_unstable();
        dec_blocks.dedup();

        let enc_blocks: Vec<Vec<usize>> =
            encoders.iter().map(Self::block_ordinals).collect();
        for (i, blocks) in enc_blocks.iter().enumerate() {
            if blocks.is_empty() {
                bail!("encoder {i} map has no block slots to fuse from");
            }
        }

        let mut pairs = Vec::new();
        let mut enc_cursor = 0usize;
        for (k, &db) in dec_blocks.iter().step_by(config.stride).enumerate() {
            if k >= config.max_pairs {
                break;
            }
            let ei = enc_cursor % encoders.len();
            enc_cursor += 1;
            let eb = Self::nearest_block(&enc_blocks[ei], db);
            pairs.push(FusedPair {
                decoder_block: db,
                encoder_idx: ei,
                encoder_block: eb,
                gate: config.gate_init,
            });
        }
        Ok(EncoderFusePlan {
            pairs,
            heads: config.heads,
            dim: config.dim,
        })
    }

    /// Generate fresh projections for one pair. Deterministic LCG so runs are
    /// reproducible without an RNG dependency; uniform in
    /// `[-gate_init, +gate_init]` (Xavier-style bound would need fan-in —
    /// caller passes resolved `dim`, and the near-zero gate dominates anyway).
    pub fn init_pair_params(dim: usize, seed: u64, gate_init: f32) -> Result<CrossAttnParams> {
        if dim == 0 {
            bail!("dim must be resolved from TensorMeta shapes before init");
        }
        let n = dim * dim;
        let mut state = seed;
        let mut next = || -> f32 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1);
            let t = (state >> 33) as f32 / (1u32 << 31) as f32;
            (t * 2.0 - 1.0) * gate_init
        };
        let mut fill = || -> Vec<f32> { (0..n).map(|_| next()).collect() };
        Ok(CrossAttnParams {
            q_proj: fill(),
            k_proj: fill(),
            v_proj: fill(),
            o_proj: fill(),
            dim,
        })
    }

    /// Gated blend kernel: `out = (1 - gate) * decoder + gate * cross`.
    /// Both slices must share length; the cross path is the trained
    /// cross-attention output at inference time.
    pub fn apply_gate(decoder: &[f32], cross: &[f32], gate: f32) -> Result<Vec<f32>> {
        if decoder.len() != cross.len() {
            bail!(
                "gate blend length mismatch: {} vs {}",
                decoder.len(),
                cross.len()
            );
        }
        Ok(decoder
            .iter()
            .zip(cross.iter())
            .map(|(d, c)| (1.0 - gate) * d + gate * c)
            .collect())
    }

    /// Distinct block ordinals present in a map, parsed from `b{idx}.` keys.
    fn block_ordinals(map: &TensorMap) -> Vec<usize> {
        let mut out = Vec::new();
        for key in map.slots.keys() {
            if let Some(rest) = key.strip_prefix('b') {
                if let Some((num, _)) = rest.split_once('.') {
                    if let Ok(idx) = num.parse::<usize>() {
                        out.push(idx);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Closest encoder block ordinal to a decoder block (ties go deeper).
    fn nearest_block(blocks: &[usize], target: usize) -> usize {
        let mut best = blocks[0];
        let mut best_dist = usize::MAX;
        for &b in blocks {
            let d = b.abs_diff(target);
            if d < best_dist || (d == best_dist && b > best) {
                best = b;
                best_dist = d;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forge_core::tensor_map::{SlotEntry, TensorMap};
    use std::collections::HashMap;

    fn map_with_blocks(blocks: &[usize]) -> TensorMap {
        let mut slots = HashMap::new();
        for &b in blocks {
            let key = format!("b{b}.attn");
            slots.insert(
                key.clone(),
                SlotEntry {
                    slot: None,
                    raw_names: vec![format!("layers.{b}.attn.weight")],
                    fused: false,
                },
            );
        }
        TensorMap {
            slots,
            by_raw: HashMap::new(),
            common: Vec::new(),
        }
    }

    #[test]
    fn attach_round_robins_over_encoders() {
        let dec = map_with_blocks(&[0, 1, 2, 3]);
        let encs = vec![map_with_blocks(&[0, 1]), map_with_blocks(&[0, 1, 2])];
        let cfg = EncoderFuseConfig {
            max_pairs: 4,
            ..Default::default()
        };
        let plan = EncoderFuser::attach(&dec, &encs, &cfg).unwrap();
        assert_eq!(plan.pairs.len(), 4);
        assert_eq!(
            plan.pairs.iter().map(|p| p.encoder_idx).collect::<Vec<_>>(),
            vec![0, 1, 0, 1]
        );
        assert_eq!(
            plan.pairs.iter().map(|p| p.decoder_block).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn attach_preserves_depth_ordering_across_depths() {
        let dec = map_with_blocks(&[0, 10, 20, 30]);
        let encs = vec![map_with_blocks(&[0, 1, 2])];
        let plan = EncoderFuser::attach(&dec, &encs, &EncoderFuseConfig::default()).unwrap();
        let ebs: Vec<usize> = plan.pairs.iter().map(|p| p.encoder_block).collect();
        assert_eq!(ebs, vec![0, 2, 2, 2]);
    }

    #[test]
    fn attach_rejects_empty_encoders_and_blockless_decoder() {
        let dec = map_with_blocks(&[0]);
        assert!(EncoderFuser::attach(&dec, &[], &EncoderFuseConfig::default()).is_err());
        let empty = TensorMap::default();
        let encs = vec![map_with_blocks(&[0])];
        assert!(EncoderFuser::attach(&empty, &encs, &EncoderFuseConfig::default()).is_err());
        assert!(EncoderFuser::attach(
            &dec,
            &encs,
            &EncoderFuseConfig { heads: 0, ..Default::default() }
        )
        .is_err());
    }

    #[test]
    fn init_params_bounded_by_gate_and_sized() {
        let p = EncoderFuser::init_pair_params(16, 7, 1e-3).unwrap();
        assert_eq!(p.q_proj.len(), 256);
        assert_eq!(p.o_proj.len(), 256);
        for w in p.q_proj.iter().chain(&p.k_proj).chain(&p.v_proj).chain(&p.o_proj) {
            assert!(w.abs() <= 1e-3 + 1e-9, "weight {w} escapes gate bound");
        }
        assert!(EncoderFuser::init_pair_params(0, 7, 1e-3).is_err());
    }

    #[test]
    fn gate_endpoints_preserve_each_path() {
        let d = vec![1.0, 2.0, 3.0];
        let c = vec![10.0, 20.0, 30.0];
        assert_eq!(EncoderFuser::apply_gate(&d, &c, 0.0).unwrap(), d);
        assert_eq!(EncoderFuser::apply_gate(&d, &c, 1.0).unwrap(), c);
        assert_eq!(EncoderFuser::apply_gate(&d, &c, 0.5).unwrap(), vec![5.5, 11.0, 16.5]);
        assert!(EncoderFuser::apply_gate(&d, &c[..2], 0.5).is_err());
    }

    #[test]
    fn stride_and_max_pairs_bound_plan() {
        let dec = map_with_blocks(&[0, 1, 2, 3, 4, 5]);
        let encs = vec![map_with_blocks(&[0, 1])];
        let plan = EncoderFuser::attach(
            &dec,
            &encs,
            &EncoderFuseConfig { stride: 2, max_pairs: 2, ..Default::default() },
        )
        .unwrap();
        assert_eq!(
            plan.pairs.iter().map(|p| p.decoder_block).collect::<Vec<_>>(),
            vec![0, 2]
        );
    }
}
