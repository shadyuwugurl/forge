//! Structural model surgery: cross-attention encoder fusion and friends.
//!
//! This crate lives apart from `forge-merge` on purpose: merging blends
//! tensors that already share a role, while surgery *changes structure* —
//! here, grafting encoder output streams into a decoder via fresh
//! cross-attention projections. Everything is plan-first: [`EncoderFuser::attach`]
//! only reads [`TensorMap`] metadata, and per-pair projection tensors are
//! generated one pair at a time so two >14B models are never resident.

pub mod cross_attention;

pub use cross_attention::{CrossAttnParams, EncoderFuseConfig, EncoderFusePlan, EncoderFuser, FusedPair};
