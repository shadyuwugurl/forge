//! AX-Ray prefix-invariance audit (arXiv 2608.22876): 2-forward-pass per-layer
//! score, no training or gradients.
//!
//! For each layer, compare the activations produced with and without a fixed
//! probe prefix. A prefix-invariant layer scores ~1.0 (cosine similarity);
//! scan/aggregation/norm leaks that mask inspection misses show up as sharp
//! per-layer drops, localizable to the exact faulty layer.
//!
//! The CLI feeds this module from activation dumps (`{"layers": [[...]]}`);
//! all math here is pure and fixture-testable.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Per-layer audit outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAudit {
    pub layer: usize,
    pub score: f32,
}

/// Cosine-similarity prefix-invariance score in [-1, 1].
/// Clean layers score ~1.0. Bails on empty or mismatched inputs.
pub fn prefix_invariance_score(clean: &[f32], prefixed: &[f32]) -> Result<f32> {
    if clean.is_empty() || prefixed.is_empty() {
        bail!("prefix audit needs non-empty activation vectors");
    }
    if clean.len() != prefixed.len() {
        bail!(
            "prefix audit length mismatch: clean={} prefixed={}",
            clean.len(),
            prefixed.len()
        );
    }
    let mut dot = 0.0f32;
    let mut nc = 0.0f32;
    let mut np = 0.0f32;
    for (a, b) in clean.iter().zip(prefixed.iter()) {
        dot += a * b;
        nc += a * a;
        np += b * b;
    }
    if nc <= 1e-12 || np <= 1e-12 {
        bail!("prefix audit hit a near-zero activation vector");
    }
    Ok((dot / (nc.sqrt() * np.sqrt())).clamp(-1.0, 1.0))
}

/// Score every layer; `clean[i]` vs `prefixed[i]` must pair up.
pub fn audit_layers(clean: &[Vec<f32>], prefixed: &[Vec<f32>]) -> Result<Vec<LayerAudit>> {
    if clean.len() != prefixed.len() {
        bail!(
            "prefix audit layer-count mismatch: clean={} prefixed={}",
            clean.len(),
            prefixed.len()
        );
    }
    clean
        .iter()
        .zip(prefixed.iter())
        .enumerate()
        .map(|(i, (c, p))| Ok(LayerAudit { layer: i, score: prefix_invariance_score(c, p)? }))
        .collect()
}

/// Layers scoring strictly below `threshold`, ascending.
pub fn localize(audits: &[LayerAudit], threshold: f32) -> Vec<usize> {
    let mut out: Vec<usize> = audits
        .iter()
        .filter(|a| a.score < threshold)
        .map(|a| a.layer)
        .collect();
    out.sort_unstable();
    out
}

/// Serializable audit report (the armor/audit V1 contract: gate + JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub threshold: f32,
    pub layers: Vec<LayerAudit>,
    pub faulty: Vec<usize>,
}

impl AuditReport {
    pub fn new(layers: Vec<LayerAudit>, threshold: f32) -> Self {
        let faulty = localize(&layers, threshold);
        Self { threshold, layers, faulty }
    }

    pub fn write_json(&self, path: &std::path::Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(path, text)?;
        Ok(())
    }
}

/// Activation-dump file format the CLI accepts: `{"layers": [[f32...], ...]}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationDump {
    pub layers: Vec<Vec<f32>>,
}

impl ActivationDump {
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer_vec(val: f32, n: usize) -> Vec<f32> {
        // Deterministic non-trivial signal: val * (1 + i/n).
        (0..n).map(|i| val * (1.0 + i as f32 / n as f32)).collect()
    }

    #[test]
    fn clean_model_scores_near_one() {
        let clean: Vec<Vec<f32>> = (0..8).map(|l| layer_vec(0.5 + l as f32 * 0.1, 64)).collect();
        // Prefix pass adds tiny numerical jitter only.
        let prefixed: Vec<Vec<f32>> = clean
            .iter()
            .map(|v| v.iter().map(|x| x * 1.0001).collect())
            .collect();
        let audits = audit_layers(&clean, &prefixed).unwrap();
        assert_eq!(audits.len(), 8);
        assert!(audits.iter().all(|a| a.score > 0.999));
        assert!(localize(&audits, 0.99).is_empty());
    }

    #[test]
    fn injected_inter_chunk_fault_localizes_exactly() {
        let clean: Vec<Vec<f32>> = (0..8).map(|l| layer_vec(0.5 + l as f32 * 0.1, 64)).collect();
        let mut prefixed = clean.clone();
        // Layer 5 leaks: replace with an unrelated direction.
        prefixed[5] = layer_vec(-3.0, 64);
        let audits = audit_layers(&clean, &prefixed).unwrap();
        let faulty = localize(&audits, 0.99);
        assert_eq!(faulty, vec![5]);
        // All other layers stay clean.
        assert!(audits.iter().filter(|a| a.layer != 5).all(|a| a.score > 0.999));
    }

    #[test]
    fn threshold_boundary_is_strict() {
        let audits = vec![
            LayerAudit { layer: 0, score: 0.99 },
            LayerAudit { layer: 1, score: 0.989 },
        ];
        assert_eq!(localize(&audits, 0.99), vec![1]);
        assert!(localize(&audits, 0.0).is_empty());
    }

    #[test]
    fn bad_inputs_bail() {
        assert!(prefix_invariance_score(&[], &[]).is_err());
        assert!(prefix_invariance_score(&[1.0], &[1.0, 2.0]).is_err());
        assert!(prefix_invariance_score(&[0.0, 0.0], &[0.0, 0.0]).is_err());
        assert!(audit_layers(&[vec![1.0]], &[vec![1.0], vec![2.0]]).is_err());
    }
}
