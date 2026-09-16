use std::collections::HashMap;
use std::path::Path;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// One row of `stats.json`: produced by Kaggle imatrix calibration
/// or by `forge imatrix`. Schema: [{layer, tensor, outlier_ratio}]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OrcaStat {
    pub layer: String,
    pub tensor: String,
    pub outlier_ratio: f32,
}

/// ORCA (Outlier Matters, AAAI-26): outlier-aware reserved-compute allocator.
/// Loads per-tensor outlier ratios and converts them into merge weights so
/// outlier-heavy tensors are preserved rather than averaged away.
#[derive(Debug, Clone)]
pub struct OrcaAllocator {
    pub threshold: f32,
    pub ratios: HashMap<String, f32>,
}

impl OrcaAllocator {
    /// Load stats.json if present; missing file -> empty map (uniform weights).
    pub fn compute(stats_path: Option<&Path>, threshold: f32) -> Result<Self> {
        let mut ratios = HashMap::new();
        if let Some(p) = stats_path {
            if p.exists() {
                let text = std::fs::read_to_string(p)?;
                let stats: Vec<OrcaStat> = serde_json::from_str(&text).unwrap_or_default();
                for s in stats {
                    ratios.insert(s.tensor.clone(), s.outlier_ratio);
                    ratios.insert(s.layer.clone(), s.outlier_ratio);
                }
            }
        }
        Ok(Self { threshold, ratios })
    }

    pub fn ratio_for(&self, name: &str) -> f32 {
        self.ratios.get(name).copied().unwrap_or(0.0)
    }

    /// Preservation weight: 1 + threshold * ratio, normalized by caller.
    pub fn weight_for(&self, name: &str) -> f32 {
        1.0 + self.threshold * self.ratio_for(name).clamp(0.0, 1.0)
    }

    /// Weighted blend of parent tensors with outlier-aware weights.
    /// `parent_ratios[i]` is the outlier ratio of this tensor in parent i.
    pub fn blend(&self, _name: &str, inputs: &[Vec<f32>], parent_ratios: &[f32]) -> Vec<f32> {
        if inputs.is_empty() {
            return Vec::new();
        }
        let n = inputs[0].len();
        let mut weights: Vec<f32> = parent_ratios
            .iter()
            .map(|r| 1.0 + self.threshold * r.clamp(0.0, 1.0))
            .collect();
        if weights.len() != inputs.len() {
            weights = vec![1.0; inputs.len()];
        }
        let sum: f32 = weights.iter().sum::<f32>().max(1e-6);
        let mut out = vec![0.0f32; n];
        for (inp, w) in inputs.iter().zip(weights.iter()) {
            let k = w / sum;
            for (o, v) in out.iter_mut().zip(inp.iter()) {
                *o += k * v;
            }
        }
        out
    }
}
