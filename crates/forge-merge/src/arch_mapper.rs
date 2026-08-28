use std::collections::HashMap;
use forge_core::{ArchitectureFamily, TensorMeta};

/// Maps tensors between two models with potentially different architectures.
/// Used for cross-architecture merging (Transformer ↔ Mamba, different sizes, etc.)
pub struct ArchitectureMapper {
    pub compatibility_threshold: f32,
}

#[derive(Debug, Clone)]
pub struct MergePlan {
    pub tensor_pairs: Vec<TensorPair>,
    pub skipped: Vec<String>,
    pub projected: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct TensorPair {
    pub name: String,
    pub source_a: String,
    pub source_b: String,
    pub needs_projection: bool,
    pub compatibility_score: f32,
}

impl ArchitectureMapper {
    pub fn new(threshold: f32) -> Self {
        Self { compatibility_threshold: threshold }
    }

    /// Compute compatibility between two tensors
    pub fn tensor_compatibility(a: &TensorMeta, b: &TensorMeta) -> f32 {
        // Type match (0.5 weight)
        let type_match = if a.dtype == b.dtype { 1.0 } else { 0.5 };

        // Dimension match (0.3 weight)
        let dim_match = if a.shape == b.shape {
            1.0
        } else if a.shape.len() == b.shape.len() {
            // Partial match: compare last dimensions
            let a_last = a.shape.last().unwrap_or(&0);
            let b_last = b.shape.last().unwrap_or(&0);
            let min_dim = (*a_last).min(*b_last) as f32;
            let max_dim = (*a_last).max(*b_last) as f32;
            min_dim / max_dim
        } else {
            0.0
        };

        // Parameter count match (0.2 weight)
        let params_a: usize = a.shape.iter().product();
        let params_b: usize = b.shape.iter().product();
        let param_match = (params_a.min(params_b) as f32) / (params_a.max(params_b) as f32);

        0.5 * type_match + 0.3 * dim_match + 0.2 * param_match
    }

    /// Plan a cross-architecture merge
    pub fn plan_merge(
        &self,
        tensors_a: &[TensorMeta],
        tensors_b: &[TensorMeta],
    ) -> MergePlan {
        let mut tensor_pairs = Vec::new();
        let mut skipped = Vec::new();
        let mut projected = Vec::new();
        let mut used_b: std::collections::HashSet<String> = std::collections::HashSet::new();

        for ta in tensors_a {
            let mut best_match: Option<(&TensorMeta, f32)> = None;

            for tb in tensors_b {
                if used_b.contains(&tb.name) {
                    continue;
                }

                // Only match tensors with same suffix/name pattern
                let name_match = Self::name_similarity(&ta.name, &tb.name);
                if name_match < 0.3 {
                    continue;
                }

                let compat = Self::tensor_compatibility(ta, tb) * name_match;
                if compat > self.compatibility_threshold {
                    if best_match.map_or(true, |(_, best)| compat > best) {
                        best_match = Some((tb, compat));
                    }
                }
            }

            if let Some((tb, score)) = best_match {
                let needs_projection = ta.shape != tb.shape;
                if needs_projection {
                    projected.push(ta.name.clone());
                }
                used_b.insert(tb.name.clone());
                tensor_pairs.push(TensorPair {
                    name: ta.name.clone(),
                    source_a: ta.name.clone(),
                    source_b: tb.name.clone(),
                    needs_projection,
                    compatibility_score: score,
                });
            } else {
                skipped.push(ta.name.clone());
            }
        }

        MergePlan { tensor_pairs, skipped, projected }
    }

    /// Compute name similarity (simple suffix matching)
    fn name_similarity(a: &str, b: &str) -> f32 {
        let a_parts: Vec<&str> = a.split('.').collect();
        let b_parts: Vec<&str> = b.split('.').collect();

        // Compare last N parts
        let max_len = a_parts.len().min(b_parts.len());
        let compare_len = max_len.min(3);

        let a_suffix: Vec<&str> = a_parts.iter().rev().take(compare_len).copied().collect();
        let b_suffix: Vec<&str> = b_parts.iter().rev().take(compare_len).copied().collect();

        let matches = a_suffix.iter().zip(b_suffix.iter())
            .filter(|(a, b)| a == b)
            .count();

        matches as f32 / compare_len as f32
    }
}
