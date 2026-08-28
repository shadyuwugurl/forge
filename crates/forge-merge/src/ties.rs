use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// TIES-Merging: Trim, Elect Sign, Disjoint Merge
pub struct TiesMerge<'a> {
    pub base: &'a [f32],
    pub models: Vec<(&'a [f32], f32, f32)>,  // (tensor, weight, density)
}

impl<'a> TiesMerge<'a> {
    pub fn new(base: &'a [f32], models: Vec<(&'a [f32], f32, f32)>) -> Self {
        Self { base, models }
    }
}

impl MergeOp for TiesMerge<'_> {
    fn merge_tensor(&self, _name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        let len = self.base.len();
        let mut result = self.base.to_vec();

        // Compute task vectors (delta from base)
        let mut task_vectors: Vec<Vec<f32>> = self.models.iter()
            .map(|(model, _, _)| {
                model.iter().zip(self.base.iter())
                    .map(|(m, b)| m - b)
                    .collect()
            })
            .collect();

        // Step 1: Trim — zero out smallest magnitude values based on density
        for (tv, (_, _, density)) in task_vectors.iter_mut().zip(self.models.iter()) {
            let threshold = compute_threshold(tv, *density);
            for val in tv.iter_mut() {
                if val.abs() < threshold {
                    *val = 0.0;
                }
            }
        }

        // Step 2: Elect sign — for each position, keep only values matching the majority sign
        for i in 0..len {
            let positive_count: i32 = task_vectors.iter()
                .map(|tv| if tv[i] > 0.0 { 1 } else { -1 })
                .sum();
            let majority_sign = if positive_count >= 0 { 1.0 } else { -1.0 };

            for tv in task_vectors.iter_mut() {
                if tv[i].signum() != majority_sign && tv[i] != 0.0 {
                    tv[i] = 0.0;
                }
            }
        }

        // Step 3: Disjoint merge — sum remaining values, add to base
        let total_weight: f32 = self.models.iter().map(|(_, w, _)| w).sum();
        for i in 0..len {
            let delta: f32 = task_vectors.iter().zip(self.models.iter())
                .map(|(tv, (_, w, _))| tv[i] * w)
                .sum();
            result[i] += delta / total_weight;
        }

        Ok(result)
    }
}

fn compute_threshold(values: &[f32], density: f32) -> f32 {
    let mut abs_values: Vec<f32> = values.iter().map(|x| x.abs()).collect();
    abs_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let keep_count = (values.len() as f32 * density).ceil() as usize;
    let idx = values.len().saturating_sub(keep_count);
    abs_values.get(idx).copied().unwrap_or(0.0)
}
