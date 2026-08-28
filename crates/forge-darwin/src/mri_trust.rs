use std::collections::HashMap;

/// MRI-Trust Fusion: adaptively balances diagnostic layer-importance
/// with evolutionary search through a learnable trust parameter.
///
/// r_final(T) = τ · r_MRI(T) + (1 - τ) · r_genome(T)
///
/// where r_MRI(T) = MRI_B(T) / (MRI_A(T) + MRI_B(T))
///       MRI(T)   = α · Static(T) + (1 - α) · Probe(T)
pub struct MriTrustFusion {
    /// Static scores (entropy + variance + L2 norm, no calibration data)
    pub static_scores: HashMap<String, f32>,
    /// Probe scores (cosine distance between reasoning-conditioned and generic activations)
    pub probe_scores: Option<HashMap<String, f32>>,
    /// Weighting between static and probe (0=static only, 1=probe only)
    pub alpha: f32,
}

impl MriTrustFusion {
    pub fn new(static_scores: HashMap<String, f32>, probe_scores: Option<HashMap<String, f32>>) -> Self {
        Self {
            static_scores,
            probe_scores,
            alpha: 0.5,
        }
    }

    /// Compute MRI score for a tensor
    pub fn mri_score(&self, tensor_name: &str) -> f32 {
        let static_score = self.static_scores.get(tensor_name).copied().unwrap_or(0.5);
        let probe_score = self.probe_scores
            .as_ref()
            .and_then(|p| p.get(tensor_name))
            .copied()
            .unwrap_or(static_score);

        self.alpha * static_score + (1.0 - self.alpha) * probe_score
    }

    /// Compute MRI ratio between two models for a tensor
    pub fn mri_ratio(&self, tensor_name: &str, mri_a: f32, mri_b: f32) -> f32 {
        if mri_a + mri_b < 1e-8 {
            0.5
        } else {
            mri_b / (mri_a + mri_b)
        }
    }

    /// Compute final merge ratio using MRI-Trust Fusion
    pub fn final_ratio(
        &self,
        tensor_name: &str,
        mri_a: f32,
        mri_b: f32,
        genome_ratio: f32,
        tau: f32,
    ) -> f32 {
        let r_mri = self.mri_ratio(tensor_name, mri_a, mri_b);
        tau * r_mri + (1.0 - tau) * genome_ratio
    }

    /// Compute static MRI scores from tensor statistics
    pub fn compute_static_scores(tensors: &[(&str, &[f32])]) -> HashMap<String, f32> {
        let mut scores = HashMap::new();

        for (name, data) in tensors {
            let n = data.len() as f32;

            // Normalized entropy
            let mean = data.iter().sum::<f32>() / n;
            let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
            let std = variance.sqrt();

            // Quantize to bins for entropy
            let num_bins = 64.min(data.len());
            let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;

            if range < 1e-8 {
                scores.insert(name.to_string(), 0.5);
                continue;
            }

            let mut bins = vec![0u32; num_bins];
            for &val in data.iter() {
                let bin = ((val - min_val) / range * (num_bins as f32 - 1.0)) as usize;
                bins[bin.min(num_bins - 1)] += 1;
            }

            let entropy: f32 = bins.iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f32 / n;
                    -p * p.log2()
                })
                .sum();

            let max_entropy = (num_bins as f32).log2();
            let norm_entropy = if max_entropy > 0.0 { entropy / max_entropy } else { 0.5 };

            // Capped L2 norm
            let l2: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
            let l2_capped = (l2 / n.sqrt()).min(1.0);

            // Combine: higher score = more important tensor
            let score = 0.4 * norm_entropy + 0.3 * (std / (std + 1.0)) + 0.3 * l2_capped;
            scores.insert(name.to_string(), score);
        }

        scores
    }
}
