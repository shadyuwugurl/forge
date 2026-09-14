use anyhow::Result;

/// B2 LoPT: Layer-partitioned Optimization with a single gradient boundary.
///
/// Post-training only. Positions before `boundary * n` are frozen (pass
/// through bit-exact); positions at/after the boundary take one SGD step
/// `w -= lr * g`. Default boundary is the tensor midpoint (0.5).
///
/// Conflict note: LoPT must NOT be composed with [`crate::lls::LlsTrainer`]
/// in a single run — LLS is pretraining-only (log-domain normalization of
/// the forward pass) while LoPT is a post-training gradient gate. Running
/// both silently double-transforms the tunable half.
pub struct LoptTrainer {
    /// Gradient boundary as a fraction of tensor length, in (0, 1).
    pub boundary: f32,
    /// Learning rate applied to the tunable partition.
    pub lr: f32,
}

impl LoptTrainer {
    pub fn new(boundary: f32, lr: f32) -> Self {
        Self {
            boundary: boundary.clamp(0.01, 0.99),
            lr: lr.clamp(0.0, 1.0),
        }
    }

    pub fn midpoint(lr: f32) -> Self {
        Self::new(0.5, lr)
    }

    /// Index of the first tunable element for a length-`n` tensor.
    pub fn split_point(&self, n: usize) -> usize {
        ((n as f32) * self.boundary) as usize
    }

    /// True when position `pos` of a length-`n` tensor takes gradients.
    pub fn is_tunable(&self, pos: usize, n: usize) -> bool {
        pos >= self.split_point(n)
    }

    /// Split one shard into (frozen_prefix, tunable_suffix) views.
    pub fn apply_boundary(&self, input: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let k = self.split_point(input.len()).min(input.len());
        (input[..k].to_vec(), input[k..].to_vec())
    }

    /// One SGD step with the gradient gate: frozen prefix passes through
    /// unchanged, tunable suffix updates as `w -= lr * g`.
    pub fn sgd_step(&self, weights: &[f32], grads: &[f32]) -> Result<Vec<f32>> {
        if weights.len() != grads.len() {
            anyhow::bail!(
                "LoptTrainer: weights len {} != grads len {}",
                weights.len(),
                grads.len()
            );
        }
        let n = weights.len();
        let k = self.split_point(n).min(n);
        let mut out = Vec::with_capacity(n);
        out.extend_from_slice(&weights[..k]);
        for (w, g) in weights[k..].iter().zip(grads[k..].iter()) {
            out.push(w - self.lr * g);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_prefix_passthrough_and_suffix_updates() {
        let t = LoptTrainer::midpoint(0.1);
        let w = vec![1.0f32, 2.0, 3.0, 4.0];
        let g = vec![1.0f32; 4];
        let out = t.sgd_step(&w, &g).unwrap();
        assert_eq!(out[0], 1.0);
        assert_eq!(out[1], 2.0);
        assert!((out[2] - 2.9).abs() < 1e-6, "got {}", out[2]);
        assert!((out[3] - 3.9).abs() < 1e-6, "got {}", out[3]);
    }

    #[test]
    fn custom_boundary_respected() {
        let t = LoptTrainer::new(0.25, 1.0);
        assert_eq!(t.split_point(8), 2);
        let w = vec![5.0f32; 8];
        let g = vec![1.0f32; 8];
        let out = t.sgd_step(&w, &g).unwrap();
        assert!(out[..2].iter().all(|&v| v == 5.0));
        assert!(out[2..].iter().all(|&v| v == 4.0));
    }

    #[test]
    fn length_mismatch_bails() {
        let t = LoptTrainer::midpoint(0.1);
        assert!(t.sgd_step(&[1.0, 2.0], &[1.0]).is_err());
    }

    #[test]
    fn boundary_split_views_partition() {
        let t = LoptTrainer::midpoint(0.1);
        let (frozen, tunable) = t.apply_boundary(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(frozen, vec![1.0, 2.0, 3.0]);
        assert_eq!(tunable, vec![4.0, 5.0, 6.0]);
        assert!(t.is_tunable(3, 6));
        assert!(!t.is_tunable(2, 6));
    }

    #[test]
    fn zero_lr_is_noop_on_suffix() {
        let t = LoptTrainer::midpoint(0.0);
        let w = vec![7.0f32; 4];
        let g = vec![3.0f32; 4];
        assert_eq!(t.sgd_step(&w, &g).unwrap(), w);
    }
}
