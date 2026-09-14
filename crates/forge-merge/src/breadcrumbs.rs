use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;

/// Model Breadcrumbs: task arithmetic with outlier removal.
/// Drops the smallest-β and largest-γ fraction of |task vector| magnitudes
/// (mask per finetune), then averages survivors. The `ties` variant adds
/// sign-consensus: a parameter survives only if its sign agrees with the
/// majority sign across models.
pub struct BreadcrumbsMerge {
    pub base_index: usize,
    pub lambda: f32,
    /// Fraction of smallest magnitudes to mask out
    pub beta: f32,
    /// Fraction of largest magnitudes to mask out
    pub gamma: f32,
    /// If true, enforce majority-sign consensus (breadcrumbs_ties)
    pub sign_consensus: bool,
}

impl BreadcrumbsMerge {
    pub fn new(lambda: f32, beta: f32, gamma: f32) -> Self {
        Self { base_index: 0, lambda, beta, gamma, sign_consensus: false }
    }

    pub fn ties(lambda: f32, beta: f32, gamma: f32) -> Self {
        Self { base_index: 0, lambda, beta, gamma, sign_consensus: true }
    }
}

impl MergeOp for BreadcrumbsMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.len() < 2 {
            anyhow::bail!("breadcrumbs: need base + ≥1 finetune");
        }
        let base_idx = self.base_index.min(inputs.len() - 1);
        let base = &inputs[base_idx];
        if base.len() != meta.num_elements() {
            anyhow::bail!("breadcrumbs: base tensor size mismatch");
        }
        let mut taus: Vec<Vec<f32>> = Vec::new();
        for (i, ft) in inputs.iter().enumerate() {
            if i == base_idx || ft.len() != base.len() {
                continue;
            }
            taus.push(base.iter().zip(ft.iter()).map(|(b, f)| f - b).collect());
        }
        if taus.is_empty() {
            return Ok(base.clone());
        }
        // Per-finetune magnitude masks
        let mut masked: Vec<Vec<f32>> = Vec::with_capacity(taus.len());
        for tau in &taus {
            let mut mags: Vec<f32> = tau.iter().map(|x| x.abs()).collect();
            mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo = mags[((self.beta.clamp(0.0, 1.0) * mags.len() as f32) as usize).min(mags.len() - 1)];
            let hi_idx = ((1.0 - self.gamma.clamp(0.0, 1.0)) * mags.len() as f32) as usize;
            let hi = mags[hi_idx.min(mags.len() - 1)];
            masked.push(
                tau.iter()
                    .map(|x| {
                        let m = x.abs();
                        if m < lo || m > hi {
                            0.0
                        } else {
                            *x
                        }
                    })
                    .collect(),
            );
        }
        let mut result = base.clone();
        for j in 0..base.len() {
            if self.sign_consensus {
                let pos: usize = masked.iter().filter(|t| t[j] > 0.0).count();
                let neg: usize = masked.iter().filter(|t| t[j] < 0.0).count();
                if pos == 0 && neg == 0 {
                    continue;
                }
                let sign = if pos >= neg { 1.0 } else { -1.0 };
                let mut acc = 0.0f32;
                let mut cnt = 0usize;
                for t in &masked {
                    if t[j] * sign > 0.0 {
                        acc += t[j];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    result[j] += self.lambda * acc / cnt as f32;
                }
            } else {
                let mut acc = 0.0f32;
                let mut cnt = 0usize;
                for t in &masked {
                    if t[j] != 0.0 {
                        acc += t[j];
                        cnt += 1;
                    }
                }
                if cnt > 0 {
                    result[j] += self.lambda * acc / cnt as f32;
                }
            }
        }
        Ok(result)
    }
}
