use anyhow::Result;
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use crate::slerp_utils::normalize_weights;

/// Karcher Mean: Riemannian (Fréchet) barycenter of model parameters.
/// Iteratively projects onto the tangent space at the current estimate,
/// averages, and retracts: μ ← μ + mean(log_μ(mᵢ)) via the spherical
/// log/exp maps, until the update norm falls below `tol` or `max_iter`.
pub struct KarcherMerge {
    pub weights: Vec<f32>,
    pub max_iter: usize,
    pub tol: f32,
}

impl KarcherMerge {
    pub fn new(weights: Vec<f32>, max_iter: usize, tol: f32) -> Self {
        Self { weights, max_iter: max_iter.max(1), tol }
    }
}

/// Spherical log map at base point `mu` (unit-norm assumed via normalization).
fn log_map(mu: &[f32], p: &[f32]) -> Vec<f32> {
    let dot: f32 = mu.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
    let theta = dot.clamp(-1.0, 1.0).acos();
    if theta < 1e-9 {
        return vec![0.0; mu.len()];
    }
    let s = theta / theta.sin();
    mu.iter().zip(p.iter()).map(|(m, x)| s * (x - dot * m)).collect()
}

/// Spherical exp map at base point `mu` along tangent `v`.
fn exp_map(mu: &[f32], v: &[f32]) -> Vec<f32> {
    let nv: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if nv < 1e-12 {
        return mu.to_vec();
    }
    let (s, c) = nv.sin_cos();
    mu.iter().zip(v.iter()).map(|(m, x)| c * m + s * x / nv).collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

impl MergeOp for KarcherMerge {
    fn merge_tensor(&self, _name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        Ok(vec![0.0f32; meta.num_elements()])
    }

    fn merge_tensors(&self, _name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            anyhow::bail!("karcher: no input tensors");
        }
        let n = meta.num_elements();
        for (i, t) in inputs.iter().enumerate() {
            if t.len() != n {
                anyhow::bail!("karcher: input {} size mismatch", i);
            }
        }
        // Work on the unit sphere: normalize copies (scale restored at the end
        // via the weighted mean norm so magnitudes stay sane).
        let scales: Vec<f32> = inputs.iter().map(|t| norm(t).max(1e-12)).collect();
        let w = normalize_weights(&self.weights, inputs.len());
        let mean_scale: f32 = scales.iter().zip(w.iter()).map(|(s, x)| s * x).sum();
        let unit: Vec<Vec<f32>> = inputs
            .iter()
            .zip(scales.iter())
            .map(|(t, s)| t.iter().map(|x| x / s).collect())
            .collect();

        // Init at normalized weighted linear mean
        let mut mu = vec![0.0f32; n];
        for (t, x) in unit.iter().zip(w.iter()) {
            for (m, v) in mu.iter_mut().zip(t.iter()) {
                *m += x * v;
            }
        }
        let nm = norm(&mu).max(1e-12);
        for m in mu.iter_mut() {
            *m /= nm;
        }

        for _ in 0..self.max_iter {
            let mut tangent = vec![0.0f32; n];
            for (t, x) in unit.iter().zip(w.iter()) {
                let l = log_map(&mu, t);
                for (a, b) in tangent.iter_mut().zip(l.iter()) {
                    *a += x * b;
                }
            }
            if norm(&tangent) < self.tol {
                break;
            }
            mu = exp_map(&mu, &tangent);
        }

        for m in mu.iter_mut() {
            *m *= mean_scale;
        }
        Ok(mu)
    }
}
