use anyhow::Result;

/// Spherical linear interpolation between two equal-length vectors.
/// Falls back to linear blend for degenerate (zero-norm / near-parallel) inputs.
pub fn slerp_pair(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        anyhow::bail!("slerp: dimension mismatch ({} vs {})", a.len(), b.len());
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-12 || nb < 1e-12 {
        return Ok(a.iter().zip(b.iter()).map(|(x, y)| (1.0 - t) * x + t * y).collect());
    }
    let cos_theta = (dot / (na * nb)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    if theta < 1e-6 {
        return Ok(a.to_vec());
    }
    // Antipodal guard: lerp when sin(theta) ~ 0
    let sin_theta = theta.sin();
    if sin_theta.abs() < 1e-6 {
        return Ok(a.iter().zip(b.iter()).map(|(x, y)| (1.0 - t) * x + t * y).collect());
    }
    let w1 = ((1.0 - t) * theta).sin() / sin_theta;
    let w2 = (t * theta).sin() / sin_theta;
    Ok(a.iter().zip(b.iter()).map(|(x, y)| w1 * x + w2 * y).collect())
}

/// Normalize a weight vector to sum to 1 (uniform fallback for empty/zero-sum).
pub fn normalize_weights(weights: &[f32], n: usize) -> Vec<f32> {
    if weights.len() == n && weights.iter().sum::<f32>() > 1e-12 {
        let s: f32 = weights.iter().sum();
        return weights.iter().map(|w| w / s).collect();
    }
    vec![1.0 / n as f32; n]
}
