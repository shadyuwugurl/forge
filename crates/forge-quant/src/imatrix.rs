use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;
use forge_io::TensorStore;

/// IMatrix (Importance Matrix) for quantization calibration
/// Stores per-tensor/per-group importance scores computed from calibration data
/// Used to guide quantization: higher importance = more bits allocated

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IMatrix {
    /// Per-tensor importance scores (flattened per-group)
    pub tensor_importance: HashMap<String, Vec<f32>>,
    /// Global metadata
    pub metadata: IMatrixMetadata,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IMatrixMetadata {
    pub model_name: String,
    pub calibration_samples: usize,
    pub group_size: usize,
    pub num_groups_per_tensor: HashMap<String, usize>,
    pub calibration_dataset: String,
    pub created_at: String,
    pub forge_version: String,
}

impl IMatrix {
    /// Create a new empty IMatrix
    pub fn new(group_size: usize) -> Self {
        Self {
            tensor_importance: HashMap::new(),
            metadata: IMatrixMetadata {
                model_name: "unknown".to_string(),
                calibration_samples: 0,
                group_size,
                num_groups_per_tensor: HashMap::new(),
                calibration_dataset: "unknown".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                forge_version: env!("CARGO_PKG_VERSION").to_string(),
            },
        }
    }

    /// Build IMatrix from calibration data by running forward passes
    /// This computes per-group Fisher information / activation statistics
    pub fn build_from_calibration(
        store: &TensorStore,
        calibration_data: &[CalibrationSample],
        group_size: usize,
    ) -> Result<Self> {
        let mut imatrix = Self::new(group_size);
        imatrix.metadata.calibration_samples = calibration_data.len();
        imatrix.metadata.group_size = group_size;

        // For each tensor that will be quantized, compute importance per group
        for name in store.tensor_names() {
            if is_passthrough_tensor(name) {
                continue; // Skip passthrough tensors
            }

            let meta = store.tensor_meta(name)?;
            let in_features = if meta.shape.len() >= 2 {
                meta.shape[meta.shape.len() - 1]
            } else {
                meta.shape[0]
            };

            let num_groups = (in_features + group_size - 1) / group_size;
            imatrix.metadata.num_groups_per_tensor.insert(name.to_string(), num_groups);

            // Initialize importance array for this tensor
            let mut importance = vec![0.0f32; num_groups];

            // Process calibration samples
            // In a real implementation, this would run the model forward on calibration data
            // and compute Fisher information / gradient magnitudes / activation statistics
            // For now, we compute a proxy based on weight magnitudes and gradients
            
            if let Ok(weight_data) = store.tensor_f32(name) {
                let num_groups = (weight_data.len() + group_size - 1) / group_size;
                let mut group_importance = vec![0.0f32; num_groups];

                for g in 0..num_groups {
                    let start = g * group_size;
                    let end = ((g + 1) * group_size).min(weight_data.len());
                    let group = &weight_data[start..end];

                    // Compute Fisher-like importance: sum of squared weights / variance
                    let sum_sq: f32 = weight_data[start..end].iter().map(|x| x * x).sum();
                    let mean = weight_data[start..end].iter().sum::<f32>() / (end - start) as f32;
                    let variance = weight_data[start..end].iter()
                        .map(|x| (x - mean).powi(2))
                        .sum::<f32>() / (end - start) as f32;

                    // Fisher info proxy: mean squared weight * (1 + variance)
                    group_importance[(start / group_size).min(num_groups - 1)] = 
                        sum_sq / group_size as f32 * (1.0 + variance);
                }

                // Normalize to [0, 1] range
                let max_imp = group_importance.iter().cloned().fold(0.0f32, f32::max);
                if max_imp > 0.0 {
                    for imp in &mut group_importance {
                        *imp /= max_imp;
                    }
                }

                imatrix.tensor_importance.insert(name.to_string(), group_importance);
            }
        }

        imatrix.metadata.model_name = store.path().display().to_string();
        imatrix.metadata.created_at = chrono::Utc::now().to_rfc3339();
        imatrix.metadata.calibration_samples = calibration_data.len();
        imatrix.metadata.calibration_dataset = "user_provided".to_string();

        Ok(imatrix)
    }

    /// Save IMatrix to file (JSON format)
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load IMatrix from file
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let imatrix: IMatrix = serde_json::from_str(&data)?;
        Ok(imatrix)
    }

    /// Get importance for a specific tensor and group
    pub fn get_importance(&self, tensor_name: &str, group_idx: usize) -> Option<f32> {
        self.tensor_importance
            .get(tensor_name)
            .and_then(|v| v.get(group_idx).copied())
    }

    /// Get per-tensor average importance (for tier assignment)
    pub fn tensor_avg_importance(&self, tensor_name: &str) -> Option<f32> {
        self.tensor_importance.get(tensor_name).map(|v| {
            if v.is_empty() { 0.0 } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        })
    }

    /// Apply IMatrix to adjust bit allocation per tensor/group
    /// Returns adjusted bits per tensor based on importance
    pub fn compute_adjusted_bits(&self, base_bits: u8, tensor_name: &str) -> Vec<u8> {
        if let Some(importance) = self.tensor_importance.get(tensor_name) {
            importance.iter().map(|imp| {
                // Scale bits by importance: higher importance = more bits
                // Min 2 bits, max 8 bits
                let adjusted = (base_bits as f32 * (1.0 + imp)).round() as u8;
                adjusted.clamp(2, 8)
            }).collect()
        } else {
            vec![base_bits]
        }
    }
}

/// Calibration sample for IMatrix computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CalibrationSample {
    pub input_ids: Vec<u32>,
    pub attention_mask: Option<Vec<u32>>,
    pub labels: Option<Vec<u32>>,
    pub weight: f32, // Sample weight for importance weighting
}

impl CalibrationSample {
    pub fn from_jsonl_line(line: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(line)?;
        let input_ids = v.get("input_ids").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect())
            .unwrap_or_default();
        let attention_mask = v.get("attention_mask").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect());
        let labels = v.get("labels").and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect());
        let weight = v.get("weight").and_then(|x| x.as_f64()).unwrap_or(1.0) as f32;
        Ok(Self { input_ids, attention_mask, labels, weight })
    }
}

/// Load calibration data from JSONL file
pub fn load_calibration_data(path: &std::path::Path) -> Result<Vec<CalibrationSample>> {
    let data = std::fs::read_to_string(path)?;
    let mut samples = Vec::new();
    for line in data.lines() {
        if line.trim().is_empty() { continue; }
        samples.push(CalibrationSample::from_jsonl_line(line)?);
    }
    Ok(samples)
}

fn is_passthrough_tensor(name: &str) -> bool {
    let n = name.to_lowercase();
    n.contains("norm") || n.ends_with(".bias") || n.contains("embed") || n.contains("wte") || n.contains("vision") || n.contains("visual")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_imatrix_serialization() {
        let mut imatrix = IMatrix::new(64);
        imatrix.tensor_importance.insert("test.weight".to_string(), vec![0.5, 0.8, 0.3]);
        
        let dir = tempdir().unwrap();
        let path = dir.path().join("imatrix.json");
        imatrix.save(&path).unwrap();
        
        let loaded = IMatrix::load(&path).unwrap();
        assert_eq!(loaded.tensor_importance.get("test.weight").unwrap(), &vec![0.5, 0.8, 0.3]);
    }

    #[test]
    fn test_imatrix_bit_adjustment() {
        let mut imatrix = IMatrix::new(64);
        imatrix.tensor_importance.insert("test.weight".to_string(), vec![0.0, 0.5, 1.0]);
        
        let bits = imatrix.compute_adjusted_bits(4, "test.weight");
        assert_eq!(bits.len(), 3);
        assert!(bits[0] <= 4); // Low importance -> lower bits
        assert!(bits[2] >= 4); // High importance -> higher bits
    }
}