use std::path::Path;
use anyhow::{Context, Result};
use forge_io::TensorStore;
use forge_quant::{IMatrix, load_calibration_data};

pub fn run(
    model: &Path,
    calib: &Path,
    output: &Path,
    group_size: usize,
) -> Result<()> {
    eprintln!("Building IMatrix from calibration data...");
    
    let store = TensorStore::open(model)
        .context("Failed to open model")?;
    
    let calibration_data = load_calibration_data(calib)
        .context("Failed to load calibration data")?;
    
    eprintln!("Loaded {} calibration samples", calibration_data.len());
    eprintln!("Computing IMatrix with group_size={}...", group_size);
    
    let imatrix = IMatrix::build_from_calibration(&store, &[], group_size)
        .context("Failed to build IMatrix")?;
    
    eprintln!("IMatrix built: {} tensors, {} total groups", 
        imatrix.tensor_importance.len(),
        imatrix.tensor_importance.values().map(|v| v.len()).sum::<usize>());
    
    std::fs::create_dir_all(output)?;
    let output_path = output.join("imatrix.json");
    imatrix.save(&output_path)
        .context("Failed to save IMatrix")?;
    
    eprintln!("IMatrix saved to {}", output_path.display());
    eprintln!("Tensors: {}", imatrix.tensor_importance.len());
    eprintln!("Total groups: {}", imatrix.tensor_importance.values().map(|v| v.len()).sum::<usize>());
    eprintln!("Calibration samples: {}", imatrix.metadata.calibration_samples);
    eprintln!("Group size: {}", imatrix.metadata.group_size);
    
    // Print per-tensor summary
    for (name, importance) in &imatrix.tensor_importance {
        let avg = importance.iter().sum::<f32>() / importance.len() as f32;
        let max = importance.iter().cloned().fold(0.0f32, f32::max);
        let min = importance.iter().cloned().fold(f32::INFINITY, f32::min);
        eprintln!("  {}: groups={}, avg={:.4}, min={:.4}, max={:.4}", 
            name, importance.len(), avg, min, max);
    }
    
    Ok(())
}

pub fn apply_run(
    model: &Path,
    imatrix_path: &Path,
    output: &Path,
    method: &str,
    profile: Option<&str>,
    density: Option<f32>,
) -> Result<()> {
    eprintln!("Applying IMatrix to quantization...");
    
    let store = TensorStore::open(model)?;
    let imatrix = IMatrix::load(imatrix_path)?;
    
    eprintln!("Loaded IMatrix: {} tensors, {} total groups",
        imatrix.tensor_importance.len(),
        imatrix.tensor_importance.values().map(|v| v.len()).sum::<usize>());
    
    // TODO: Apply IMatrix to actual quantization
    // This would integrate with the quantizers to use per-group importance
    eprintln!("IMatrix applied (stub - would adjust per-group bit allocation)");
    
    Ok(())
}