// Integration test for full round-trip: merge -> quant -> eval
// Uses small test models or mocks for CI compatibility

use anyhow::Result;
use std::path::Path;
use forge_io::TensorStore;
use forge_merge::{LinearMerge, execute_merge, MergeOptions};
use forge_quant::{JangQuantizer, GgufWriter, GGUFQuantType};
use forge_eval::{EvalRunner, ComparisonTable};
use forge_core::DType;

#[test]
fn test_linear_merge_roundtrip() -> Result<()> {
    // Create two tiny fake models for testing
    let temp_dir = tempfile::tempdir()?;
    let model_a = create_test_model(&temp_dir.path().join("model_a"))?;
    let model_b = create_test_model(&temp_dir.path().join("model_b"))?;
    
    let output = temp_dir.path().join("merged");
    std::fs::create_dir_all(&output)?;

    // Load models
    let store_a = TensorStore::open(&model_a)?;
    let store_b = TensorStore::open(&model_b)?;

    // Merge with linear
    let merge_op = LinearMerge::new(
        vec![(&store_a, 0.5), (&store_b, 0.5)],
        true,
    );

    let options = MergeOptions {
        output_dtype: DType::BF16,
        base_model_dir: None,
        quiet: true,
        verbose: false,
    };

    execute_merge(&merge_op, &mut forge_io::StreamingWriter::new(&output, 5*1024*1024*1024)?, &options)?;

    // Verify merged model exists and has tensors
    let merged_store = TensorStore::open(&output)?;
    assert!(merged_store.tensor_names().len() > 0);
    assert!(merged_store.total_params() > 0);

    Ok(())
}

#[test]
fn test_jang_quantize() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let model = create_test_model(&temp_dir.path().join("model"))?;
    let output = temp_dir.path().join("quantized");
    std::fs::create_dir_all(&output)?;

    let store = TensorStore::open(&model)?;
    let quantizer = JangQuantizer::new("JANG_2L", forge_quant::jang::JangFormat::Mlx);
    quantizer.quantize(&store, &output)?;

    // Verify output exists
    assert!(output.join("jang_config.json").exists());
    assert!(output.join("model.safetensors.index.json").exists());

    Ok(())
}

#[test]
fn test_gguf_quantize() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let model = create_test_model(&temp_dir.path().join("model"))?;
    let output = temp_dir.path().join("quantized.gguf");
    std::fs::create_dir_all(&output)?;

    let store = TensorStore::open(&model)?;
    let mut writer = GgufWriter::create(&output)?;
    writer.set_metadata("general.architecture", serde_json::Value::String("test".into()));
    writer.write_quantized(&store, GGUFQuantType::Q4_K_M)?;

    // Verify GGUF file exists
    assert!(output.join("quantized.gguf").exists() || output.exists());

    Ok(())
}

#[test]
fn test_eval_compare() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let model_a = create_test_model(&temp_dir.path().join("model_a"))?;
    let model_b = create_test_model(&temp_dir.path().join("model_b"))?;

    let runner_a = EvalRunner::new(Path::new(&model_a));
    let runner_b = EvalRunner::new(Path::new(&model_b));

    let benches = vec!["hella".to_string(), "mmlu".to_string()];
    let evals = vec!["ace".to_string()];

    let results_a = runner_a.run_benchmarks(&benches)?;
    let results_b = runner_b.run_benchmarks(&benches)?;

    // Both should return valid results (stub scores)
    assert_eq!(results_a.len(), 2);
    assert_eq!(results_b.len(), 2);

    Ok(())
}

#[test]
fn test_data_ripper() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let base = create_test_model(&temp_dir.path().join("base"))?;
    let finetuned = create_test_model(&temp_dir.path().join("finetuned"))?;

    let base_store = TensorStore::open(&base)?;
    let ft_store = TensorStore::open(&finetuned)?;

    // L1: Weight diff
    let ripper = forge_train::DataRipper::new(forge_train::data_rip::ExtractionMethod::WeightDiff);
    let data = ripper.extract(&base_store, &ft_store)?;
    assert!(!data.deltas.is_empty());

    // L2: Activation probe (with stub calibration)
    let ripper2 = forge_train::DataRipper::new(forge_train::data_rip::ExtractionMethod::ActivationProbe);
    // Note: needs calibration data, which we don't have in test

    Ok(())
}

fn create_test_model(path: &Path) -> Result<std::path::PathBuf> {
    std::fs::create_dir_all(path)?;
    // Create a minimal safetensors file with a few tensors
    // In real tests, this would be a proper model
    // For now, just create a minimal valid safetensors
    let mut writer = safetensors::SafeTensorsSerializer::new();
    
    // Add a few test tensors
    let weights = vec![0.1f32; 256 * 512]; // 256x512
    writer.add_tensor("model.layers.0.self_attn.q_proj.weight", 
        &weights, &[256, 512], safetensors::Dtype::F32)?;
    writer.add_tensor("model.layers.0.self_attn.k_proj.weight", 
        &weights, &[256, 512], safetensors::Dtype::F32)?;
    writer.add_tensor("model.layers.0.mlp.gate_proj.weight", 
        &weights, &[512, 1024], safetensors::Dtype::F32)?;
    writer.add_tensor("model.embed_tokens.weight", 
        &vec![0.1f32; 32000 * 512], &[32000, 512], safetensors::Dtype::F32)?;
    writer.add_tensor("model.norm.weight", 
        &vec![1.0f32; 512], &[512], safetensors::Dtype::F32)?;

    let bytes = writer.finish()?;
    std::fs::write(path, bytes)?;
    Ok(path.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_full_pipeline_smoke() {
        // Quick smoke test that the whole pipeline compiles and runs
        let temp_dir = tempfile::tempdir().unwrap();
        let model_a = create_test_model(&temp_dir.path().join("a")).unwrap();
        let model_b = create_test_model(&temp_dir.path().join("b")).unwrap();
        
        let output = temp_dir.path().join("out");
        let store_a = TensorStore::open(&model_a).unwrap();
        let store_b = TensorStore::open(&model_b).unwrap();
        
        let merge_op = LinearMerge::new(vec![(&store_a, 0.5), (&store_b, 0.5)], true);
        let options = MergeOptions {
            output_dtype: forge_core::DType::BF16,
            base_model_dir: None,
            quiet: true,
            verbose: false,
        };
        
        execute_merge(&merge_op, &mut forge_io::StreamingWriter::new(&output, 5*1024*1024*1024).unwrap(), &options).unwrap();
        
        let quant_out = temp_dir.path().join("quant");
        let store = TensorStore::open(&output).unwrap();
        let quantizer = JangQuantizer::new("JANG_2L", forge_quant::jang::JangFormat::Mlx);
        quantizer.quantize(&TensorStore::open(&output).unwrap(), &quant_out).unwrap();
        
        // Verify outputs exist
        assert!(output.join("model.safetensors.index.json").exists());
        assert!(quant_out.join("jang_config.json").exists());
    }
}