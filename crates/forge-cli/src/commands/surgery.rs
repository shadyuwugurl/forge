use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use forge_core::{ModelProfile, TensorStore, TensorMeta, FamilyRegistry, TensorMap};
use forge_io::TensorStore as _;
use forge_darwin::OrcaAllocator;
use forge_merge::ExpertWeaver;
use forge_surgery::{EncoderFuser, EncoderFuseConfig};

pub fn run(
    action: String,
    model: String,
    stats: Option<PathBuf>,
    threshold: f32,
    num_experts: usize,
    teacher: Option<PathBuf>,
    temperature: f32,
    decoder: Option<PathBuf>,
    encoders: Vec<PathBuf>,
    heads: usize,
    max_pairs: usize,
    stride: usize,
    output: Option<PathBuf>,
) -> Result<()> {
    let action = match action.as_str() {
        "orca" => SurgeryAction::Orca,
        "sparsify" => SurgeryAction::Sparsify,
        "densify" => SurgeryAction::Densify,
        "encoder-fuse" => SurgeryAction::EncoderFuse,
        _ => return Err(anyhow::anyhow!("Unknown surgery action: {}", action)),
    };
    
    match action {
        SurgeryAction::Orca => {
            let stats_path = stats.context("--stats required for orca")?;
            let allocator = OrcaAllocator::compute(stats_path, threshold)?;
            eprintln!("ORCA allocator computed from {}", stats_path.display());
            eprintln!("  Threshold: {}", threshold);
            eprintln!("  Gamma: {}", allocator.gamma);
            eprintln!("  Alpha attn: {}", allocator.alpha_attn);
            eprintln!("  Alpha ffn: {}", allocator.alpha_ffn);
            eprintln!("  Alpha emb: {}", allocator.alpha_emb);
            eprintln!("  Tau: {}", allocator.tau);
            // Print sample weights for first few tensors
            eprintln!("\nSample tensor weights:");
            for i in 0..10.min(allocator.weights.len()) {
                eprintln!("  tensor[{}] = {:.4}", i, allocator.weights[i]);
            }
        }
        SurgeryAction::Sparsify => {
            let model_path = Path::new(&model);
            let store = TensorStore::open(model_path)?;
            
            // Collect tensor shapes for profile detection
            let mut tensor_shapes = Vec::new();
            for name in store.tensor_names() {
                if let Ok(meta) = store.tensor_meta(name) {
                    tensor_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let profile = ModelProfile::detect(model_path, &tensor_shapes)?;
            let tensor_names: Vec<String> = tensor_shapes.iter().map(|(n, _)| n.clone()).collect();
            let tensor_map = TensorMap::build(&profile, &tensor_names, &registry)?;
            
            // Dry-run GLU count
            let weaver = ExpertWeaver::new(num_experts, None);
            eprintln!("Sparsify dry-run on {}", model_path.display());
            eprintln!("  Family: {}", profile.family);
            eprintln!("  Layers: {}", profile.num_layers);
            eprintln!("  Target experts: {}", num_experts);
            
            // Check for GLU pattern in expert tensors
            let glu_count = tensor_names.iter()
                .filter(|n| n.contains("gate_proj") || n.contains("up_proj"))
                .count();
            eprintln!("  GLU projections found: {}", glu_count);
            eprintln!("  Would route {} experts per layer", num_experts);
        }
        SurgeryAction::Densify => {
            let model_path = Path::new(&model);
            let teacher_path = teacher.context("--teacher required for densify")?;
            
            let store = TensorStore::open(model_path)?;
            let teacher_store = TensorStore::open(&teacher_path)?;
            
            // Collect tensor shapes
            let mut tensor_shapes = Vec::new();
            for name in store.tensor_names() {
                if let Ok(meta) = store.tensor_meta(name) {
                    tensor_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let profile = ModelProfile::detect(model_path, &tensor_shapes)?;
            let tensor_names: Vec<String> = tensor_shapes.iter().map(|(n, _)| n.clone()).collect();
            let tensor_map = TensorMap::build(&profile, &tensor_names, &registry)?;
            
            eprintln!("Densify plan: {} -> {}", model_path.display(), teacher_path.display());
            eprintln!("  Student family: {}", profile.family);
            eprintln!("  Temperature: {}", temperature);
            eprintln!("  Student tensors: {}", tensor_names.len());
            
            // Check teacher tensors
            let mut teacher_shapes = Vec::new();
            for name in teacher_store.tensor_names() {
                if let Ok(meta) = teacher_store.tensor_meta(name) {
                    teacher_shapes.push((name.to_string(), meta.shape));
                }
            }
            eprintln!("  Teacher tensors: {}", teacher_shapes.len());
        }
        SurgeryAction::EncoderFuse => {
            let decoder_path = decoder.context("--decoder required for encoder-fuse")?;
            if encoders.is_empty() {
                return Err(anyhow::anyhow!("At least one encoder required"));
            }
            
            // Open decoder store
            let decoder_store = TensorStore::open(&decoder_path)?;
            
            // Collect decoder tensor shapes
            let mut decoder_shapes = Vec::new();
            for name in decoder_store.tensor_names() {
                if let Ok(meta) = decoder_store.tensor_meta(name) {
                    decoder_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let decoder_profile = ModelProfile::detect(&decoder_path, &decoder_shapes)?;
            let decoder_names: Vec<String> = decoder_shapes.iter().map(|(n, _)| n.clone()).collect();
            let decoder_map = TensorMap::build(&decoder_profile, &decoder_names, &registry)?;
            
            // Build encoder tensor maps
            let mut encoder_maps = Vec::new();
            for enc_path in &encoders {
                let enc_store = TensorStore::open(enc_path)?;
                let mut enc_shapes = Vec::new();
                for name in enc_store.tensor_names() {
                    if let Ok(meta) = enc_store.tensor_meta(name) {
                        enc_shapes.push((name.to_string(), meta.shape));
                    }
                }
                let enc_profile = ModelProfile::detect(enc_path, &enc_shapes)?;
                let enc_names: Vec<String> = enc_shapes.iter().map(|(n, _)| n.clone()).collect();
                let enc_map = TensorMap::build(&enc_profile, &enc_names, &registry)?;
                encoder_maps.push(enc_map);
            }
            
            // Attach encoders
            let config = EncoderFuseConfig {
                heads,
                dim: 0,
                gate_init: 1e-3,
                max_pairs,
                stride,
                freeze_encoders: true,
            };
            
            let plan = EncoderFuser::attach(&decoder_map, &encoder_maps, &config)?;
            
            eprintln!("Encoder fusion plan:");
            eprintln!("  Decoder: {}", decoder_path.display());
            eprintln!("  Encoders: {}", encoders.len());
            eprintln!("  Heads: {}", plan.heads);
            eprintln!("  Cross-attn dim: {}", plan.dim);
            eprintln!("  Fused pairs: {}", plan.pairs.len());
            
            for (i, pair) in plan.pairs.iter().enumerate() {
                eprintln!("  Pair {}: decoder b{}.{} <- encoder{} b{}.{} (gate={:.4})", 
                    i, pair.decoder_block, pair.decoder_block, 
                    pair.encoder_idx, pair.encoder_block, pair.gate);
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub enum SurgeryAction {
    Orca,
    Sparsify,
    Densify,
    EncoderFuse,
}