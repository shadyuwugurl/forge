use std::path::Path;
use anyhow::{Result, Context};
use clap::ValueEnum;
use forge_core::{MergeMethod, MergeConfig, TensorMeta, DType, MemoryGuard};
use forge_io::{TensorStore, StreamingWriter};
use forge_merge::{LinearMerge, LatentMerge, ExpertWeaver, MoeDenseDistill, HeteroMerge, HeteroMode, ChimeraMerge, PocketPrune, AetherRemap};
use forge_merge::orchestrator::{execute_merge as orch_execute_merge, MergeOp, MergeOptions};

pub fn run(
    config: Option<&Path>,
    models: Option<&[std::path::PathBuf]>,
    method: Option<&str>,
    output: &Path,
    t: Option<f32>,
    generations: usize,
    population: usize,
    vae: Option<&Path>,
    latent_dim: usize,
    orca_stats: Option<&Path>,
    orca_threshold: f32,
    num_experts: usize,
    _shared_dim: Option<usize>,
    teacher: Option<&Path>,
    distill_temp: f32,
    hetero_mode: String,
    hetero_weights: Option<String>,
    nparent: usize,
    chimera_threshold: f32,
    aether_grid: usize,
    pocket_keep: usize,
    max_memory_gb: f32,
) -> Result<()> {
    // Memory guard check
    let guard = MemoryGuard::new(max_memory_gb as f64);
    let shard_size = 64 * 1024 * 1024; // 64MB shard buffer
    
    if let Some(config_path) = config {
        let config_str = std::fs::read_to_string(config_path)?;
        let merge_config: MergeConfig = serde_yaml::from_str(&config_str)?;
        execute_config_merge(merge_config, output, &guard, shard_size, generations, population, nparent, max_memory_gb)
    } else if let Some(model_paths) = models {
        let method_name = method.unwrap_or("linear");
        
        // Parse hetero_weights from comma-separated string if provided
        let hetero_weights_vec: Option<Vec<f32>> = hetero_weights.as_ref().map(|s| {
            s.split(',')
                .filter_map(|w| w.trim().parse().ok())
                .collect()
        });
        
        let merge_method = parse_merge_method(t, method_name, vae, latent_dim, orca_stats, orca_threshold, 
                                              num_experts, teacher, distill_temp,
                                              &hetero_mode, hetero_weights_vec, nparent,
                                              chimera_threshold, aether_grid, pocket_keep)?;
        
        // Open all model stores
        let mut stores = Vec::new();
        for path in model_paths {
            let (sf, _) = super::resolve_model(path);
            let store = TensorStore::open(&sf)?;
            stores.push(store);
        }
        
        // Run memory guard check on first store
        if let Some(first_store) = stores.first() {
            // Check first tensor's size for memory estimate
            for name in first_store.tensor_names() {
                if let Ok(meta) = first_store.tensor_meta(name) {
                    let tensor_bytes = meta.size as u64; // meta.size is usize
                    let peak = guard.plan_merge(tensor_bytes, shard_size as u64);
                    eprintln!("Merge peak memory estimate: {:.2} GB (budget: {} GB) - {}", peak.peak_bytes as f64 / 1e9, max_memory_gb, if peak.fits { "OK" } else { "OVER" });
                    guard.check(tensor_bytes)?;
                    break;
                }
            }
        }
        
        execute_merge_impl(stores, merge_method, output, t, generations, population, max_memory_gb)
    } else {
        return Err(anyhow::anyhow!("Either --config or --models is required"));
    }
}

fn parse_merge_method(
    t: Option<f32>,
    method: &str,
    vae: Option<&Path>,
    latent_dim: usize,
    orca_stats: Option<&Path>,
    orca_threshold: f32,
    num_experts: usize,
    teacher: Option<&Path>,
    distill_temp: f32,
    hetero_mode: &str,
    hetero_weights: Option<Vec<f32>>,
    nparent: usize,
    chimera_threshold: f32,
    aether_grid: usize,
    pocket_keep: usize,
) -> Result<MergeMethod> {
    Ok(match method {
        "linear" => MergeMethod::Linear,
        "slerp" => MergeMethod::Slerp { t: t.unwrap_or(0.5) },
        "nuslerp" => MergeMethod::NuSlerp,
        "task_arithmetic" => MergeMethod::TaskArithmetic { lambda: t.unwrap_or(1.0) },
        "ties" => MergeMethod::Ties,
        "dare" => MergeMethod::Dare,
        "dare_ties" => MergeMethod::DareTies,
        "della_linear" => MergeMethod::DellaLinear,
        "della" => MergeMethod::Della,
        "passthrough" => MergeMethod::Passthrough,
        "darwin" => MergeMethod::Darwin { generations: 30, population: 40 },
        "frankenmerge" => MergeMethod::FrankenMerge,
        "model_stock" => MergeMethod::ModelStock,
        "breadcrumbs" => MergeMethod::Breadcrumbs,
        "nearswap" => MergeMethod::Nearswap,
        "ram" => MergeMethod::Ram,
        "latent" | "ls_merge" => MergeMethod::Latent {
            vae_path: vae.map(|p| p.to_path_buf()),
            latent_dim,
        },
        "orca" => MergeMethod::Orca {
            stats_path: orca_stats.map(|p| p.to_path_buf()),
            threshold: orca_threshold,
        },
        "expert_weaver" => MergeMethod::ExpertWeaver {
            num_experts,
        },
        "moe_dense_distill" => MergeMethod::MoeDenseDistill {
            teacher: teacher.map(|p| p.to_path_buf()),
            temperature: distill_temp,
        },
        "hetero" | "hetero_merge" | "hetero-merge" => MergeMethod::Hetero {
            mode: hetero_mode.to_string(),
            weights: hetero_weights.clone().unwrap_or_default(),
        },
        "chimera" => MergeMethod::Chimera {
            threshold: t.unwrap_or(chimera_threshold),
        },
        "aether" => MergeMethod::Aether {
            grid: aether_grid,
        },
        "pocket" => MergeMethod::Pocket {
            keep: pocket_keep,
        },
        _ => MergeMethod::Linear,
    })
}

fn execute_merge_impl(
    stores: Vec<TensorStore>,
    method: MergeMethod,
    output: &Path,
    _t: Option<f32>,
    generations: usize,
    population: usize,
    max_memory_gb: f32,
) -> Result<()> {
    if stores.is_empty() {
        return Err(anyhow::anyhow!("No models provided"));
    }
    
    // Create the merge op
    let n_models = stores.len();
    let op = create_merge_op(&method, n_models)?;
    
    // Use the orchestrator's execute_merge which handles streaming properly
    let stores_refs: Vec<&TensorStore> = stores.iter().collect();
    let options = MergeOptions {
        output_dtype: DType::F16,
        base_model_dir: None,
        quiet: false,
        verbose: true,
    };
    
    // Use the orchestrator's execute_merge which handles streaming properly
    orch_execute_merge(&*op, &stores_refs, output, &options)?;
    copy_sidecars(&stores, output);
    eprintln!("Output written to {}", output.display());
    Ok(())
}

/// Carry over sidecar files (config/tokenizer) from the first parent so the
/// output dir is self-describing for `inspect` and downstream loaders.
fn copy_sidecars(stores: &[TensorStore], output: &Path) {
    if let Some(first) = stores.first() {
        if let Some(parent_dir) = first.path().parent() {
            for sidecar in [
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
            ] {
                let src = parent_dir.join(sidecar);
                if src.exists() {
                    let _ = std::fs::copy(&src, output.join(sidecar));
                }
            }
        }
    }
}

fn execute_config_merge(
    config: MergeConfig,
    output: &Path,
    guard: &forge_core::MemoryGuard,
    shard_size: usize,
    generations: usize,
    population: usize,
    nparent: usize,
    max_memory_gb: f32,
) -> Result<()> {
    // Load all models from config
    let mut stores = Vec::new();
    for entry in &config.models {
        let (sf, _) = super::resolve_model(&entry.path);
        let store = TensorStore::open(&sf)?;
        stores.push(store);
    }
    
    // Memory guard check
    if let Some(first_store) = stores.first() {
        for name in first_store.tensor_names() {
            if let Ok(meta) = first_store.tensor_meta(name) {
                let tensor_bytes = meta.size as u64; // meta.size is usize
                let peak = guard.plan_merge(tensor_bytes, shard_size as u64);
                eprintln!("Merge peak memory estimate: {:.2} GB (budget: {} GB) - {}", peak.peak_bytes as f64 / 1e9, max_memory_gb, if peak.fits { "OK" } else { "OVER" });
                guard.check(tensor_bytes)?;
                break;
            }
        }
    }
    
    let method = config.merge_method.clone();
    let n_models = stores.len();
    let op = create_merge_op(&method, n_models)?;
    
    // Use the orchestrator's execute_merge which handles streaming properly
    let stores_refs: Vec<&TensorStore> = stores.iter().collect();
    let options = MergeOptions {
        output_dtype: DType::F16,
        base_model_dir: None,
        quiet: false,
        verbose: true,
    };
    
    orch_execute_merge(&*op, &stores_refs, output, &options)?;
    copy_sidecars(&stores, output);
    Ok(())
}

fn create_merge_op(method: &MergeMethod, n_models: usize) -> Result<Box<dyn MergeOp + Sync>> {
    // Use a helper to coerce each arm to Box<dyn MergeOp>
    fn linear_merge() -> Box<dyn MergeOp + Sync> {
        Box::new(LinearMerge { 
            models: vec![], 
            normalize: true 
        })
    }
    
    Ok(match method {
        MergeMethod::Linear => linear_merge(),
        MergeMethod::Slerp { t } => linear_merge(),
        MergeMethod::NuSlerp => linear_merge(),
        MergeMethod::TaskArithmetic { lambda } => linear_merge(),
        MergeMethod::Ties => linear_merge(),
        MergeMethod::Dare => linear_merge(),
        MergeMethod::DareTies => linear_merge(),
        MergeMethod::DellaLinear => linear_merge(),
        MergeMethod::Della => linear_merge(),
        MergeMethod::Passthrough => Box::new(LinearMerge { models: vec![], normalize: false }),
        MergeMethod::Darwin { generations, population } => linear_merge(),
        MergeMethod::FrankenMerge => linear_merge(),
        MergeMethod::ModelStock => linear_merge(),
        MergeMethod::Breadcrumbs => linear_merge(),
        MergeMethod::Nearswap => linear_merge(),
        MergeMethod::Ram => linear_merge(),
        MergeMethod::Latent { vae_path, latent_dim } => {
            Box::new(LatentMerge::new(vae_path.clone(), *latent_dim))
        }
        MergeMethod::Orca { stats_path, threshold } => linear_merge(),
        MergeMethod::ExpertWeaver { num_experts } => {
            Box::new(ExpertWeaver::new(*num_experts, None))
        }
        MergeMethod::MoeDenseDistill { teacher, temperature } => {
            Box::new(MoeDenseDistill::new(*temperature))
        }
        MergeMethod::Hetero { mode, weights } => {
            let mode = HeteroMode::parse(mode)?;
            if !weights.is_empty() {
                Box::new(HeteroMerge::with_weights(mode, weights.to_vec())?)
            } else {
                Box::new(HeteroMerge::new(mode, n_models))
            }
        }
        MergeMethod::Chimera { threshold } => {
            Box::new(ChimeraMerge::new(*threshold))
        }
        MergeMethod::Aether { grid } => {
            Box::new(AetherRemap::new(*grid))
        }
        MergeMethod::Pocket { keep } => {
            Box::new(PocketPrune::new(*keep))
        }
    })
}