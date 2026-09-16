use std::path::{Path, PathBuf};
use anyhow::{Result, Context};
use forge_core::{ModelProfile, TensorMeta, FamilyRegistry, TensorMap};
use forge_io::TensorStore;
use forge_darwin::{OrcaAllocator, AuditReport, ActivationDump, audit_layers};
use forge_merge::{ExpertWeaver, ChimeraMerge, ChimeraRouter, greedy_diverse_select, mixed_precision_plan};
use forge_surgery::{EncoderFuser, EncoderFuseConfig};
use forge_eval::{EvalRunner, ArmorGate, armor_bundle, evaluate};

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
    parents: Vec<PathBuf>,
    router_out: Option<PathBuf>,
    keep: usize,
    clean: Option<PathBuf>,
    prefixed: Option<PathBuf>,
    audit_threshold: f32,
    chimera_threshold: f32,
) -> Result<()> {
    let action = match action.as_str() {
        "orca" => SurgeryAction::Orca,
        "sparsify" => SurgeryAction::Sparsify,
        "densify" => SurgeryAction::Densify,
        "encoder-fuse" => SurgeryAction::EncoderFuse,
        "chimera" => SurgeryAction::Chimera,
        "audit" => SurgeryAction::Audit,
        "armor" => SurgeryAction::Armor,
        "pocket" => SurgeryAction::Pocket,
        _ => return Err(anyhow::anyhow!("Unknown surgery action: {}", action)),
    };
    
    match action {
        SurgeryAction::Orca => {
            let stats_path = stats.context("--stats required for orca")?;
            let allocator = OrcaAllocator::compute(Some(stats_path.as_path()), threshold)?;
            eprintln!("ORCA allocator computed from {}", stats_path.display());
            eprintln!("  Threshold: {}", threshold);
            // Print sample weights for first few tensors
            eprintln!("\nSample tensor weights:");
            let mut count = 0;
            for (name, ratio) in allocator.ratios.iter() {
                if count >= 10 { break; }
                let weight = allocator.weight_for(name);
                eprintln!("  {}: ratio={:.4}, weight={:.4}", name, ratio, weight);
                count += 1;
            }
        }
        SurgeryAction::Sparsify => {
            let model_path = Path::new(&model);
            let (store_file, config_dir) = super::resolve_model(model_path);
            let store = TensorStore::open(&store_file)?;
            
            // Collect tensor shapes for profile detection
            let mut tensor_shapes = Vec::new();
            for name in store.tensor_names() {
                if let Ok(meta) = store.tensor_meta(name) {
                    tensor_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let profile = ModelProfile::detect(&config_dir, &tensor_shapes)?;
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
            
            let (store_file, config_dir) = super::resolve_model(model_path);
            let store = TensorStore::open(&store_file)?;
            let (teacher_file, _) = super::resolve_model(&teacher_path);
            let teacher_store = TensorStore::open(&teacher_file)?;
            
            // Collect tensor shapes
            let mut tensor_shapes = Vec::new();
            for name in store.tensor_names() {
                if let Ok(meta) = store.tensor_meta(name) {
                    tensor_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let profile = ModelProfile::detect(&config_dir, &tensor_shapes)?;
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
            let (decoder_file, decoder_dir) = super::resolve_model(&decoder_path);
            let decoder_store = TensorStore::open(&decoder_file)?;
            
            // Collect decoder tensor shapes
            let mut decoder_shapes = Vec::new();
            for name in decoder_store.tensor_names() {
                if let Ok(meta) = decoder_store.tensor_meta(name) {
                    decoder_shapes.push((name.to_string(), meta.shape));
                }
            }
            
            let registry = FamilyRegistry::builtin();
            let decoder_profile = ModelProfile::detect(&decoder_dir, &decoder_shapes)?;
            let decoder_names: Vec<String> = decoder_shapes.iter().map(|(n, _)| n.clone()).collect();
            let decoder_map = TensorMap::build(&decoder_profile, &decoder_names, &registry)?;
            
            // Build encoder tensor maps
            let mut encoder_maps = Vec::new();
            for enc_path in &encoders {
                let (enc_file, enc_dir) = super::resolve_model(enc_path);
                let enc_store = TensorStore::open(&enc_file)?;
                let mut enc_shapes = Vec::new();
                for name in enc_store.tensor_names() {
                    if let Ok(meta) = enc_store.tensor_meta(name) {
                        enc_shapes.push((name.to_string(), meta.shape));
                    }
                }
                let enc_profile = ModelProfile::detect(&enc_dir, &enc_shapes)?;
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
                eprintln!("  Pair {}: decoder b{} <- encoder{} b{} (gate={:.4})",
                    i, pair.decoder_block,
                    pair.encoder_idx, pair.encoder_block, pair.gate);
            }
        }
        SurgeryAction::Chimera => {
            if parents.len() < 2 {
                return Err(anyhow::anyhow!("chimera needs >= 2 --parents"));
            }
            let merger = ChimeraMerge::new(chimera_threshold);
            let mut stores = Vec::new();
            for p in &parents {
                let (pf, _) = super::resolve_model(p);
                stores.push(TensorStore::open(&pf)?);
            }
            // Union of tensor names across parents (streaming: one tensor at a time).
            let mut names = std::collections::HashSet::new();
            for s in &stores {
                for n in s.tensor_names() {
                    names.insert(n.to_string());
                }
            }
            let mut n_avg = 0usize;
            let mut n_transplant = 0usize;
            let mut decisions = Vec::new();
            for name in &names {
                let mut inputs = Vec::new();
                for s in &stores {
                    if let Ok(v) = s.tensor_f32(name) {
                        inputs.push(v);
                    }
                }
                if inputs.len() < 2 {
                    n_transplant += 1;
                    decisions.push(serde_json::json!({"tensor": name, "score": 0.0, "decision": "transplant-single-holder"}));
                    continue;
                }
                let (score, d) = merger.decide_tensors(&inputs[0], &inputs[1]);
                let label = match d {
                    forge_merge::MergeDecision::Average => { n_avg += 1; "average" }
                    forge_merge::MergeDecision::Transplant => { n_transplant += 1; "transplant" }
                };
                decisions.push(serde_json::json!({"tensor": name, "score": score, "decision": label}));
            }
            let router = ChimeraRouter::new(stores.len());
            eprintln!("Chimera plan over {} parents (threshold={})", stores.len(), chimera_threshold);
            eprintln!("  Tensors scored: {}", names.len());
            eprintln!("  Average (crossbreed): {}", n_avg);
            eprintln!("  Transplant: {}", n_transplant);
            eprintln!("  Router init logits: {:?}", router.logits);
            let doc = serde_json::json!({
                "threshold": chimera_threshold,
                "n_parents": stores.len(),
                "router_logits": router.logits,
                "n_average": n_avg,
                "n_transplant": n_transplant,
                "decisions": decisions,
            });
            if let Some(path) = router_out.as_ref().or(output.as_ref()) {
                std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
                eprintln!("  Router plan written to {}", path.display());
            }
        }
        SurgeryAction::Pocket => {
            let model_path = Path::new(&model);
            let (store_file, config_dir) = super::resolve_model(model_path);
            let store = TensorStore::open(&store_file)?;
            // Group expert tensors per (layer, expert): `...experts.<E>.<proj>...`,
            // layer = first numeric path segment.
            let mut experts: std::collections::HashMap<(usize, usize), (f64, Vec<f32>)> =
                std::collections::HashMap::new();
            for name in store.tensor_names() {
                let segs: Vec<&str> = name.split('.').collect();
                let ei = segs.iter().position(|s| *s == "experts");
                let (layer, expert) = match ei {
                    Some(i) if i + 1 < segs.len() => {
                        let e: usize = match segs[i + 1].parse() { Ok(v) => v, Err(_) => continue };
                        let l = segs.iter().take(i).find_map(|s| s.parse::<usize>().ok()).unwrap_or(0);
                        (l, e)
                    }
                    _ => continue,
                };
                let w = store.tensor_f32(name)?;
                let entry = experts.entry((layer, expert)).or_insert((0.0, Vec::new()));
                entry.0 += w.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>();
                // Downsampled signature (every 997th element) for diversity distance.
                for (j, v) in w.iter().enumerate().step_by(997) {
                    let _ = j;
                    entry.1.push(*v);
                }
            }
            if experts.is_empty() {
                return Err(anyhow::anyhow!("pocket found no `experts.<N>` tensors in {}", model_path.display()));
            }
            let mut layers: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
            for (layer, expert) in experts.keys() {
                layers.entry(*layer).or_default().push(*expert);
            }
            let mut layer_ids: Vec<usize> = layers.keys().copied().collect();
            layer_ids.sort_unstable();
            let mut plan_layers = Vec::new();
            for l in &layer_ids {
                let mut ids = layers[l].clone();
                ids.sort_unstable();
                let scores: Vec<f32> = ids.iter().map(|e| (experts[&(*l, *e)].0 as f32).sqrt()).collect();
                let sigs: Vec<Vec<f32>> = ids.iter().map(|e| experts[&(*l, *e)].1.clone()).collect();
                let kept_pos = greedy_diverse_select(&sigs, &scores, keep);
                let kept: Vec<usize> = kept_pos.iter().map(|p| ids[*p]).collect();
                let dropped: Vec<usize> = ids.iter().filter(|e| !kept.contains(e)).copied().collect();
                eprintln!("  Layer {}: {} experts -> keep {} drop {}", l, ids.len(), kept.len(), dropped.len());
                plan_layers.push(serde_json::json!({"layer": l, "kept": kept, "dropped": dropped}));
            }
            let mix = mixed_precision_plan(&[("ko-hard", 1.0), ("en-hard", 0.8), ("base", 0.0)], 2);
            eprintln!("  Mixed-precision plan: {:?}", mix);
            if let Some(path) = output.as_ref() {
                let doc = serde_json::json!({"keep": keep, "mixed_precision": mix, "layers": plan_layers});
                std::fs::write(path, serde_json::to_string_pretty(&doc)?)?;
                eprintln!("  Pocket plan written to {}", path.display());
            }
        }
        SurgeryAction::Audit => {
            match (clean.as_ref(), prefixed.as_ref()) {
                (Some(c), Some(p)) => {
                    let cd = ActivationDump::load(c)?;
                    let pd = ActivationDump::load(p)?;
                    let audits = audit_layers(&cd.layers, &pd.layers)?;
                    let report = AuditReport::new(audits, audit_threshold);
                    eprintln!("Prefix audit over {} layers (threshold={})", report.layers.len(), audit_threshold);
                    for a in &report.layers {
                        eprintln!("  layer {:>3}: score {:.4}{}", a.layer, a.score,
                            if report.faulty.contains(&a.layer) { "  FAULT" } else { "" });
                    }
                    if report.faulty.is_empty() {
                        eprintln!("  No faulty layers.");
                    } else {
                        eprintln!("  Faulty layers: {:?}", report.faulty);
                    }
                    let out = output.clone().unwrap_or_else(|| PathBuf::from("audit_report.json"));
                    report.write_json(&out)?;
                    eprintln!("  Report written to {}", out.display());
                }
                _ => {
                    eprintln!("AX-Ray prefix-invariance gate (2-pass, no training):");
                    eprintln!("  Provide --clean and --prefixed activation dumps:");
                    eprintln!("    {{\"layers\": [[f32...], ...]}}  (clean[i] vs prefixed[i] per layer)");
                    eprintln!("  Layers scoring below --audit-threshold (default 0.99) are reported faulty.");
                }
            }
        }
        SurgeryAction::Armor => {
            let bundle = armor_bundle();
            let gate = ArmorGate::strict();
            let runner = EvalRunner::new(Path::new(&model));
            let scores: Vec<(String, f64)> = match runner.run_evals(&bundle) {
                Ok(results) => results.into_iter().map(|r| (r.name, r.score)).collect(),
                Err(e) => {
                    eprintln!("  harness error (offline?): {e}");
                    vec![]
                }
            };
            let report = evaluate(&model, &scores, &gate);
            eprintln!("Armor gate on {} (min_score={})", model, gate.min_score);
            for s in &report.scores {
                eprintln!("  {:<16} {:.3} {}", s.name, s.score, if s.passed { "PASS" } else { "FAIL" });
            }
            eprintln!("  Verdict: {}", if report.passed { "ARMORED" } else { "NOT ARMORED" });
            if let Some(n) = &report.note {
                eprintln!("  Note: {n}");
            }
            let out = output.clone().unwrap_or_else(|| PathBuf::from("armor_report.json"));
            report.write_json(&out)?;
            eprintln!("  Report written to {}", out.display());
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
    Chimera,
    Audit,
    Armor,
    Pocket,
}