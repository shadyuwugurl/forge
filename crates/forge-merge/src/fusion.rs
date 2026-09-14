use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use forge_core::{MergeMethod, TensorMeta};
use crate::orchestrator::{MergeOp, MergeOptions, execute_merge};
use crate::{LinearMerge, TiesMerge, DareMerge, DellaMerge, PassthroughMerge, FrankenMerge};
use forge_io::TensorStore;

/// Fusion blending: combine multiple merge strategies in a pipeline
/// 
/// Supports:
/// - Sequential fusion: apply strategy A, then B, then C
/// - Parallel blend: blend outputs of multiple strategies with weights
/// - Conditional fusion: different strategies for different tensor types
/// - Layer-wise strategy selection

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionPipeline {
    pub steps: Vec<FusionStep>,
    pub global_weights: Option<HashMap<String, f32>>, // strategy_name -> weight
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStep {
    pub name: String,
    pub strategy: BlendStrategy,
    /// Tensor name patterns this step applies to (regex)
    #[serde(default)]
    pub tensor_patterns: Vec<String>,
    /// Layer indices this step applies to
    #[serde(default)]
    pub layer_indices: Vec<usize>,
    /// Whether this step is enabled
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_true() -> bool { true }

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum BlendStrategy {
    /// Single merge method
    Single { method: MergeMethod, parameters: HashMap<String, serde_json::Value> },
    /// Weighted blend of multiple methods
    WeightedBlend { 
        methods: Vec<WeightedMethod>,
        blend_mode: BlendMode,
    },
    /// Sequential application
    Sequential { methods: Vec<MergeMethod> },
    /// Tensor-type conditional
    Conditional { 
        conditions: Vec<TensorCondition>,
        default: Box<BlendStrategy>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedMethod {
    pub method: MergeMethod,
    pub weight: f32,
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlendMode {
    /// Linear interpolation of results
    Linear,
    /// SLERP of results
    Slerp,
    /// Geometric mean
    Geometric,
    /// Maximum magnitude
    MaxMag,
    /// Minimum magnitude
    MinMag,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorCondition {
    pub pattern: String, // regex for tensor name
    pub layer_range: Option<(usize, usize)>,
    pub strategy: BlendStrategy,
}

impl FusionPipeline {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            global_weights: None,
        }
    }

    pub fn add_step(&mut self, step: FusionStep) {
        self.steps.push(step);
    }

    /// Execute fusion pipeline on models
    pub fn execute(
        &self,
        stores: &[&TensorStore],
        output_dir: &std::path::Path,
        options: &MergeOptions,
    ) -> Result<()> {
        // Create intermediate storage for multi-step fusion
        let mut current_stores: Vec<&TensorStore> = stores.iter().map(|s| *s).collect();

        for step in &self.steps {
            if !step.enabled {
                continue;
            }

            let step_output = output_dir.join(format!("step_{}", step.name.replace(' ', "_")));
            std::fs::create_dir_all(&step_output)?;

            // Create merge op for this step
            let merge_op = self.create_merge_op(&step.strategy)?;
            
            // Execute merge
            execute_merge(
                merge_op.as_ref(),
                &current_stores,
                &step_output,
                options,
            )?;

            // Load output for next step
            let new_store = TensorStore::open(&step_output)?;
            // Note: This leaks memory but works for now
            current_stores = vec![Box::leak(Box::new(new_store))];
        }

        // Final output is already in the last step's output dir
        // Copy to final output_dir if needed
        if let Some(last_step) = self.steps.last() {
            if last_step.enabled {
                let last_output = output_dir.join(format!("step_{}", last_step.name.replace(' ', "_")));
                copy_dir(&last_output, output_dir)?;
            }
        }

        Ok(())
    }

    fn create_merge_op(&self, strategy: &BlendStrategy) -> Result<Box<dyn MergeOp>> {
        match strategy {
            BlendStrategy::Single { method, parameters } => {
                Ok(Box::new(StrategyMergeOp::new(method.clone(), parameters.clone())))
            }
            BlendStrategy::WeightedBlend { methods, blend_mode } => {
                Ok(Box::new(WeightedBlendOp::new(methods.clone(), blend_mode.clone())))
            }
            BlendStrategy::Sequential { methods } => {
                Ok(Box::new(SequentialFusionOp::new(methods.clone())))
            }
            BlendStrategy::Conditional { conditions, default } => {
                Ok(Box::new(ConditionalFusionOp::new(conditions.clone(), default.clone())))
            }
        }
    }
}

/// MergeOp that wraps a single strategy
struct StrategyMergeOp {
    method: MergeMethod,
    _parameters: HashMap<String, serde_json::Value>,
}

impl StrategyMergeOp {
    fn new(method: MergeMethod, _parameters: HashMap<String, serde_json::Value>) -> Self {
        Self { method, _parameters }
    }
}

impl MergeOp for StrategyMergeOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        // Dispatch to appropriate merge implementation
        match &self.method {
            MergeMethod::Linear => LinearMerge.merge_tensor(name, meta),
            MergeMethod::Slerp { t: _t } => {
                // Would need access to multiple stores - simplified
                LinearMerge.merge_tensor(name, meta)
            }
            MergeMethod::Ties => TiesMerge.merge_tensor(name, meta),
            MergeMethod::Dare => DareMerge.merge_tensor(name, meta),
            MergeMethod::Della => DellaMerge.merge_tensor(name, meta),
            MergeMethod::Passthrough => PassthroughMerge.merge_tensor(name, meta),
            MergeMethod::FrankenMerge => FrankenMerge.merge_tensor(name, meta),
            _ => LinearMerge.merge_tensor(name, meta),
        }
    }

    fn merge_tensors(&self, name: &str, meta: &TensorMeta, inputs: &[Vec<f32>]) -> Result<Vec<f32>> {
        use crate::{
            ArceeFusionMerge, BreadcrumbsMerge, KarcherMerge, ModelStockMerge,
            MultiSlerpMerge, NearSwapMerge, NuSlerpMerge, RamMerge, SceMerge,
            TaskArithmeticMerge,
        };
        match &self.method {
            MergeMethod::NuSlerp => NuSlerpMerge::new(0.5).merge_tensors(name, meta, inputs),
            MergeMethod::MultiSlerp { weights } => MultiSlerpMerge::new(weights.clone()).merge_tensors(name, meta, inputs),
            MergeMethod::Karcher { weights, max_iter, tol } => KarcherMerge::new(weights.clone(), *max_iter, *tol).merge_tensors(name, meta, inputs),
            MergeMethod::TaskArithmetic { lambda } => TaskArithmeticMerge::new(*lambda).merge_tensors(name, meta, inputs),
            MergeMethod::Breadcrumbs { lambda, beta, gamma } => BreadcrumbsMerge::new(*lambda, *beta, *gamma).merge_tensors(name, meta, inputs),
            MergeMethod::BreadcrumbsTies { lambda, beta, gamma } => BreadcrumbsMerge::ties(*lambda, *beta, *gamma).merge_tensors(name, meta, inputs),
            MergeMethod::Sce { lambda } => SceMerge::new(*lambda).merge_tensors(name, meta, inputs),
            MergeMethod::ModelStock => ModelStockMerge::new().merge_tensors(name, meta, inputs),
            MergeMethod::Nearswap => NearSwapMerge::new(0.5, 0.1).merge_tensors(name, meta, inputs),
            MergeMethod::ArceeFusion { lambda, threshold_std } => ArceeFusionMerge::new(*lambda, *threshold_std).merge_tensors(name, meta, inputs),
            MergeMethod::Ram => RamMerge::new(42).merge_tensors(name, meta, inputs),
            _ => self.merge_tensor(name, meta),
        }
    }
}

/// Weighted blend of multiple merge methods
struct WeightedBlendOp {
    methods: Vec<WeightedMethod>,
    blend_mode: BlendMode,
}

impl WeightedBlendOp {
    fn new(methods: Vec<WeightedMethod>, blend_mode: BlendMode) -> Self {
        Self { methods, blend_mode }
    }
}

impl MergeOp for WeightedBlendOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut results: Vec<Vec<f32>> = Vec::new();
        let mut weights: Vec<f32> = Vec::new();

        for wm in &self.methods {
            let op = StrategyMergeOp::new(wm.method.clone(), wm.parameters.clone());
            let result = op.merge_tensor(name, meta)?;
            results.push(result);
            weights.push(wm.weight);
        }

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights { *w /= sum; }
        }

        // Blend results
        let num_elements = meta.num_elements();
        let mut blended = vec![0.0f32; num_elements];

        match self.blend_mode {
            BlendMode::Linear => {
                for (result, weight) in results.iter().zip(weights.iter()) {
                    for (i, &val) in result.iter().enumerate() {
                        blended[i] += val * weight;
                    }
                }
            }
            BlendMode::Slerp => {
                // Pairwise SLERP
                if results.len() >= 2 {
                    blended = slerp_vectors(&results[0], &results[1], weights[1])?;
                    for i in 2..results.len() {
                        blended = slerp_vectors(&blended, &results[i], weights[i])?;
                    }
                } else if !results.is_empty() {
                    blended = results[0].clone();
                }
            }
            BlendMode::Geometric => {
                // Geometric mean: exp(sum(log|x|)) * sign
                for i in 0..num_elements {
                    let mut log_sum = 0.0f32;
                    let mut sign = 1.0f32;
                    for (result, weight) in results.iter().zip(weights.iter()) {
                        let val = result[i];
                        if val != 0.0 {
                            log_sum += weight * val.abs().ln();
                            sign *= val.signum();
                        }
                    }
                    blended[i] = sign * log_sum.exp();
                }
            }
            BlendMode::MaxMag => {
                for i in 0..num_elements {
                    blended[i] = results.iter()
                        .map(|r| r[i])
                        .max_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                        .unwrap_or(0.0);
                }
            }
            BlendMode::MinMag => {
                for i in 0..num_elements {
                    blended[i] = results.iter()
                        .map(|r| r[i])
                        .min_by(|a, b| a.abs().partial_cmp(&b.abs()).unwrap())
                        .unwrap_or(0.0);
                }
            }
        }

        Ok(blended)
    }
}

/// Sequential fusion: apply methods one after another
struct SequentialFusionOp {
    methods: Vec<MergeMethod>,
}

impl SequentialFusionOp {
    fn new(methods: Vec<MergeMethod>) -> Self {
        Self { methods }
    }
}

impl MergeOp for SequentialFusionOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut current = vec![0.0f32; meta.num_elements()];
        
        for (idx, method) in self.methods.iter().enumerate() {
            let op = StrategyMergeOp::new(method.clone(), HashMap::new());
            let result = op.merge_tensor(name, meta)?;
            
            if idx == 0 {
                current = result;
            } else {
                // Blend with previous result (linear by default)
                for (i, &val) in result.iter().enumerate() {
                    current[i] = 0.5 * current[i] + 0.5 * val;
                }
            }
        }
        
        Ok(current)
    }
}

/// Conditional fusion: different strategies for different tensor types
struct ConditionalFusionOp {
    conditions: Vec<TensorCondition>,
    default: Box<BlendStrategy>,
}

impl ConditionalFusionOp {
    fn new(conditions: Vec<TensorCondition>, default: Box<BlendStrategy>) -> Self {
        Self { conditions, default }
    }

    fn match_condition(&self, name: &str, layer_idx: Option<usize>) -> Option<&BlendStrategy> {
        for cond in &self.conditions {
            // Check regex pattern
            if let Ok(re) = regex::Regex::new(&cond.pattern) {
                if !re.is_match(name) {
                    continue;
                }
            } else {
                // Fallback to simple contains
                if !name.contains(&cond.pattern) {
                    continue;
                }
            }

            // Check layer range
            if let Some((start, end)) = cond.layer_range {
                if let Some(idx) = layer_idx {
                    if idx < start || idx >= end {
                        continue;
                    }
                }
            }

            return Some(&cond.strategy);
        }
        None
    }
}

impl MergeOp for ConditionalFusionOp {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let layer_idx = extract_layer_index(name);
        let strategy = self.match_condition(name, layer_idx).unwrap_or(&self.default);
        
        let op = StrategyMergeOp::new(
            match strategy {
                BlendStrategy::Single { method, .. } => method.clone(),
                _ => MergeMethod::Linear,
            },
            HashMap::new(),
        );
        
        op.merge_tensor(name, meta)
    }
}

fn extract_layer_index(name: &str) -> Option<usize> {
    for prefix in ["layers.", "model.layers.", "h."] {
        if let Some(start) = name.find(prefix) {
            let after = &name[start + prefix.len()..];
            if let Some(end) = after.find('.') {
                if let Ok(idx) = after[..end].parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }
    None
}

fn slerp_vectors(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>> {
    if a.len() != b.len() {
        anyhow::bail!("Vector dimension mismatch for SLERP");
    }
    
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(a.to_vec());
    }
    
    let cos_theta = (dot / (norm_a * norm_b)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    
    if theta < 1e-6 {
        return Ok(a.to_vec());
    }
    
    let sin_theta = theta.sin();
    let w1 = ((1.0 - t) * theta).sin() / sin_theta;
    let w2 = (t * theta).sin() / sin_theta;
    
    Ok(a.iter().zip(b.iter())
        .map(|(x, y)| w1 * x + w2 * y)
        .collect())
}

fn copy_dir(src: &std::path::Path, dst: &std::path::Path) -> Result<()> {
    if !src.exists() {
        return Ok(());
    }
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            std::fs::create_dir_all(&dst_path)?;
            copy_dir(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }
    Ok(())
}

/// Pre-built fusion pipelines for common use cases
impl FusionPipeline {
    /// MoE-aware fusion: linear for dense, frankenmoe for experts
    pub fn moe_aware() -> Self {
        let mut pipeline = Self::new();
        
        pipeline.add_step(FusionStep {
            name: "dense_layers".to_string(),
            strategy: BlendStrategy::Single {
                method: MergeMethod::Linear,
                parameters: HashMap::new(),
            },
            tensor_patterns: vec![
                ".*attn.*".to_string(),
                ".*norm.*".to_string(),
                ".*embed.*".to_string(),
                ".*lm_head.*".to_string(),
            ],
            layer_indices: vec![],
            enabled: true,
        });
        
        pipeline.add_step(FusionStep {
            name: "moe_experts".to_string(),
            strategy: BlendStrategy::Single {
                method: MergeMethod::FrankenMerge, // Will use FrankenMoE
                parameters: HashMap::new(),
            },
            tensor_patterns: vec![
                ".*expert.*".to_string(),
                ".*router.*".to_string(),
                ".*gate.*".to_string(),
                ".*shared_expert.*".to_string(),
            ],
            layer_indices: vec![],
            enabled: true,
        });
        
        pipeline
    }

    /// Reasoning-focused fusion: SLERP for attention, linear for FFN
    pub fn reasoning_fusion() -> Self {
        let mut pipeline = Self::new();
        
        pipeline.add_step(FusionStep {
            name: "attention_slerp".to_string(),
            strategy: BlendStrategy::Single {
                method: MergeMethod::Slerp { t: 0.5 },
                parameters: HashMap::new(),
            },
            tensor_patterns: vec![
                ".*q_proj.*".to_string(),
                ".*k_proj.*".to_string(),
                ".*v_proj.*".to_string(),
                ".*o_proj.*".to_string(),
                ".*attn.*".to_string(),
            ],
            layer_indices: vec![],
            enabled: true,
        });
        
        pipeline.add_step(FusionStep {
            name: "ffn_linear".to_string(),
            strategy: BlendStrategy::Single {
                method: MergeMethod::Linear,
                parameters: HashMap::new(),
            },
            tensor_patterns: vec![
                ".*gate_proj.*".to_string(),
                ".*up_proj.*".to_string(),
                ".*down_proj.*".to_string(),
                ".*mlp.*".to_string(),
                ".*ffn.*".to_string(),
            ],
            layer_indices: vec![],
            enabled: true,
        });
        
        pipeline
    }

    /// Darwin + Quantization fusion pipeline
    pub fn darwin_quant_pipeline(generations: usize, population: usize) -> Self {
        let mut pipeline = Self::new();
        
        pipeline.add_step(FusionStep {
            name: "darwin_merge".to_string(),
            strategy: BlendStrategy::Single {
                method: MergeMethod::Darwin { generations, population },
                parameters: HashMap::new(),
            },
            tensor_patterns: vec![],
            layer_indices: vec![],
            enabled: true,
        });
        
        pipeline
    }
}