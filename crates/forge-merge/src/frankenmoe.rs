use anyhow::{Result, Context};
use forge_core::TensorMeta;
use crate::orchestrator::MergeOp;
use std::collections::HashMap;

/// FrankenMoE: MoE-aware layer stacking with expert-level granularity
/// 
/// Supports:
/// - Per-expert merging across MoE models
/// - Router/gate weight merging strategies
/// - Shared expert handling
/// - Cross-architecture MoE (different num_experts, experts_per_token)
/// - Layer-wise expert selection (like FrankenMerge but for MoE)

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FrankenMoEConfig {
    /// How to merge router/gate weights
    #[serde(default = "default_router_strategy")]
    pub router_strategy: RouterMergeStrategy,
    /// How to merge expert weights
    #[serde(default = "default_expert_strategy")]
    pub expert_strategy: ExpertMergeStrategy,
    /// How to handle shared experts
    #[serde(default = "default_shared_strategy")]
    pub shared_expert_strategy: SharedExpertStrategy,
    /// Per-layer expert selection (layer_idx -> expert_indices)
    #[serde(default)]
    pub layer_expert_map: HashMap<usize, Vec<usize>>,
    /// Whether to merge experts across different num_experts configs
    #[serde(default = "default_true")]
    pub adapt_expert_count: bool,
    /// Density for expert pruning (0.0 - 1.0)
    #[serde(default)]
    pub expert_density: Option<f32>,
}

fn default_router_strategy() -> RouterMergeStrategy { RouterMergeStrategy::Linear { weight: None } }
fn default_expert_strategy() -> ExpertMergeStrategy { ExpertMergeStrategy::Linear { weight: None } }
fn default_shared_strategy() -> SharedExpertStrategy { SharedExpertStrategy::Concatenate }
fn default_true() -> bool { true }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum RouterMergeStrategy {
    /// Weighted average of router logits
    Linear { weight: Option<f32> },
    /// SLERP for router weights
    Slerp { t: f32 },
    /// Task arithmetic on router
    TaskArithmetic { lambda: f32 },
    /// DARE on router (drop and rescale)
    Dare { density: f32 },
    /// TIES on router
    Ties { density: f32 },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum ExpertMergeStrategy {
    /// Weighted average per expert
    Linear { weight: Option<f32> },
    /// SLERP per expert
    Slerp { t: f32 },
    /// FrankenMerge-style: select experts from different models per layer
    FrankenSelect { source_model_per_layer: HashMap<usize, usize> },
    /// Concatenate experts (increase num_experts)
    Concatenate,
    /// Prune to top-k experts by norm
    TopK { k: usize },
    /// Merge experts with dimension adaptation
    DimensionAdapt { target_experts: usize },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum SharedExpertStrategy {
    /// Concatenate shared experts
    Concatenate,
    /// Average shared experts
    Average,
    /// Keep only first model's shared expert
    FirstModel,
    /// Merge with dimension adaptation
    DimensionAdapt { target_dim: usize },
}

pub struct FrankenMoE<'a> {
    pub stores: &'a [&'a dyn TensorStoreLike],
    pub config: FrankenMoEConfig,
    pub model_archs: Vec<MoEArchitecture>,
}

#[derive(Debug, Clone)]
pub struct MoEArchitecture {
    pub num_experts: usize,
    pub experts_per_token: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub has_shared_expert: bool,
    pub shared_expert_intermediate_size: Option<usize>,
    pub router_type: RouterType,
    pub layer_indices: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterType {
    Standard,
    SwitchTransformer,
    DeepSeekV2,
    QwenMoE,
    Mixtral,
    Custom,
}

pub trait TensorStoreLike {
    fn tensor_names(&self) -> Vec<String>;
    fn tensor_meta(&self, name: &str) -> Result<TensorMeta>;
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>>;
    fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>>;
    fn total_params(&self) -> usize;
}

impl<'a> FrankenMoE<'a> {
    pub fn new(stores: &'a [&'a dyn TensorStoreLike], config: FrankenMoEConfig) -> Result<Self> {
        let model_archs = Self::detect_moe_architectures(stores)?;
        Ok(Self { stores, config, model_archs })
    }

    fn detect_moe_architectures(stores: &[&dyn TensorStoreLike]) -> Result<Vec<MoEArchitecture>> {
        let mut archs = Vec::new();
        
        for store in stores {
            let names = store.tensor_names();
            
            // Detect MoE pattern
            let expert_names: Vec<_> = names.iter()
                .filter(|n| n.contains("expert") || n.contains("switch_mlp") || n.contains("mlp.experts"))
                .collect();
            
            if expert_names.is_empty() {
                // Not an MoE model
                archs.push(MoEArchitecture {
                    num_experts: 0,
                    experts_per_token: 0,
                    hidden_size: 0,
                    intermediate_size: 0,
                    has_shared_expert: false,
                    shared_expert_intermediate_size: None,
                    router_type: RouterType::Custom,
                    layer_indices: vec![],
                });
                continue;
            }

            // Infer architecture from tensor names
            let mut num_experts = 0;
            let mut layer_indices = std::collections::HashSet::new();
            let mut has_shared_expert = false;
            let mut hidden_size = 0;
            let mut intermediate_size = 0;

            for name in expert_names {
                if let Some(meta) = store.tensor_meta(name).ok() {
                    if meta.shape.len() >= 3 {
                        // Shape: [num_experts, hidden_size, intermediate_size] or similar
                        num_experts = num_experts.max(meta.shape[0]);
                        hidden_size = hidden_size.max(meta.shape[1]);
                        intermediate_size = intermediate_size.max(meta.shape[2]);
                    }
                    
                    // Extract layer index
                    if let Some(idx) = Self::extract_layer_index(name) {
                        layer_indices.insert(idx);
                    }
                    
                    if name.contains("shared_expert") {
                        has_shared_expert = true;
                    }
                }
            }

            // Detect router type
            let router_type = Self::detect_router_type(&names);
            
            archs.push(MoEArchitecture {
                num_experts,
                experts_per_token: 2, // Default, would need config
                hidden_size,
                intermediate_size,
                has_shared_expert,
                shared_expert_intermediate_size: None,
                router_type,
                layer_indices: layer_indices.into_iter().collect(),
            });
        }

        Ok(archs)
    }

    fn extract_layer_index(name: &str) -> Option<usize> {
        // Try patterns: "layers.5.", "model.layers.5.", "h.5."
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

    fn detect_router_type(names: &[String]) -> RouterType {
        if names.iter().any(|n| n.contains("switch_mlp")) {
            RouterType::SwitchTransformer
        } else if names.iter().any(|n| n.contains("gate.weight") && n.contains("expert")) {
            RouterType::DeepSeekV2
        } else if names.iter().any(|n| n.contains("router")) {
            RouterType::Mixtral
        } else if names.iter().any(|n| n.contains("mlp.gate")) {
            RouterType::QwenMoE
        } else {
            RouterType::Standard
        }
    }

    /// Merge router weights from multiple models
    fn merge_router(&self, name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut router_data: Vec<Vec<f32>> = Vec::new();
        
        for store in self.stores {
            if let Ok(data) = store.tensor_f32(name) {
                router_data.push(data);
            }
        }

        if router_data.is_empty() {
            anyhow::bail!("No router data found for {}", name);
        }

        match &self.config.router_strategy {
            RouterMergeStrategy::Linear { weight } => {
                let w = weight.unwrap_or(1.0 / router_data.len() as f32);
                let mut result = vec![0.0f32; router_data[0].len()];
                for data in &router_data {
                    for (i, &val) in data.iter().enumerate() {
                        result[i] += val * w;
                    }
                }
                Ok(result)
            }
            RouterMergeStrategy::Slerp { t } => {
                // SLERP for 2 models, sequential for more
                let mut result = router_data[0].clone();
                for data in &router_data[1..] {
                    result = slerp_vectors(&result, data, *t)?;
                }
                Ok(result)
            }
            RouterMergeStrategy::TaskArithmetic { lambda } => {
                // base + lambda * (finetuned - base)
                let base = &router_data[0];
                let mut result = base.clone();
                for data in &router_data[1..] {
                    for (i, &val) in data.iter().enumerate() {
                        result[i] += *lambda * (val - base[i]);
                    }
                }
                Ok(result)
            }
            RouterMergeStrategy::Dare { density } => {
                let base = &router_data[0];
                let mut result = base.clone();
                use rand::Rng;
                let mut rng = rand::thread_rng();
                for data in &router_data[1..] {
                    for (i, &val) in data.iter().enumerate() {
                        if rng.gen::<f32>() < *density {
                            result[i] += val / *density;
                        }
                    }
                }
                Ok(result)
            }
            RouterMergeStrategy::Ties { density } => {
                // Trim, Elect Sign, Disjoint Merge
                let base = &router_data[0];
                let mut result = vec![0.0f32; base.len()];
                let mut counts = vec![0usize; base.len()];
                
                for data in &router_data[1..] {
                    // Get magnitude threshold
                    let mut mags: Vec<f32> = data.iter().map(|v| v.abs()).collect();
                    mags.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let threshold_idx = ((1.0 - density) * mags.len() as f32) as usize;
                    let threshold = mags.get(threshold_idx).copied().unwrap_or(0.0);
                    
                    for (i, &val) in data.iter().enumerate() {
                        if val.abs() >= threshold {
                            // Elect sign: only add if same sign as base
                            if (val > 0.0) == (base[i] > 0.0) || base[i] == 0.0 {
                                result[i] += val;
                                counts[i] += 1;
                            }
                        }
                    }
                }
                
                // Rescale
                for (i, count) in counts.iter().enumerate() {
                    if *count > 0 {
                        result[i] /= *count as f32;
                    }
                }
                Ok(result)
            }
        }
    }

    /// Merge expert weights with MoE-aware strategy
    fn merge_experts(&self, name: &str, _meta: &TensorMeta, layer_idx: usize) -> Result<Vec<f32>> {
        // Collect expert tensors from all models
        let mut expert_data: Vec<Vec<f32>> = Vec::new();
        let mut expert_shapes: Vec<Vec<usize>> = Vec::new();
        
        for store in self.stores {
            if let Ok(data) = store.tensor_f32(name) {
                if let Ok(meta) = store.tensor_meta(name) {
                    expert_data.push(data);
                    expert_shapes.push(meta.shape.clone());
                }
            }
        }

        if expert_data.is_empty() {
            anyhow::bail!("No expert data found for {}", name);
        }

        match &self.config.expert_strategy {
            ExpertMergeStrategy::Linear { weight } => {
                let w = weight.unwrap_or(1.0 / expert_data.len() as f32);
                let mut result = vec![0.0f32; expert_data[0].len()];
                for data in &expert_data {
                    for (i, &val) in data.iter().enumerate() {
                        result[i] += val * w;
                    }
                }
                Ok(result)
            }
            ExpertMergeStrategy::Slerp { t } => {
                let mut result = expert_data[0].clone();
                for data in &expert_data[1..] {
                    result = slerp_vectors(&result, data, *t)?;
                }
                Ok(result)
            }
            ExpertMergeStrategy::FrankenSelect { source_model_per_layer } => {
                // Select experts from specific model per layer
                if let Some(&model_idx) = source_model_per_layer.get(&layer_idx) {
                    if model_idx < expert_data.len() {
                        Ok(expert_data[model_idx].clone())
                    } else {
                        Ok(expert_data[0].clone())
                    }
                } else {
                    Ok(expert_data[0].clone())
                }
            }
            ExpertMergeStrategy::Concatenate => {
                // Concatenate all experts along expert dimension
                let mut result = Vec::new();
                for data in &expert_data {
                    result.extend_from_slice(data);
                }
                Ok(result)
            }
            ExpertMergeStrategy::TopK { k } => {
                // Keep top-k experts by L2 norm
                if expert_data.len() == 1 {
                    return Ok(expert_data[0].clone());
                }
                
                // Assume first dim is num_experts
                let _experts_per_model: Vec<usize> = expert_shapes.iter()
                    .map(|s| s.first().copied().unwrap_or(1))
                    .collect();
                
                let mut all_experts: Vec<(usize, usize, f32, Vec<f32>)> = Vec::new(); // (model_idx, expert_idx, norm, data)
                
                for (model_idx, (data, shape)) in expert_data.iter().zip(expert_shapes.iter()).enumerate() {
                    let num_experts = shape.first().copied().unwrap_or(1);
                    let expert_size = data.len() / num_experts.max(1);
                    
                    for e in 0..num_experts {
                        let start = e * expert_size;
                        let end = start + expert_size;
                        let expert_slice = &data[start..end];
                        let norm = expert_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
                        all_experts.push((model_idx, e, norm, expert_slice.to_vec()));
                    }
                }
                
                // Sort by norm descending
                all_experts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
                
                // Take top-k
                let selected: Vec<f32> = all_experts.into_iter()
                    .take(*k)
                    .flat_map(|(_, _, _, data)| data)
                    .collect();
                
                Ok(selected)
            }
            ExpertMergeStrategy::DimensionAdapt { target_experts } => {
                // Adapt expert count to target
                let current_total: usize = expert_shapes.iter()
                    .map(|s| s.first().copied().unwrap_or(1))
                    .sum();
                
                if current_total == *target_experts {
                    // Just concatenate
                    let mut result = Vec::new();
                    for data in &expert_data {
                        result.extend_from_slice(data);
                    }
                    return Ok(result);
                }
                
                if current_total < *target_experts {
                    // Need to interpolate/expand
                    return self.expand_experts(&expert_data, &expert_shapes, *target_experts);
                } else {
                    // Need to merge/prune
                    return self.prune_experts(&expert_data, &expert_shapes, *target_experts);
                }
            }
        }
    }

    fn expand_experts(&self, expert_data: &[Vec<f32>], expert_shapes: &[Vec<usize>], target: usize) -> Result<Vec<f32>> {
        let current: usize = expert_shapes.iter().map(|s| s.first().copied().unwrap_or(1)).sum();
        let expert_size = expert_data[0].len() / current.max(1);
        let needed = target - current;
        
        let mut result = Vec::new();
        for data in expert_data {
            result.extend_from_slice(data);
        }
        
        // Interpolate new experts from existing ones
        for i in 0..needed {
            let src_idx = i % current;
            let start = src_idx * expert_size;
            let end = start + expert_size;
            let mut new_expert = expert_data[0][start..end].to_vec();
            
            // Add small noise
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for val in &mut new_expert {
                *val += rng.gen::<f32>() * 0.01 - 0.005;
            }
            result.extend_from_slice(&new_expert);
        }
        
        Ok(result)
    }

    fn prune_experts(&self, expert_data: &[Vec<f32>], expert_shapes: &[Vec<usize>], target: usize) -> Result<Vec<f32>> {
        // Use TopK logic
        let k = target;
        let mut all_experts: Vec<(usize, usize, f32, Vec<f32>)> = Vec::new();
        
        for (model_idx, (data, shape)) in expert_data.iter().zip(expert_shapes.iter()).enumerate() {
            let num_experts = shape.first().copied().unwrap_or(1);
            let expert_size = data.len() / num_experts.max(1);
            
            for e in 0..num_experts {
                let start = e * expert_size;
                let end = start + expert_size;
                let expert_slice = &data[start..end];
                let norm = expert_slice.iter().map(|v| v * v).sum::<f32>().sqrt();
                all_experts.push((model_idx, e, norm, expert_slice.to_vec()));
            }
        }
        
        all_experts.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
        
        let selected: Vec<f32> = all_experts.into_iter()
            .take(k)
            .flat_map(|(_, _, _, data)| data)
            .collect();
        
        Ok(selected)
    }

    /// Merge shared expert weights
    fn merge_shared_expert(&self, name: &str, _meta: &TensorMeta) -> Result<Vec<f32>> {
        let mut shared_data: Vec<Vec<f32>> = Vec::new();
        
        for store in self.stores {
            if let Ok(data) = store.tensor_f32(name) {
                shared_data.push(data);
            }
        }

        if shared_data.is_empty() {
            anyhow::bail!("No shared expert data for {}", name);
        }

        match &self.config.shared_expert_strategy {
            SharedExpertStrategy::Concatenate => {
                let mut result = Vec::new();
                for data in &shared_data {
                    result.extend_from_slice(data);
                }
                Ok(result)
            }
            SharedExpertStrategy::Average => {
                let mut result = vec![0.0f32; shared_data[0].len()];
                for data in &shared_data {
                    for (i, &val) in data.iter().enumerate() {
                        result[i] += val;
                    }
                }
                for val in &mut result {
                    *val /= shared_data.len() as f32;
                }
                Ok(result)
            }
            SharedExpertStrategy::FirstModel => {
                Ok(shared_data[0].clone())
            }
            SharedExpertStrategy::DimensionAdapt { target_dim } => {
                // Interpolate to target dimension
                let current_dim = shared_data[0].len();
                if current_dim == *target_dim {
                    return Ok(shared_data[0].clone());
                }
                
                let scale = *target_dim as f32 / current_dim as f32;
                let mut result = vec![0.0f32; *target_dim];
                for (i, &val) in shared_data[0].iter().enumerate() {
                    let target_idx = (i as f32 * scale) as usize;
                    if target_idx < *target_dim {
                        result[target_idx] += val;
                    }
                }
                Ok(result)
            }
        }
    }

    pub fn is_moe_tensor(&self, name: &str) -> bool {
        let n = name.to_lowercase();
        n.contains("expert") || n.contains("switch_mlp") || n.contains("mlp.experts") || 
        n.contains("shared_expert") || n.contains("router") || n.contains("gate") && n.contains("expert")
    }

    pub fn is_router_tensor(&self, name: &str) -> bool {
        let n = name.to_lowercase();
        n.contains("router") || (n.contains("gate") && n.contains("weight") && n.contains("expert"))
    }

    pub fn is_shared_expert_tensor(&self, name: &str) -> bool {
        name.to_lowercase().contains("shared_expert")
    }
}

impl<'a> MergeOp for FrankenMoE<'a> {
    fn merge_tensor(&self, name: &str, meta: &TensorMeta) -> Result<Vec<f32>> {
        let layer_idx = FrankenMoE::extract_layer_index(name).unwrap_or(0);

        if self.is_router_tensor(name) {
            return self.merge_router(name, meta);
        }

        if self.is_shared_expert_tensor(name) {
            return self.merge_shared_expert(name, meta);
        }

        if self.is_moe_tensor(name) {
            return self.merge_experts(name, meta, layer_idx);
        }

        // Non-MoE tensors: use linear merge as fallback
        let mut result = vec![0.0f32; meta.num_elements()];
        for store in self.stores {
            if let Ok(data) = store.tensor_f32(name) {
                for (i, &val) in data.iter().enumerate() {
                    result[i] += val / self.stores.len() as f32;
                }
            }
        }
        Ok(result)
    }
}

/// SLERP for vectors
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

impl TensorStoreLike for forge_io::TensorStore {
    fn tensor_names(&self) -> Vec<String> {
        self.tensor_names()
    }
    fn tensor_meta(&self, name: &str) -> Result<TensorMeta> {
        self.tensor_meta(name).context("tensor_meta failed")
    }
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        self.tensor_f32(name).context("tensor_f32 failed")
    }
    fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        self.tensor_bytes(name).context("tensor_bytes failed")
    }
    fn total_params(&self) -> usize {
        self.total_params()
    }
}

impl<'a> TensorStoreLike for &'a forge_io::TensorStore {
    fn tensor_names(&self) -> Vec<String> {
        (*self).tensor_names()
    }
    fn tensor_meta(&self, name: &str) -> Result<TensorMeta> {
        (*self).tensor_meta(name).context("tensor_meta failed")
    }
    fn tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        (*self).tensor_f32(name).context("tensor_f32 failed")
    }
    fn tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        (*self).tensor_bytes(name).context("tensor_bytes failed")
    }
    fn total_params(&self) -> usize {
        (*self).total_params()
    }
}