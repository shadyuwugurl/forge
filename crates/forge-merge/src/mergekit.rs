use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::collections::HashMap;
use forge_core::{MergeConfig, MergeMethod, QuantMethod, ModelEntry, SliceSpec, DType};

/// Full MergeKit YAML configuration support
/// Compatible with mergekit's merge.yml format

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeKitConfig {
    /// Merge method (mergekit compatible)
    pub merge_method: String,
    /// Base model path
    #[serde(default)]
    pub base_model: Option<PathBuf>,
    /// Models to merge
    pub models: Vec<MergeKitModelEntry>,
    /// Slices for passthrough/frankenmerge
    #[serde(default)]
    pub slices: Vec<MergeKitSlice>,
    /// Output dtype
    #[serde(default = "default_dtype")]
    pub dtype: String,
    /// Method-specific parameters
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    /// Output configuration
    #[serde(default)]
    pub output: Option<MergeKitOutput>,
    /// Quantization config
    #[serde(default)]
    pub quantize: Option<MergeKitQuantConfig>,
}

fn default_dtype() -> String { "bfloat16".to_string() }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeKitModelEntry {
    pub model: PathBuf,
    #[serde(default = "default_weight")]
    pub weight: f32,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

fn default_weight() -> f32 { 1.0 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeKitSlice {
    pub model: PathBuf,
    #[serde(rename = "layer_range")]
    pub layer_range: Option<(usize, usize)>,
    #[serde(rename = "layer_indices")]
    pub layer_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeKitOutput {
    pub path: PathBuf,
    #[serde(default = "default_dtype")]
    pub dtype: String,
    #[serde(default)]
    pub shard_size: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeKitQuantConfig {
    pub method: String,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Loader for MergeKit YAML configs
pub struct MergeKitLoader;

impl MergeKitLoader {
    /// Load MergeKit config from YAML file
    pub fn load_from_file(path: &PathBuf) -> Result<MergeKitConfig> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read mergekit config file")?;
        Self::load_from_str(&content)
    }

    /// Load MergeKit config from YAML string
    pub fn load_from_str(content: &str) -> Result<MergeKitConfig> {
        let config: MergeKitConfig = serde_yaml::from_str(content)
            .context("Failed to parse mergekit YAML")?;
        Ok(config)
    }

    /// Convert MergeKit config to Forge MergeConfig
    pub fn to_forge_config(&self, config: &MergeKitConfig) -> Result<MergeConfig> {
        let merge_method = Self::parse_merge_method(&config.merge_method, &config.parameters)?;
        
        let models = config.models.iter().map(|m| ModelEntry {
            path: m.model.clone(),
            weight: m.weight,
            density: 1.0,
            epsilon: 0.0,
        }).collect();

        let slices = config.slices.iter().map(|s| SliceSpec {
            model: s.model.clone(),
            layer_range: s.layer_range.unwrap_or((0, usize::MAX)),
        }).collect();

        let dtype = Self::parse_dtype(&config.dtype);
        let output = config.output.as_ref().map(|o| forge_core::OutputConfig {
            path: o.path.clone(),
            dtype: Self::parse_dtype(&o.dtype),
            shard_size: o.shard_size,
        });

        let quant = config.quantize.as_ref().map(|q| Self::parse_quant_method(&q.method, &q.parameters))
            .transpose()?;

        Ok(MergeConfig {
            merge_method,
            base_model: config.base_model.clone(),
            models,
            slices,
            dtype,
            parameters: config.parameters.clone(),
            output,
            quant,
            darwin: None,
        })
    }

    fn parse_merge_method(method: &str, params: &HashMap<String, serde_json::Value>) -> Result<MergeMethod> {
        match method.to_lowercase().as_str() {
            "linear" => Ok(MergeMethod::Linear),
            "slerp" => {
                let t = params.get("t").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                Ok(MergeMethod::Slerp { t })
            }
            "nuslerp" => Ok(MergeMethod::NuSlerp),
            "multislerp" | "multi_slerp" | "multi-slerp" => {
                let weights = params.get("weights")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                    .unwrap_or_default();
                Ok(MergeMethod::MultiSlerp { weights })
            }
            "karcher" | "karcher_mean" | "karcher-mean" => {
                let weights = params.get("weights")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|f| f as f32)).collect())
                    .unwrap_or_default();
                let max_iter = params.get("max_iter").and_then(|v| v.as_u64()).unwrap_or(20) as usize;
                let tol = params.get("tol").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
                Ok(MergeMethod::Karcher { weights, max_iter, tol })
            }
            "task_arithmetic" | "task-arithmetic" => {
                let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                Ok(MergeMethod::TaskArithmetic { lambda })
            }
            "ties" => Ok(MergeMethod::Ties),
            "dare" => Ok(MergeMethod::Dare),
            "dare_ties" | "dare-ties" => Ok(MergeMethod::DareTies),
            "della_linear" | "della-linear" => Ok(MergeMethod::DellaLinear),
            "della" => Ok(MergeMethod::Della),
            "passthrough" => Ok(MergeMethod::Passthrough),
            "darwin" => {
                let generations = params.get("generations").and_then(|v| v.as_u64()).unwrap_or(30) as usize;
                let population = params.get("population").and_then(|v| v.as_u64()).unwrap_or(40) as usize;
                Ok(MergeMethod::Darwin { generations, population })
            }
            "frankenmerge" => Ok(MergeMethod::FrankenMerge),
            "model_stock" | "model-stock" => Ok(MergeMethod::ModelStock),
            "breadcrumbs" => {
                let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let beta = params.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
                let gamma = params.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
                Ok(MergeMethod::Breadcrumbs { lambda, beta, gamma })
            }
            "breadcrumbs_ties" | "breadcrumbs-ties" => {
                let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let beta = params.get("beta").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
                let gamma = params.get("gamma").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
                Ok(MergeMethod::BreadcrumbsTies { lambda, beta, gamma })
            }
            "sce" => {
                let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                Ok(MergeMethod::Sce { lambda })
            }
            "arcee_fusion" | "arcee-fusion" => {
                let lambda = params.get("lambda").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                let threshold_std = params.get("threshold_std").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
                Ok(MergeMethod::ArceeFusion { lambda, threshold_std })
            }
            "nearswap" => {
                let t = params.get("t").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                let threshold = params.get("threshold").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32;
                Ok(MergeMethod::Nearswap { t, threshold })
            }
            "ram" => {
                let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(42);
                Ok(MergeMethod::Ram { seed })
            }
            "frankenmoe" => {
                let bottom_layers = params.get("bottom_layers").and_then(|v| v.as_u64()).unwrap_or(4) as usize;
                let middle_experts = params.get("middle_experts").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
                let top_layers = params.get("top_layers").and_then(|v| v.as_u64()).unwrap_or(2) as usize;
                Ok(MergeMethod::FrankenMoE { bottom_layers, middle_experts, top_layers })
            }
            _ => anyhow::bail!("Unknown merge method: {}", method),
        }
    }

    fn parse_dtype(dtype: &str) -> DType {
        match dtype.to_lowercase().as_str() {
            "float32" | "fp32" | "f32" => DType::F32,
            "float16" | "fp16" | "f16" | "half" => DType::F16,
            "bfloat16" | "bf16" => DType::BF16,
            _ => DType::BF16,
        }
    }

    fn parse_quant_method(method: &str, params: &HashMap<String, serde_json::Value>) -> Result<QuantMethod> {
        match method.to_lowercase().as_str() {
            "jang" => {
                let profile = params.get("profile")
                    .and_then(|v| v.as_str())
                    .unwrap_or("JANG_2L")
                    .to_string();
                let output_format = params.get("output_format")
                    .and_then(|v| v.as_str())
                    .unwrap_or("mlx");
                let format = match output_format {
                    "gguf" => forge_core::JangOutputFormat::Gguf,
                    _ => forge_core::JangOutputFormat::Mlx,
                };
                Ok(QuantMethod::Jang { profile, output_format: format })
            }
            "dynamic3" => {
                let density = params.get("density").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
                Ok(QuantMethod::Dynamic3 { density, model_specific: true })
            }
            "apex" => {
                let tier = params.get("tier").and_then(|v| v.as_str()).unwrap_or("balanced").to_string();
                Ok(QuantMethod::Apex { tier })
            }
            "btl4" => {
                let target_bpw = params.get("target_bpw").and_then(|v| v.as_f64()).unwrap_or(4.0) as f32;
                Ok(QuantMethod::Btl4Compact { target_bpw })
            }
            "mixed" => {
                let per_layer_bits = params.get("per_layer_bits")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter()
                        .filter_map(|v| {
                            let obj = v.as_object()?;
                            let name = obj.get("layer")?.as_str()?.to_string();
                            let bits = obj.get("bits")?.as_u64()? as u8;
                            Some((name, bits))
                        })
                        .collect())
                    .unwrap_or_default();
                Ok(QuantMethod::MixedPrecision { per_layer_bits })
            }
            _ => anyhow::bail!("Unknown quantization method: {}", method),
        }
    }
}

/// Generate MergeKit-compatible YAML from Forge config
pub fn to_mergekit_yaml(config: &MergeConfig) -> Result<String> {
    let mk_config = MergeKitConfig {
        merge_method: merge_method_to_string(&config.merge_method),
        base_model: config.base_model.clone(),
        models: config.models.iter().map(|m| MergeKitModelEntry {
            model: m.path.clone(),
            weight: m.weight,
            parameters: HashMap::new(),
        }).collect(),
        slices: config.slices.iter().map(|s| MergeKitSlice {
            model: s.model.clone(),
            layer_range: Some(s.layer_range),
            layer_indices: None,
            parameters: HashMap::new(),
        }).collect(),
        dtype: dtype_to_string(config.dtype),
        parameters: config.parameters.clone(),
        output: config.output.as_ref().map(|o| MergeKitOutput {
            path: o.path.clone(),
            dtype: dtype_to_string(o.dtype),
            shard_size: o.shard_size,
        }),
        quantize: config.quant.as_ref().map(|q| quant_method_to_config(q)),
    };
    
    serde_yaml::to_string(&mk_config).context("Failed to serialize to YAML")
}

fn merge_method_to_string(method: &MergeMethod) -> String {
    match method {
        MergeMethod::Linear => "linear".to_string(),
        MergeMethod::Slerp { .. } => "slerp".to_string(),
        MergeMethod::NuSlerp => "nuslerp".to_string(),
        MergeMethod::MultiSlerp { .. } => "multislerp".to_string(),
        MergeMethod::Karcher { .. } => "karcher".to_string(),
        MergeMethod::TaskArithmetic { .. } => "task_arithmetic".to_string(),
        MergeMethod::Ties => "ties".to_string(),
        MergeMethod::Dare => "dare".to_string(),
        MergeMethod::DareTies => "dare_ties".to_string(),
        MergeMethod::DellaLinear => "della_linear".to_string(),
        MergeMethod::Della => "della".to_string(),
        MergeMethod::Passthrough => "passthrough".to_string(),
        MergeMethod::Darwin { .. } => "darwin".to_string(),
        MergeMethod::FrankenMerge => "frankenmerge".to_string(),
        MergeMethod::FrankenMoE { .. } => "frankenmoe".to_string(),
        MergeMethod::Fusion { .. } => "fusion".to_string(),
        MergeMethod::ModelStock => "model_stock".to_string(),
        MergeMethod::Breadcrumbs { .. } => "breadcrumbs".to_string(),
        MergeMethod::BreadcrumbsTies { .. } => "breadcrumbs_ties".to_string(),
        MergeMethod::Sce { .. } => "sce".to_string(),
        MergeMethod::ArceeFusion { .. } => "arcee_fusion".to_string(),
        MergeMethod::Nearswap { .. } => "nearswap".to_string(),
        MergeMethod::Ram { .. } => "ram".to_string(),
        MergeMethod::Latent { .. } => "latent".to_string(),
        MergeMethod::Orca { .. } => "orca".to_string(),
        MergeMethod::ExpertWeaver { .. } => "expert_weaver".to_string(),
        MergeMethod::MoeDenseDistill { .. } => "moe_dense_distill".to_string(),
        MergeMethod::Hetero { .. } => "hetero".to_string(),
        MergeMethod::Chimera { .. } => "chimera".to_string(),
        MergeMethod::Aether { .. } => "aether".to_string(),
        MergeMethod::Pocket { .. } => "pocket".to_string(),
    }
}

fn dtype_to_string(dtype: DType) -> String {
    match dtype {
        DType::F32 => "float32".to_string(),
        DType::F16 => "float16".to_string(),
        DType::BF16 => "bfloat16".to_string(),
        _ => "bfloat16".to_string(),
    }
}

fn quant_method_to_config(method: &QuantMethod) -> MergeKitQuantConfig {
    match method {
        QuantMethod::Jang { profile, output_format } => MergeKitQuantConfig {
            method: "jang".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("profile".to_string(), serde_json::json!(profile));
                m.insert("output_format".to_string(), serde_json::json!(
                    match output_format {
                        forge_core::JangOutputFormat::Mlx => "mlx",
                        forge_core::JangOutputFormat::Gguf => "gguf",
                    }
                ));
                m
            },
        },
        QuantMethod::Dynamic3 { density, .. } => MergeKitQuantConfig {
            method: "dynamic3".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("density".to_string(), serde_json::json!(density));
                m
            },
        },
        QuantMethod::Apex { tier } => MergeKitQuantConfig {
            method: "apex".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("tier".to_string(), serde_json::json!(tier));
                m
            },
        },
        QuantMethod::Btl4Compact { target_bpw } => MergeKitQuantConfig {
            method: "btl4".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("target_bpw".to_string(), serde_json::json!(target_bpw));
                m
            },
        },
        QuantMethod::MixedPrecision { per_layer_bits } => MergeKitQuantConfig {
            method: "mixed".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("per_layer_bits".to_string(), serde_json::json!(per_layer_bits));
                m
            },
        },
        QuantMethod::Bsqat { bits, block } => MergeKitQuantConfig {
            method: "bsqat".to_string(),
            parameters: {
                let mut m = HashMap::new();
                m.insert("bits".to_string(), serde_json::json!(bits));
                m.insert("block".to_string(), serde_json::json!(block));
                m
            },
        },
        other => MergeKitQuantConfig {
            method: format!("{:?}", other).to_lowercase(),
            parameters: HashMap::new(),
        },
    }
}