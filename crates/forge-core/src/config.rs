use std::path::PathBuf;
use crate::model::{SliceSpec, ModelEntry};

/// Merge method selection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum MergeMethod {
    /// Simple weighted averaging
    #[serde(rename = "linear")]
    Linear,
    /// Spherical linear interpolation (2 models)
    #[serde(rename = "slerp")]
    Slerp { t: f32 },
    /// Multi-model SLERP via sequential pairwise
    #[serde(rename = "nuslerp")]
    NuSlerp,
    /// Barycentric spherical interpolation over N models
    #[serde(rename = "multislerp")]
    MultiSlerp {
        #[serde(default)]
        weights: Vec<f32>,
    },
    /// Riemannian (Karcher/Fréchet) mean on the sphere
    #[serde(rename = "karcher")]
    Karcher {
        #[serde(default)]
        weights: Vec<f32>,
        #[serde(default = "default_karcher_iter")]
        max_iter: usize,
        #[serde(default = "default_karcher_tol")]
        tol: f32,
    },
    /// Task vector arithmetic
    #[serde(rename = "task_arithmetic")]
    TaskArithmetic { lambda: f32 },
    /// Trim, elect sign, disjoint merge
    #[serde(rename = "ties")]
    Ties,
    /// Random dropout + rescaled merge
    #[serde(rename = "dare")]
    Dare,
    /// DARE + TIES
    #[serde(rename = "dare_ties")]
    DareTies,
    /// Magnitude-aware dropout + linear
    #[serde(rename = "della_linear")]
    DellaLinear,
    /// Magnitude-aware dropout + TIES
    #[serde(rename = "della")]
    Della,
    /// Layer passthrough / frankenmerge
    #[serde(rename = "passthrough")]
    Passthrough,
    /// Darwin V6 evolutionary merge
    #[serde(rename = "darwin")]
    Darwin {
        generations: usize,
        population: usize,
    },
    /// FrankenMerge with dimension adaptation
    #[serde(rename = "frankenmerge")]
    FrankenMerge,
    /// Model stock geometric interpolation
    #[serde(rename = "model_stock")]
    ModelStock,
    /// Breadcrumbs merge (outlier-filtered task arithmetic)
    #[serde(rename = "breadcrumbs")]
    Breadcrumbs {
        #[serde(default = "default_lambda_one")]
        lambda: f32,
        #[serde(default = "default_breadcrumb_beta")]
        beta: f32,
        #[serde(default = "default_breadcrumb_gamma")]
        gamma: f32,
    },
    /// Breadcrumbs + TIES sign consensus
    #[serde(rename = "breadcrumbs_ties")]
    BreadcrumbsTies {
        #[serde(default = "default_lambda_one")]
        lambda: f32,
        #[serde(default = "default_breadcrumb_beta")]
        beta: f32,
        #[serde(default = "default_breadcrumb_gamma")]
        gamma: f32,
    },
    /// SCE variance-weighted task arithmetic
    #[serde(rename = "sce")]
    Sce {
        #[serde(default = "default_lambda_one")]
        lambda: f32,
    },
    /// Arcee fusion dynamic-threshold fusion
    #[serde(rename = "arcee_fusion")]
    ArceeFusion {
        #[serde(default = "default_lambda_one")]
        lambda: f32,
        #[serde(default = "default_arcee_thresh")]
        threshold_std: f32,
    },
    /// Nearswap merge
    #[serde(rename = "nearswap")]
    Nearswap {
        #[serde(default = "default_nearswap_t")]
        t: f32,
        #[serde(default = "default_nearswap_thresh")]
        threshold: f32,
    },
    /// RAM merge
    #[serde(rename = "ram")]
    Ram {
        #[serde(default = "default_ram_seed")]
        seed: u64,
    },
    /// FrankenMoE burger-style MoE merge
    #[serde(rename = "frankenmoe")]
    FrankenMoE {
        #[serde(default = "default_frankenmoe_bottom")]
        bottom_layers: usize,
        #[serde(default = "default_frankenmoe_experts")]
        middle_experts: usize,
        #[serde(default = "default_frankenmoe_top")]
        top_layers: usize,
    },
    /// Generic fusion pipeline (multi-strategy)
    #[serde(rename = "fusion")]
    Fusion {
        #[serde(default)]
        steps: usize,
    },
    /// ORCA outlier-aware allocation (AAAI-26)
    #[serde(rename = "orca")]
    Orca {
        stats_path: Option<PathBuf>,
        #[serde(default = "default_orca_threshold")]
        threshold: f32,
    },
    /// LS-Merge latent-space merge (ICLR-26)
    #[serde(rename = "latent")]
    Latent {
        vae_path: Option<PathBuf>,
        #[serde(default = "default_latent_dim")]
        latent_dim: usize,
    },
    /// ExpertWeaver GLU sparsification merge (ICML-26)
    #[serde(rename = "expert_weaver")]
    ExpertWeaver {
        #[serde(default = "default_num_experts")]
        num_experts: usize,
    },
    /// MoE-to-dense distillation merge
    #[serde(rename = "moe_dense_distill")]
    MoeDenseDistill {
        teacher: Option<PathBuf>,
        #[serde(default = "default_distill_temp")]
        temperature: f32,
    },
    /// Heterogeneous cross-family merge (Union/Intersection)
    #[serde(rename = "hetero")]
    Hetero {
        #[serde(default = "default_hetero_mode")]
        mode: String,
        #[serde(default)]
        weights: Vec<f32>,
    },
    /// Chimera compat-gated merge (frozen-router MoE)
    #[serde(rename = "chimera")]
    Chimera {
        #[serde(default = "default_chimera_threshold")]
        threshold: f32,
    },
    /// Aether-7B-5Attn Latin-square MoE layer remap
    #[serde(rename = "aether")]
    Aether {
        #[serde(default = "default_aether_grid")]
        grid: usize,
    },
    /// POCKET-35B structured width prune (experts keep count)
    #[serde(rename = "pocket")]
    Pocket {
        #[serde(default = "default_pocket_keep")]
        keep: usize,
    },
}

/// Quantization method selection
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum QuantMethod {
    /// JangQ mixed-precision for MLX
    #[serde(rename = "jang")]
    Jang {
        profile: String,
        #[serde(default)]
        output_format: JangOutputFormat,
    },
    /// Unsloth Dynamic 3.0
    #[serde(rename = "dynamic3")]
    Dynamic3 {
        density: f32,
        #[serde(default = "default_true")]
        model_specific: bool,
    },
    /// Apex MoE-aware quantization
    #[serde(rename = "apex")]
    Apex { tier: String },
    /// BTL4 compact quantization
    #[serde(rename = "btl4")]
    Btl4Compact { target_bpw: f32 },
    /// Generic mixed precision
    #[serde(rename = "mixed")]
    MixedPrecision { per_layer_bits: Vec<(String, u8)> },
    /// BSQAT block-scaled quantization (4-bit default)
    #[serde(rename = "bsqat")]
    Bsqat {
        #[serde(default = "default_bsqat_bits")]
        bits: u8,
        #[serde(default = "default_bsqat_block")]
        block: usize,
    },
    /// OneComp one-line progressive quantization
    #[serde(rename = "onecomp")]
    OneComp {
        #[serde(default = "default_onecomp_hi")]
        hi_bits: u8,
        #[serde(default = "default_onecomp_lo")]
        lo_bits: u8,
    },
    /// QuEPT elastic-precision quantization
    #[serde(rename = "quept")]
    QuEPT {
        #[serde(default = "default_quept_width")]
        width: u8,
    },
    /// NanoQuant LB-ADMM sub-1-bit (ICML-26)
    #[serde(rename = "nanoquant")]
    NanoQuant {
        #[serde(default = "default_nano_bits")]
        bits: u8,
        #[serde(default = "default_nano_iters")]
        iters: usize,
    },
    /// ARB-LLM alternating refinement + codebook-gradient (1-bit)
    #[serde(rename = "arb")]
    Arb {
        #[serde(default = "default_arb_bits")]
        bits: u8,
        #[serde(default = "default_true")]
        reorder: bool,
    },
    /// HBLLM wavelet-domain binary coding
    #[serde(rename = "hbllm")]
    Hbllm {
        #[serde(default = "default_hbllm_levels")]
        levels: u8,
    },
    /// DBellQuant dual-bell non-uniform (strict 1-bit)
    #[serde(rename = "dbell")]
    Dbell {
        #[serde(default = "default_dbell_bits")]
        bits: u8,
    },
    /// AF1 strict-1.00bpw (no residual)
    #[serde(rename = "af1")]
    Af1 {
        #[serde(default = "default_af1_bits")]
        bits: u8,
    },
    /// BTC-LLM vector codebook
    #[serde(rename = "btcllm")]
    BtcLlm {
        #[serde(default = "default_btc_bits")]
        bits: u8,
        #[serde(default = "default_btc_codebook")]
        codebook: usize,
    },
    /// LittleBit residual-compensated sub-1-bit (0.1-1.0bpw)
    #[serde(rename = "littlebit")]
    LittleBit {
        #[serde(default = "default_little_bits")]
        bits: u8,
        #[serde(default = "default_true")]
        residual: bool,
    },
}

fn default_true() -> bool { true }
fn default_lambda_one() -> f32 { 1.0 }
fn default_breadcrumb_beta() -> f32 { 0.1 }
fn default_breadcrumb_gamma() -> f32 { 0.1 }
fn default_arcee_thresh() -> f32 { 1.0 }
fn default_nearswap_t() -> f32 { 0.5 }
fn default_nearswap_thresh() -> f32 { 0.1 }
fn default_ram_seed() -> u64 { 42 }
fn default_karcher_iter() -> usize { 20 }
fn default_karcher_tol() -> f32 { 1e-5 }
fn default_frankenmoe_bottom() -> usize { 4 }
fn default_frankenmoe_experts() -> usize { 8 }
fn default_frankenmoe_top() -> usize { 2 }
fn default_orca_threshold() -> f32 { 3.0 }
fn default_latent_dim() -> usize { 512 }
fn default_num_experts() -> usize { 8 }
fn default_distill_temp() -> f32 { 1.0 }
fn default_hetero_mode() -> String { "union".to_string() }
fn default_bsqat_bits() -> u8 { 4 }
fn default_bsqat_block() -> usize { 128 }
fn default_onecomp_hi() -> u8 { 4 }
fn default_onecomp_lo() -> u8 { 2 }
fn default_quept_width() -> u8 { 4 }
fn default_chimera_threshold() -> f32 { 0.5 }
fn default_aether_grid() -> usize { 7 }
fn default_pocket_keep() -> usize { 128 }
fn default_nano_bits() -> u8 { 1 }
fn default_nano_iters() -> usize { 50 }
fn default_arb_bits() -> u8 { 1 }
fn default_hbllm_levels() -> u8 { 3 }
fn default_dbell_bits() -> u8 { 1 }
fn default_af1_bits() -> u8 { 1 }
fn default_btc_bits() -> u8 { 1 }
fn default_btc_codebook() -> usize { 256 }
fn default_little_bits() -> u8 { 1 }

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum JangOutputFormat {
    /// MLX-native .jang safetensors
    #[serde(rename = "mlx")]
    Mlx,
    /// GGUF with JangQ-style mixed precision
    #[serde(rename = "gguf")]
    Gguf,
}

impl Default for JangOutputFormat {
    fn default() -> Self { JangOutputFormat::Mlx }
}

/// Output configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutputConfig {
    pub path: PathBuf,
    #[serde(default = "default_dtype")]
    pub dtype: crate::DType,
    #[serde(default)]
    pub shard_size: Option<usize>,
}

fn default_dtype() -> crate::DType { crate::DType::BF16 }

/// Top-level merge configuration (mergekit YAML compatible)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MergeConfig {
    pub merge_method: MergeMethod,
    #[serde(default)]
    pub base_model: Option<PathBuf>,
    pub models: Vec<ModelEntry>,
    #[serde(default)]
    pub slices: Vec<SliceSpec>,
    pub dtype: crate::DType,
    #[serde(default)]
    pub parameters: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub output: Option<OutputConfig>,
    #[serde(default)]
    pub quant: Option<QuantMethod>,
    #[serde(default)]
    pub darwin: Option<DarwinConfig>,
}

/// Darwin-specific configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DarwinConfig {
    pub generations: usize,
    pub population: usize,
    #[serde(default)]
    pub benchmark: Option<String>,
    #[serde(default)]
    pub tau_init: f32,
    #[serde(default = "default_lambda")]
    pub lambda: f32,
}

impl Default for DarwinConfig {
    fn default() -> Self {
        Self {
            generations: 30,
            population: 40,
            benchmark: None,
            tau_init: 0.45,
            lambda: 0.1,
        }
    }
}

fn default_lambda() -> f32 { 0.1 }

/// Eval configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EvalConfig {
    pub model: PathBuf,
    #[serde(default)]
    pub benchmarks: Vec<String>,
    #[serde(default)]
    pub evals: Vec<String>,
    #[serde(default)]
    pub original_model: Option<PathBuf>,
}

/// Fusing pipeline step
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FuseStep {
    pub name: String,
    pub operation: FuseOperation,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum FuseOperation {
    ExtractLora,
    MergeLoras,
    FuseIntoBase,
    Quantize,
    Evaluate,
}
