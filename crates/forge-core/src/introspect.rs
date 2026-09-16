//! 0A. Model Introspection Layer.
//!
//! Runtime detection of architecture family + structural features from
//! `config.json` and the safetensors tensor inventory. Zero hardcoded
//! per-model assumptions in downstream crates: everything consumes
//! [`ModelProfile`], and family quirks live as declarative metadata in
//! [`FamilyRegistry`] (extensible via TOML, no core changes needed).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Feature enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    Mha,
    Gqa,
    Mqa,
    Mla,
    SlidingWindow,
    Linear,
    Ssm,
    Hybrid,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnType {
    SwiGlu,
    GeGlu,
    GatedMlp,
    StandardMlp,
    MoeTopK,
    MoeExpertChoice,
    MoeSharedExpert,
    MambaMixer,
    RwkvChannelMix,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NormType {
    RmsNorm,
    LayerNorm,
    PreNorm,
    PostNorm,
    Parallel,
    QkNorm,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PositionalType {
    Rope,
    RopeScaled,
    YaRN,
    Ntk,
    Alibi,
    Nope,
    Learned,
    Relative,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerType {
    Bpe,
    SentencePiece,
    Tiktoken,
    Unigram,
    WordPiece,
    Unknown,
}

// ---------------------------------------------------------------------------
// Family descriptor + registry
// ---------------------------------------------------------------------------

/// Declarative per-family metadata. New families are added as data
/// (TOML or [`FamilyRegistry::register`]) — never as core code branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyDescriptor {
    pub id: String,
    /// HF `model_type` values that map to this family.
    pub model_types: Vec<String>,
    /// HF `architectures` class names that map to this family.
    pub architectures: Vec<String>,
    /// Dotted container holding the repeated blocks, e.g. `model.layers`.
    pub layer_container: String,
    /// Infix between `<container>.<idx>.` and the projection name,
    /// e.g. `self_attn`, `self_attention`, `mixer`.
    pub attn_infix: String,
    pub q_names: Vec<String>,
    pub k_names: Vec<String>,
    pub v_names: Vec<String>,
    pub o_names: Vec<String>,
    pub gate_names: Vec<String>,
    pub up_names: Vec<String>,
    pub down_names: Vec<String>,
    pub fused_qkv: bool,
    pub fused_gate_up: bool,
    pub gated_ffn: bool,
    pub moe: bool,
    pub ssm_backbone: bool,
    pub hybrid_ssm_attn: bool,
    pub sliding_window: bool,
    pub mla: bool,
    pub qkv_bias: bool,
    /// Declarative quirks, e.g. `qwen3.5 attention_bias drops output bias`.
    pub quirks: Vec<String>,
}

impl FamilyDescriptor {
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: &str,
        model_types: &[&str],
        architectures: &[&str],
        layer_container: &str,
        attn_infix: &str,
    ) -> Self {
        Self {
            id: id.to_string(),
            model_types: model_types.iter().map(|s| s.to_string()).collect(),
            architectures: architectures.iter().map(|s| s.to_string()).collect(),
            layer_container: layer_container.to_string(),
            attn_infix: attn_infix.to_string(),
            q_names: vec!["q_proj".into(), "wq".into(), "query_key_value".into()],
            k_names: vec!["k_proj".into(), "wk".into()],
            v_names: vec!["v_proj".into(), "wv".into()],
            o_names: vec!["o_proj".into(), "wo".into(), "dense".into()],
            gate_names: vec!["gate_proj".into(), "w1".into(), "gate".into()],
            up_names: vec!["up_proj".into(), "w3".into()],
            down_names: vec!["down_proj".into(), "w2".into(), "output".into()],
            fused_qkv: false,
            fused_gate_up: false,
            gated_ffn: true,
            moe: false,
            ssm_backbone: false,
            hybrid_ssm_attn: false,
            sliding_window: false,
            mla: false,
            qkv_bias: false,
            quirks: vec![],
        }
    }
}

/// Plugin-style registry: builtin coverage for 30 families, plus TOML loading
/// and runtime registration for future architectures.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FamilyRegistry {
    pub families: Vec<FamilyDescriptor>,
}

impl FamilyRegistry {
    pub fn builtin() -> Self {
        let mut r = Self { families: vec![] };
        // --- Llama (also covers SmolLM, TinyLlama: same model_type/arch) ---
        let mut llama = FamilyDescriptor::new(
            "llama",
            &["llama"],
            &["LlamaForCausalLM"],
            "model.layers",
            "self_attn",
        );
        llama.quirks.push("covers smollm/tinyllama/stablelm-llama-arch via shared llama model_type".into());
        r.families.push(llama);
        // --- Mistral ---
        let mut mistral = FamilyDescriptor::new(
            "mistral",
            &["mistral"],
            &["MistralForCausalLM"],
            "model.layers",
            "self_attn",
        );
        mistral.sliding_window = true;
        mistral.quirks.push("sliding_window=4096 typical; verify per checkpoint".into());
        r.families.push(mistral);
        // --- Mixtral (MoE) ---
        let mut mixtral = FamilyDescriptor::new(
            "mixtral",
            &["mixtral"],
            &["MixtralForCausalLM"],
            "model.layers",
            "self_attn",
        );
        mixtral.moe = true;
        mixtral.sliding_window = true;
        mixtral.quirks.push("top-2 router, 8 experts typical".into());
        r.families.push(mixtral);
        // --- Gemma / Gemma2 ---
        r.families.push(FamilyDescriptor::new(
            "gemma",
            &["gemma"],
            &["GemmaForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        let mut gemma2 = FamilyDescriptor::new(
            "gemma2",
            &["gemma2"],
            &["Gemma2ForCausalLM"],
            "model.layers",
            "self_attn",
        );
        gemma2.sliding_window = true;
        gemma2.quirks.push("sliding-window + full-attention interleaved layers".into());
        r.families.push(gemma2);
        // --- Phi ---
        r.families.push(FamilyDescriptor::new(
            "phi",
            &["phi", "phi3"],
            &["PhiForCausalLM", "Phi3ForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        // --- DeepSeek V2/V3 (MLA + MoE) ---
        let mut dsv2 = FamilyDescriptor::new(
            "deepseek_v2",
            &["deepseek_v2"],
            &["DeepseekV2ForCausalLM"],
            "model.layers",
            "self_attn",
        );
        dsv2.mla = true;
        dsv2.quirks.push("MLA: fused kv_a_proj_with_mqa + q_a_proj; no plain k_proj".into());
        r.families.push(dsv2);
        let mut dsv3 = FamilyDescriptor::new(
            "deepseek_v3",
            &["deepseek_v3"],
            &["DeepseekV3ForCausalLM"],
            "model.layers",
            "self_attn",
        );
        dsv3.mla = true;
        dsv3.moe = true;
        dsv3.quirks.push("MLA + fine-grained MoE + shared expert".into());
        r.families.push(dsv3);
        // --- Qwen2 / Qwen3 / Qwen3-MoE / Qwen3.5 ---
        let mut qwen2 = FamilyDescriptor::new(
            "qwen2",
            &["qwen2"],
            &["Qwen2ForCausalLM", "Qwen2MoeForCausalLM"],
            "model.layers",
            "self_attn",
        );
        qwen2.qkv_bias = true;
        qwen2.quirks.push("qkv has bias unlike most llama-arch models".into());
        r.families.push(qwen2);
        r.families.push(FamilyDescriptor::new(
            "qwen3",
            &["qwen3"],
            &["Qwen3ForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        let mut qwen3moe = FamilyDescriptor::new(
            "qwen3_moe",
            &["qwen3_moe"],
            &["Qwen3MoeForCausalLM"],
            "model.layers",
            "self_attn",
        );
        qwen3moe.moe = true;
        qwen3moe.quirks.push("shared expert + qwen3 attention_bias semantics".into());
        r.families.push(qwen3moe);
        let mut qwen35 = FamilyDescriptor::new(
            "qwen3_5",
            &["qwen3_5"],
            &["Qwen3_5ForCausalLM", "Qwen3_5MoeForCausalLM"],
            "model.layers",
            "self_attn",
        );
        qwen35.quirks.push("attention_bias drops output bias; GDN + full-attn hybrid blocks".into());
        r.families.push(qwen35);
        // --- Falcon (fused QKV) ---
        let mut falcon = FamilyDescriptor::new(
            "falcon",
            &["falcon", "RefinedWebModel"],
            &["FalconForCausalLM", "RWForCausalLM"],
            "transformer.h",
            "self_attention",
        );
        falcon.fused_qkv = true;
        falcon.gated_ffn = false;
        falcon.quirks.push("fused query_key_value tensor; parallel attn+mlp".into());
        r.families.push(falcon);
        // --- Cohere / Command-R ---
        r.families.push(FamilyDescriptor::new(
            "cohere",
            &["cohere", "command-r"],
            &["CohereForCausalLM", "Cohere2ForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        // --- Yi ---
        r.families.push(FamilyDescriptor::new(
            "yi",
            &["yi"],
            &["YiForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        // --- OLMo ---
        r.families.push(FamilyDescriptor::new(
            "olmo",
            &["olmo", "olmo2"],
            &["OlmoForCausalLM", "Olmo2ForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        // --- DBRX (MoE, blocks container) ---
        let mut dbrx = FamilyDescriptor::new(
            "dbrx",
            &["dbrx"],
            &["DbrxForCausalLM"],
            "transformer.blocks",
            "norm_attn_norm",
        );
        dbrx.moe = true;
        r.families.push(dbrx);
        // --- Grok ---
        let mut grok = FamilyDescriptor::new(
            "grok",
            &["grok"],
            &["GrokForCausalLM"],
            "transformer.decoder_layer",
            "self_attn",
        );
        grok.moe = true;
        r.families.push(grok);
        // --- Mamba (SSM) ---
        let mut mamba = FamilyDescriptor::new(
            "mamba",
            &["mamba", "mamba2"],
            &["MambaForCausalLM", "Mamba2ForCausalLM"],
            "backbone.layers",
            "mixer",
        );
        mamba.ssm_backbone = true;
        mamba.gated_ffn = false;
        mamba.q_names = vec!["in_proj".into()];
        mamba.o_names = vec!["out_proj".into()];
        mamba.quirks.push("no q/k/v/o; scan via in_proj/out_proj + dt/A/D params".into());
        r.families.push(mamba);
        // --- Jamba (hybrid) ---
        let mut jamba = FamilyDescriptor::new(
            "jamba",
            &["jamba"],
            &["JambaForCausalLM"],
            "model.layers",
            "self_attn",
        );
        jamba.hybrid_ssm_attn = true;
        jamba.moe = true;
        jamba.quirks.push("interleaved mamba + attention + MoE layers".into());
        r.families.push(jamba);
        // --- RWKV ---
        let mut rwkv = FamilyDescriptor::new(
            "rwkv",
            &["rwkv"],
            &["RwkvForCausalLM"],
            "rwkv.blocks",
            "att",
        );
        rwkv.ssm_backbone = true;
        rwkv.gated_ffn = false;
        rwkv.quirks.push("time-mix/channel-mix, no attention projections".into());
        r.families.push(rwkv);
        // --- Zamba (hybrid) ---
        let mut zamba = FamilyDescriptor::new(
            "zamba",
            &["zamba"],
            &["ZambaForCausalLM"],
            "model.layers",
            "self_attn",
        );
        zamba.hybrid_ssm_attn = true;
        zamba.quirks.push("shared global attention + mamba blocks".into());
        r.families.push(zamba);
        // --- Aether-7B-5Attn (hetero-attention MoE, 7x7 Latin square) ---
        let mut aether = FamilyDescriptor::new(
            "aether",
            &["aether"],
            &["AetherForCausalLM"],
            "model.layers",
            "self_attn",
        );
        aether.moe = true;
        aether.quirks.push("49 layers, 5 hetero attn types on 7x7 Latin square; see AetherLayout".into());
        r.families.push(aether);
        // --- Nemotron ---
        r.families.push(FamilyDescriptor::new(
            "nemotron",
            &["nemotron"],
            &["NemotronForCausalLM"],
            "model.layers",
            "self_attn",
        ));
        // --- Granite ---
        let mut granite = FamilyDescriptor::new(
            "granite",
            &["granite"],
            &["GraniteForCausalLM", "GraniteMoeForCausalLM"],
            "model.layers",
            "self_attn",
        );
        granite.quirks.push("multiplier on q/k/o; check scaling in config".into());
        r.families.push(granite);
        // --- StableLM / Pythia / GPT-NeoX / OPT / Bloom (classic) ---
        let mut stablelm = FamilyDescriptor::new(
            "stablelm",
            &["stablelm"],
            &["StableLmForCausalLM"],
            "model.layers",
            "self_attn",
        );
        stablelm.gated_ffn = false;
        r.families.push(stablelm);
        let mut pythia = FamilyDescriptor::new(
            "pythia",
            &["gpt_neox"],
            &["GPTNeoXForCausalLM"],
            "gpt_neox.layers",
            "attention",
        );
        pythia.gated_ffn = false;
        pythia.quirks.push("pythia checkpoints use gpt_neox model_type".into());
        r.families.push(pythia);
        let mut opt = FamilyDescriptor::new(
            "opt",
            &["opt"],
            &["OPTForCausalLM"],
            "model.decoder.layers",
            "self_attn",
        );
        opt.gated_ffn = false;
        r.families.push(opt);
        let mut bloom = FamilyDescriptor::new(
            "bloom",
            &["bloom"],
            &["BloomForCausalLM"],
            "transformer.h",
            "self_attention",
        );
        bloom.fused_qkv = true;
        bloom.gated_ffn = false;
        r.families.push(bloom);
        // --- Encoders (for 0D fusion) ---
        r.families.push(FamilyDescriptor::new(
            "bert",
            &["bert"],
            &["BertModel", "BertForMaskedLM"],
            "encoder.layer",
            "attention.self",
        ));
        r.families.push(FamilyDescriptor::new(
            "t5",
            &["t5"],
            &["T5Model", "T5ForConditionalGeneration"],
            "encoder.block",
            "layer.0.SelfAttention",
        ));
        r.families.push(FamilyDescriptor::new(
            "vit",
            &["vit"],
            &["ViTModel"],
            "encoder.layer",
            "attention.attention",
        ));
        r.families.push(FamilyDescriptor::new(
            "whisper",
            &["whisper"],
            &["WhisperModel", "WhisperForConditionalGeneration"],
            "encoder.layers",
            "self_attn",
        ));
        r
    }

    /// Load additional / overriding families from a TOML file:
    /// `[[family]] id = ... model_types = [...] ...`
    pub fn load_toml_file(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path)?;
        #[derive(Deserialize)]
        struct File {
            #[serde(default)]
            family: Vec<FamilyDescriptor>,
        }
        let file: File = toml::from_str(&text).context("parse family registry TOML")?;
        for fam in file.family {
            self.register(fam);
        }
        Ok(())
    }

    pub fn register(&mut self, fam: FamilyDescriptor) {
        if let Some(slot) = self.families.iter_mut().find(|f| f.id == fam.id) {
            *slot = fam;
        } else {
            self.families.push(fam);
        }
    }

    pub fn get(&self, id: &str) -> Option<&FamilyDescriptor> {
        self.families.iter().find(|f| f.id == id)
    }

    /// Lookup by family id, falling back to a generic descriptor so
    /// unregistered families still align on the shared role vocabulary.
    /// Keeps family-specific naming knowledge in the registry, never in core.
    pub fn get_or_fallback(&self, id: &str) -> FamilyDescriptor {
        self.get(id).cloned().unwrap_or_else(|| {
            FamilyDescriptor::new("unknown", &[], &[], "model.layers", "self_attn")
        })
    }

    /// Score-based match: architecture string first, model_type second,
    /// tensor-name evidence as tiebreak. Never panics on unknown input.
    pub fn detect_family(
        &self,
        model_type: Option<&str>,
        architectures: &[String],
        tensor_names: &[String],
    ) -> Option<&FamilyDescriptor> {
        let mut best: Option<(&FamilyDescriptor, u32)> = None;
        for fam in &self.families {
            let mut score = 0u32;
            for arch in architectures {
                if fam.architectures.iter().any(|a| a == arch) {
                    score += 100;
                }
            }
            if let Some(mt) = model_type {
                if fam.model_types.iter().any(|m| m == mt) {
                    score += 50;
                }
            }
            // Tiebreak: +1 per tensor matching this family's container/infix.
            if score > 0 {
                let hits = tensor_names
                    .iter()
                    .filter(|n| {
                        n.contains(&fam.layer_container) || n.contains(&fam.attn_infix)
                    })
                    .take(8)
                    .count() as u32;
                score += hits;
            }
            if score > 0 && best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((fam, score));
            }
        }
        best.map(|(f, _)| f)
    }
}

// ---------------------------------------------------------------------------
// ModelProfile
// ---------------------------------------------------------------------------

/// Architecture-agnostic model description. Every downstream crate consumes
/// this — never raw config keys or family branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub family: String,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub tie_embeddings: bool,
    pub attention: AttentionType,
    pub ffn: FfnType,
    pub norm: NormType,
    pub positional: PositionalType,
    pub tokenizer: TokenizerType,
    pub moe_experts: Option<usize>,
    pub quirks: Vec<String>,
}

impl ModelProfile {
    /// Detect from a parsed `config.json` plus `(name, shape)` inventory.
    /// Pure function — IO stays at the call site (CLI feeds TensorStore names).
    pub fn detect_from_parts(
        config: &serde_json::Value,
        tensors: &[(String, Vec<usize>)],
        registry: &FamilyRegistry,
    ) -> Result<Self> {
        let get = |k: &str| config.get(k);
        let u = |k: &str, d: usize| get(k).and_then(|v| v.as_u64()).unwrap_or(d as u64) as usize;
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let architectures: Vec<String> = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
            .unwrap_or_default();
        let names: Vec<String> = tensors.iter().map(|(n, _)| n.clone()).collect();
        let fam = registry
            .detect_family(model_type.as_deref(), &architectures, &names)
            .cloned()
            .unwrap_or_else(|| FamilyDescriptor::new("unknown", &[], &[], "model.layers", "self_attn"));

        let num_heads = u("num_attention_heads", u("n_head", 32));
        let kv_heads = u("num_key_value_heads", u("n_head_kv", num_heads));
        let hidden = u("hidden_size", u("n_embd", 4096));
        let head_dim = u("head_dim", hidden.checked_div(num_heads.max(1)).unwrap_or(128));

        // Attention type from structural evidence.
        let attention = if fam.mla {
            AttentionType::Mla
        } else if fam.ssm_backbone && !fam.hybrid_ssm_attn {
            AttentionType::Ssm
        } else if fam.hybrid_ssm_attn {
            AttentionType::Hybrid
        } else if get("linear_attn").is_some() || model_type.as_deref() == Some("zamba") && false {
            AttentionType::Linear
        } else if fam.sliding_window || get("sliding_window").is_some() {
            // Sliding-window models still use GQA underneath; flag the window.
            if kv_heads == 1 {
                AttentionType::Mqa
            } else if kv_heads < num_heads {
                AttentionType::Gqa
            } else {
                AttentionType::SlidingWindow
            }
        } else if kv_heads == 1 {
            AttentionType::Mqa
        } else if kv_heads < num_heads {
            AttentionType::Gqa
        } else {
            AttentionType::Mha
        };

        // FFN type from tensor evidence + family flags.
        let has_gate = names.iter().any(|n| n.contains("gate_proj") || n.ends_with(".w1"));
        let has_router = names.iter().any(|n| n.contains("block_sparse_moe") || n.contains("router") || n.contains("gate") && n.contains("experts"));
        let has_shared = names.iter().any(|n| n.contains("shared_expert"));
        let moe_experts = if fam.moe || has_router {
            let n = config
                .get("num_local_experts")
                .or_else(|| config.get("n_routed_experts"))
                .or_else(|| config.get("num_experts"))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize);
            // Count expert dirs as fallback.
            n.or_else(|| {
                let mut idx = std::collections::HashSet::new();
                for n in &names {
                    for part in n.split('.') {
                        if let Ok(i) = part.parse::<usize>() {
                            if n.contains("expert") {
                                idx.insert(i);
                            }
                        }
                    }
                }
                if idx.is_empty() { None } else { Some(idx.len()) }
            })
        } else {
            None
        };
        let ffn = if fam.ssm_backbone && !fam.hybrid_ssm_attn && model_type.as_deref() == Some("mamba") {
            FfnType::MambaMixer
        } else if model_type.as_deref() == Some("rwkv") {
            FfnType::RwkvChannelMix
        } else if moe_experts.is_some() {
            if has_shared {
                FfnType::MoeSharedExpert
            } else {
                FfnType::MoeTopK
            }
        } else if has_gate || fam.gated_ffn {
            // SwiGLU vs GeGLU from config hint; default SwiGLU.
            let act = config
                .get("hidden_act")
                .and_then(|v| v.as_str())
                .unwrap_or("silu");
            if act.contains("gelu") {
                FfnType::GeGlu
            } else {
                FfnType::SwiGlu
            }
        } else {
            FfnType::StandardMlp
        };

        // Norm: rms flag or epsilon key naming.
        let norm = if names.iter().any(|n| n.contains("layernorm") || n.contains("LayerNorm")) {
            NormType::LayerNorm
        } else if names.iter().any(|n| n.contains("qk_norm") || n.contains("q_norm")) {
            NormType::QkNorm
        } else {
            NormType::RmsNorm
        };

        // Positional.
        let rope_theta = get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(10000.0);
        let positional = if get("rope_scaling").is_some() {
            let s = get("rope_scaling").unwrap();
            let t = s.get("type").and_then(|v| v.as_str()).unwrap_or("");
            if t.contains("yarn") {
                PositionalType::YaRN
            } else if t.contains("ntk") || t.contains("dynamic") {
                PositionalType::Ntk
            } else {
                PositionalType::RopeScaled
            }
        } else if rope_theta != 10000.0 || get("rotary_emb").is_some() || get("rope").is_some() {
            PositionalType::Rope
        } else if get("alibi").is_some() || model_type.as_deref() == Some("bloom") {
            PositionalType::Alibi
        } else if fam.ssm_backbone && !fam.hybrid_ssm_attn {
            PositionalType::Nope
        } else {
            PositionalType::Rope
        };

        // Layer count: max index under the family container, fallback config keys.
        let num_layers = max_layer_index(&names, &fam.layer_container)
            .map(|m| m + 1)
            .unwrap_or_else(|| u("num_hidden_layers", u("n_layer", u("num_layers", 32))));

        Ok(Self {
            family: fam.id.clone(),
            num_layers,
            hidden_size: hidden,
            num_heads,
            kv_heads,
            head_dim,
            intermediate_size: u("intermediate_size", u("n_inner", hidden * 4)),
            vocab_size: u("vocab_size", u("padded_vocab_size", 32000)),
            tie_embeddings: get("tie_word_embeddings")
                .and_then(|v| v.as_bool())
                .unwrap_or_else(|| {
                    // Tied if no lm_head tensor present.
                    !names.iter().any(|n| n.contains("lm_head") || n.contains("output"))
                }),
            attention,
            ffn,
            norm,
            positional,
            tokenizer: TokenizerType::Unknown,
            moe_experts,
            quirks: fam.quirks.clone(),
        })
    }

    /// Load `config.json` from a model dir and detect. Convenience wrapper.
    pub fn detect(dir: &Path, tensors: &[(String, Vec<usize>)]) -> Result<Self> {
        let text = std::fs::read_to_string(dir.join("config.json")).context("read config.json")?;
        let config: serde_json::Value = serde_json::from_str(&text)?;
        Self::detect_from_parts(&config, tensors, &FamilyRegistry::builtin())
    }

    pub fn kv_groups(&self) -> usize {
        if self.num_heads == 0 {
            1
        } else {
            self.num_heads / self.kv_heads.max(1)
        }
    }
}

fn max_layer_index(names: &[String], container: &str) -> Option<usize> {
    let tail = container.rsplit('.').next().unwrap_or(container);
    let mut best = None;
    for n in names {
        let segs: Vec<&str> = n.split('.').collect();
        for w in segs.windows(2) {
            if w[0] == tail {
                if let Ok(i) = w[1].parse::<usize>() {
                    best = Some(best.map(|b: usize| b.max(i)).unwrap_or(i));
                }
            }
        }
    }
    // Fallback: any `<digits>` segment directly before a known projection name.
    if best.is_none() {
        for n in names {
            let segs: Vec<&str> = n.split('.').collect();
            for w in segs.windows(2) {
                if ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "attention", "mlp", "self_attn", "input_layernorm"]
                    .contains(&w[1])
                {
                    if let Ok(i) = w[0].parse::<usize>() {
                        best = Some(best.map(|b: usize| b.max(i)).unwrap_or(i));
                    }
                }
            }
        }
    }
    best
}

/// Flat registry serialized for `forge inspect` output.
pub fn registry_summary() -> Vec<HashMap<String, String>> {
    FamilyRegistry::builtin()
        .families
        .iter()
        .map(|f| {
            let mut m = HashMap::new();
            m.insert("id".into(), f.id.clone());
            m.insert("model_types".into(), f.model_types.join(","));
            m.insert("moe".into(), f.moe.to_string());
            m.insert("quirks".into(), f.quirks.join("; "));
            m
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests (success criterion: >=3 per new module)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn llama_tensors() -> Vec<(String, Vec<usize>)> {
        vec![
            ("model.embed_tokens.weight".into(), vec![32000, 4096]),
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![4096, 4096]),
            ("model.layers.0.self_attn.k_proj.weight".into(), vec![1024, 4096]),
            ("model.layers.0.mlp.gate_proj.weight".into(), vec![11008, 4096]),
            ("model.layers.31.self_attn.q_proj.weight".into(), vec![4096, 4096]),
            ("lm_head.weight".into(), vec![32000, 4096]),
        ]
    }

    fn llama_config() -> serde_json::Value {
        json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "hidden_act": "silu",
            "rope_theta": 500000.0
        })
    }

    #[test]
    fn detects_llama_gqa_profile() {
        let p = ModelProfile::detect_from_parts(&llama_config(), &llama_tensors(), &FamilyRegistry::builtin()).unwrap();
        assert_eq!(p.family, "llama");
        assert_eq!(p.num_layers, 32);
        assert_eq!(p.attention, AttentionType::Gqa);
        assert_eq!(p.ffn, FfnType::SwiGlu);
        assert_eq!(p.norm, NormType::RmsNorm);
    }

    #[test]
    fn registry_covers_20_plus_families() {
        assert!(FamilyRegistry::builtin().families.len() >= 20);
    }

    #[test]
    fn unknown_model_never_panics() {
        let p = ModelProfile::detect_from_parts(&json!({}), &[], &FamilyRegistry::builtin()).unwrap();
        assert_eq!(p.family, "unknown");
        assert_eq!(p.num_layers, 32); // config default
    }

    #[test]
    fn detects_moe_from_tensors() {
        let mut t = llama_tensors();
        t.push(("model.layers.0.block_sparse_moe.gate.weight".into(), vec![8, 4096]));
        t.push(("model.layers.0.block_sparse_moe.experts.0.w1.weight".into(), vec![14336, 4096]));
        let mut c = llama_config();
        c["model_type"] = json!("mixtral");
        c["architectures"] = json!(["MixtralForCausalLM"]);
        let p = ModelProfile::detect_from_parts(&c, &t, &FamilyRegistry::builtin()).unwrap();
        assert_eq!(p.family, "mixtral");
        assert!(matches!(p.ffn, FfnType::MoeTopK));
        assert!(p.moe_experts.is_some());
    }

    #[test]
    fn detects_mla_and_rope_scaling() {
        let mut c = llama_config();
        c["model_type"] = json!("deepseek_v3");
        c["architectures"] = json!(["DeepseekV3ForCausalLM"]);
        c["rope_scaling"] = json!({"type": "yarn", "factor": 40.0});
        let p = ModelProfile::detect_from_parts(&c, &llama_tensors(), &FamilyRegistry::builtin()).unwrap();
        assert_eq!(p.attention, AttentionType::Mla);
        assert_eq!(p.positional, PositionalType::YaRN);
    }

    #[test]
    fn detects_aether_family_and_latin_square_geometry() {
        let tensors = vec![
            ("model.layers.0.self_attn.q_proj.weight".into(), vec![4096, 4096]),
            ("model.layers.48.self_attn.q_proj.weight".into(), vec![4096, 4096]),
            ("model.layers.0.block_sparse_moe.gate.weight".into(), vec![8, 4096]),
            ("model.layers.0.block_sparse_moe.experts.0.w1.weight".into(), vec![14336, 4096]),
            ("model.layers.0.block_sparse_moe.experts.7.w1.weight".into(), vec![14336, 4096]),
        ];
        let c = serde_json::json!({
            "model_type": "aether",
            "architectures": ["AetherForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 49,
        });
        let p = ModelProfile::detect_from_parts(&c, &tensors, &FamilyRegistry::builtin()).unwrap();
        assert_eq!(p.family, "aether");
        assert_eq!(p.num_layers, 49);
        assert!(p.moe_experts.is_some());
        assert!(p.quirks.iter().any(|q| q.contains("Latin square")));
    }
}
