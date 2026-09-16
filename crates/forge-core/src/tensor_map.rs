//! 0B keystone: canonical tensor-slot map built from a [`ModelProfile`].
//!
//! Every merge / surgery / quant path aligns tensors by **canonical slot**,
//! never by raw layer index. A slot is a `(block, role)` pair where role is a
//! stable vocabulary (`attn.q_proj`, `ffn.down_proj`, `tok_emb`, …) and block
//! is the transformer-block ordinal when the tensor belongs to one.
//!
//! The map is built from tensor *names* alone — no weights are loaded — so it
//! works on 40B+ checkpoints inside the streaming memory budget.
//!
//! Family-specific naming knowledge lives in the [`FamilyDescriptor`]
//! (resolved through the [`FamilyRegistry`]); this module contains zero
//! `if family == …` branches.

use crate::introspect::{FamilyDescriptor, FamilyRegistry, ModelProfile};
use anyhow::Result;
use std::collections::HashMap;

/// Stable per-role vocabulary shared by every family.
pub const ROLE_VOCAB: &[&str] = &[
    "tok_emb",
    "attn.q_proj",
    "attn.k_proj",
    "attn.v_proj",
    "attn.o_proj",
    "attn.q_norm",
    "attn.k_norm",
    "ffn.gate_proj",
    "ffn.up_proj",
    "ffn.down_proj",
    "attn_norm",
    "ffn_norm",
    "final_norm",
    "lm_head",
    "score_head",
];

/// A canonical slot: one alignable position in a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalSlot {
    /// Block ordinal, or `None` for model-global tensors (embeddings, head).
    pub block: Option<usize>,
    /// Role from [`ROLE_VOCAB`] (or a family-specific passthrough role).
    pub role: String,
}

impl CanonicalSlot {
    pub fn named(block: Option<usize>, role: impl Into<String>) -> Self {
        Self { block, role: role.into() }
    }

    /// Human-readable key, e.g. `b12.ffn.down_proj` or `global.lm_head`.
    pub fn key(&self) -> String {
        match self.block {
            Some(b) => format!("b{}.{}", b, self.role),
            None => format!("global.{}", self.role),
        }
    }
}

/// One aligned entry: the slot plus every raw tensor name that fills it.
#[derive(Debug, Clone, Default)]
pub struct SlotEntry {
    pub slot: Option<CanonicalSlot>,
    pub raw_names: Vec<String>,
    /// True when >1 raw tensors share this slot (e.g. QKV-fused checkpoints).
    pub fused: bool,
}

/// Bidirectional map between raw tensor names and canonical slots.
#[derive(Debug, Clone, Default)]
pub struct TensorMap {
    /// slot-key -> entry
    pub slots: HashMap<String, SlotEntry>,
    /// raw tensor name -> slot-key
    pub by_raw: HashMap<String, String>,
    /// Slots present in *every* profile this map was intersected over.
    pub common: Vec<String>,
}

impl TensorMap {
    /// Build a map for a single profile (no I/O: names come from the caller).
    /// The family descriptor is resolved from the registry by profile id,
    /// with a generic fallback for unregistered families.
    pub fn build(
        profile: &ModelProfile,
        tensor_names: &[String],
        registry: &FamilyRegistry,
    ) -> Result<Self> {
        let desc = registry.get_or_fallback(&profile.family);
        let mut map = Self::default();
        for raw in tensor_names {
            let slot = classify(&desc, profile.num_layers, raw);
            let key = slot.key();
            map.by_raw.insert(raw.clone(), key.clone());
            let entry = map.slots.entry(key).or_insert_with(|| SlotEntry {
                slot: Some(slot),
                ..Default::default()
            });
            entry.raw_names.push(raw.clone());
            if entry.raw_names.len() > 1 {
                entry.fused = true;
            }
        }
        map.common = map.slots.keys().cloned().collect();
        map.common.sort();
        Ok(map)
    }

    /// Intersect several single-model maps: keep only shared slots.
    pub fn intersect(maps: &[TensorMap]) -> Self {
        if maps.is_empty() {
            return Self::default();
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for m in maps {
            for k in m.slots.keys() {
                *counts.entry(k.clone()).or_default() += 1;
            }
        }
        let mut out = Self::default();
        for (k, c) in counts {
            if c == maps.len() {
                if let Some(entry) = maps[0].slots.get(&k) {
                    out.slots.insert(k.clone(), entry.clone());
                    for raw in &entry.raw_names {
                        out.by_raw.insert(raw.clone(), k.clone());
                    }
                    out.common.push(k);
                }
            }
        }
        out.common.sort();
        out
    }

    /// Union of several maps (hetero path): every slot any parent provides.
    pub fn union(maps: &[TensorMap]) -> Self {
        let mut out = Self::default();
        for m in maps {
            for (k, entry) in &m.slots {
                let slot_entry = out.slots.entry(k.clone()).or_insert_with(|| SlotEntry {
                    slot: entry.slot.clone(),
                    ..Default::default()
                });
                for raw in &entry.raw_names {
                    if !slot_entry.raw_names.contains(raw) {
                        slot_entry.raw_names.push(raw.clone());
                    }
                    out.by_raw.insert(raw.clone(), k.clone());
                }
            }
        }
        for e in out.slots.values_mut() {
            e.fused = e.raw_names.len() > 1;
        }
        out.common = out.slots.keys().cloned().collect();
        out.common.sort();
        out
    }

    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub fn common_count(&self) -> usize {
        self.common.len()
    }
}

/// Classify one raw tensor name into a canonical slot.
///
/// Matching is segment-aware against the family descriptor's projection-name
/// lists so vendor renames still resolve — the names live in the descriptor,
/// not in branches here.
fn classify(desc: &FamilyDescriptor, num_layers: usize, raw: &str) -> CanonicalSlot {
    let lower = raw.to_lowercase();
    let block = extract_block(desc, num_layers, &lower);
    let role = detect_role(desc, &lower).unwrap_or_else(|| passthrough_role(&lower));
    CanonicalSlot::named(block, role)
}

/// Extract the block ordinal via the family's `layer_container`
/// (e.g. `model.layers.12.`), with generic fallbacks for unregistered
/// naming (`transformer.h.N`, `blocks.N`, …).
fn extract_block(desc: &FamilyDescriptor, num_layers: usize, lower: &str) -> Option<usize> {
    // Pad with a leading dot so containers at string start
    // (`model.layers.0…`) match the same `.container.` markers.
    let hay = format!(".{lower}");
    // (marker, trusted): the family's own container is high-precision —
    // accept any sane index. Generic fallbacks keep a slack guard against
    // spurious digits elsewhere in the name.
    let mut markers: Vec<(String, bool)> = Vec::new();
    if !desc.layer_container.is_empty() {
        markers.push((format!(".{}.", desc.layer_container.trim_matches('.')), true));
    }
    markers.extend(
        [".layers.", ".h.", ".block.", ".blocks.", ".layer.", ".dense_seq."]
            .iter()
            .map(|s| (s.to_string(), false)),
    );
    for (marker, trusted) in &markers {
        if let Some(idx) = hay.find(marker.as_str()) {
            let rest = &hay[idx + marker.len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<usize>() {
                if *trusted && n < 10_000 || !trusted && n < num_layers.max(1) + 8 {
                    return Some(n);
                }
            }
        }
    }
    None
}

/// Match the role against the descriptor's projection-name lists using
/// dot-segment equality (avoids `output` matching `output_layer`, etc.).
fn detect_role(desc: &FamilyDescriptor, lower: &str) -> Option<String> {
    let segs: Vec<&str> = lower.split(['.', '/']).collect();
    let has = |names: &[String]| names.iter().any(|n| segs.iter().any(|s| *s == n.as_str()));
    if has(&desc.q_names) {
        return Some("attn.q_proj".into());
    }
    if has(&desc.k_names) {
        return Some("attn.k_proj".into());
    }
    if has(&desc.v_names) {
        return Some("attn.v_proj".into());
    }
    if has(&desc.o_names) {
        return Some("attn.o_proj".into());
    }
    if has(&desc.gate_names) {
        return Some("ffn.gate_proj".into());
    }
    if has(&desc.up_names) {
        return Some("ffn.up_proj".into());
    }
    if has(&desc.down_names) {
        return Some("ffn.down_proj".into());
    }
    // Norms / embeddings / heads: generic vocabulary, family-independent.
    if lower.contains("input_layernorm")
        || lower.contains("input_norm")
        || lower.contains("self_attn_layer_norm")
    {
        return Some("attn_norm".into());
    }
    if lower.contains("post_attention_layernorm") || lower.contains("post_attn_norm") {
        return Some("ffn_norm".into());
    }
    if lower.contains("q_norm") || lower.contains("query_norm") {
        return Some("attn.q_norm".into());
    }
    if lower.contains("k_norm") || lower.contains("key_norm") {
        return Some("attn.k_norm".into());
    }
    if segs.iter().any(|s| {
        *s == "embed_tokens"
            || *s == "wte"
            || *s == "word_embeddings"
            || *s == "tok_embeddings"
            || *s == "embedding"
    }) {
        return Some("tok_emb".into());
    }
    if segs.iter().any(|s| *s == "lm_head" || *s == "output" || *s == "unembed") {
        return Some("lm_head".into());
    }
    if segs.iter().any(|s| *s == "score" || *s == "classifier") {
        return Some("score_head".into());
    }
    // Model-global norm: `<...>.norm.weight` with no block marker
    // (per-layer norms are caught by the specific checks above).
    if block_marker_absent(lower) {
        let tail: Vec<&str> = segs.iter().rev().take(2).cloned().collect();
        if tail.len() == 2
            && tail[0] == "weight"
            && matches!(tail[1], "norm" | "final_norm" | "final_layernorm" | "ln_f")
        {
            return Some("final_norm".into());
        }
    }
    None
}

/// Aether-7B-5Attn hetero-attention layout: 5 attention types arranged on a
/// 7x7 Latin square over 49 layers. `attn_for_layer` assigns each layer one
/// of the 5 types via `(row + col) % 5` so every row/column mixes types and
/// no two adjacent layers share a type along either axis.
pub const AETHER_ATTN_TYPES: &[&str] = &["mha", "gqa", "mla", "sliding_window", "linear"];

/// Grid-correct Aether layer→attention-type layout.
#[derive(Debug, Clone)]
pub struct AetherLayout {
    pub grid: usize,
    pub num_layers: usize,
}

impl AetherLayout {
    /// Canonical Aether-7B-5Attn geometry: 7x7 = 49 layers.
    pub fn aether49() -> Self {
        Self { grid: 7, num_layers: 49 }
    }

    /// Attention-type index in `0..5` for a layer ordinal.
    pub fn attn_for_layer(&self, layer: usize) -> usize {
        let g = self.grid.max(1);
        ((layer / g) + (layer % g)) % AETHER_ATTN_TYPES.len()
    }

    /// `(layer, attn_type_idx)` over every layer in the layout.
    pub fn slot_map(&self) -> Vec<(usize, usize)> {
        (0..self.num_layers).map(|l| (l, self.attn_for_layer(l))).collect()
    }

    /// Attn-type-aware alignment: `attn.*` roles align only when both blocks
    /// share the attention type; every other role always aligns. Prevents
    /// hetero merges from mixing e.g. an MLA q_proj with an MHA q_proj.
    pub fn align_ok(&self, slot_a: &CanonicalSlot, slot_b: &CanonicalSlot) -> bool {
        let attn_role = |s: &CanonicalSlot| s.role.starts_with("attn.");
        match (slot_a.block, slot_b.block) {
            (Some(la), Some(lb)) if attn_role(slot_a) && attn_role(slot_b) => {
                self.attn_for_layer(la) == self.attn_for_layer(lb)
            }
            _ => true,
        }
    }
}

/// True when the name carries no block ordinal (global tensor candidate).
fn block_marker_absent(lower: &str) -> bool {    for marker in [".layers.", ".h.", ".block.", ".blocks.", ".layer."] {
        if let Some(idx) = lower.find(marker) {
            let rest = &lower[idx + marker.len()..];
            if rest.chars().next().is_some_and(|c| c.is_ascii_digit()) {
                return false;
            }
        }
    }
    true
}

/// Last-resort role: sanitized tail of the tensor name so exotic tensors
/// (rotary inv_freq, router weights, biases) still get a stable slot.
fn passthrough_role(lower: &str) -> String {
    let tail = lower.rsplit(['.', '/']).next().unwrap_or(lower);
    format!("x.{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::introspect::FamilyRegistry;

    fn setup() -> (FamilyRegistry, ModelProfile) {
        let reg = FamilyRegistry::builtin();
        let config = serde_json::json!({
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "num_hidden_layers": 64,
            "hidden_size": 5120,
        });
        let tensors = vec![(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            vec![5120, 5120],
        )];
        let p = ModelProfile::detect_from_parts(&config, &tensors, &reg).unwrap();
        (reg, p)
    }

    #[test]
    fn classifies_qwen_attention_roles() {
        let (reg, p) = setup();
        let names = vec![
            "model.layers.3.self_attn.q_proj.weight".to_string(),
            "model.layers.3.self_attn.o_proj.weight".to_string(),
            "model.layers.3.mlp.down_proj.weight".to_string(),
        ];
        let map = TensorMap::build(&p, &names, &reg).unwrap();
        assert_eq!(map.slot_count(), 3);
        assert!(map.slots.contains_key("b3.attn.q_proj"));
        assert!(map.slots.contains_key("b3.attn.o_proj"));
        assert!(map.slots.contains_key("b3.ffn.down_proj"));
    }

    #[test]
    fn global_tensors_have_no_block() {
        let (reg, p) = setup();
        let names = vec![
            "model.embed_tokens.weight".to_string(),
            "lm_head.weight".to_string(),
            "model.norm.weight".to_string(),
        ];
        let map = TensorMap::build(&p, &names, &reg).unwrap();
        assert!(map.slots.contains_key("global.tok_emb"));
        assert!(map.slots.contains_key("global.lm_head"));
        assert!(map.slots.contains_key("global.final_norm"));
    }

    #[test]
    fn intersect_keeps_shared_slots_only() {
        let (reg, p) = setup();
        let a = TensorMap::build(
            &p,
            &[
                "model.layers.0.self_attn.q_proj.weight".to_string(),
                "lm_head.weight".to_string(),
            ],
            &reg,
        )
        .unwrap();
        let b = TensorMap::build(&p, &["model.layers.0.self_attn.q_proj.weight".to_string()], &reg)
            .unwrap();
        let i = TensorMap::intersect(&[a, b]);
        assert_eq!(i.common_count(), 1);
        assert!(i.slots.contains_key("b0.attn.q_proj"));
    }

    #[test]
    fn union_keeps_everything() {
        let (reg, p) = setup();
        let a =
            TensorMap::build(&p, &["model.layers.0.self_attn.q_proj.weight".to_string()], &reg)
                .unwrap();
        let b = TensorMap::build(&p, &["lm_head.weight".to_string()], &reg).unwrap();
        let u = TensorMap::union(&[a, b]);
        assert_eq!(u.slot_count(), 2);
    }

    #[test]
    fn block_index_not_confused_by_hidden_size_digits() {
        let (reg, p) = setup();
        let names = vec!["model.layers.12.mlp.gate_proj.weight".to_string()];
        let map = TensorMap::build(&p, &names, &reg).unwrap();
        assert!(map.slots.contains_key("b12.ffn.gate_proj"));
    }

    #[test]
    fn unknown_family_still_aligns_on_generic_vocab() {        let reg = FamilyRegistry::builtin();
        let config = serde_json::json!({"model_type": "mysterymix"});
        let tensors = vec![("mystery.blocks.7.q_proj.weight".to_string(), vec![1024, 1024])];
        let p = ModelProfile::detect_from_parts(&config, &tensors, &reg).unwrap();
        let names = vec!["mystery.blocks.7.q_proj.weight".to_string()];
        let map = TensorMap::build(&p, &names, &reg).unwrap();
        assert!(map.slots.contains_key("b7.attn.q_proj"));
    }

    #[test]
    fn aether_layout_covers_49_layers_with_five_types() {
        let layout = AetherLayout::aether49();
        let map = layout.slot_map();
        assert_eq!(map.len(), 49);
        let mut seen = std::collections::HashSet::new();
        for (_, t) in &map {
            seen.insert(*t);
        }
        assert_eq!(seen.len(), 5);
        // Every row and column of the 7x7 grid mixes types.
        for r in 0..7 {
            let mut row = std::collections::HashSet::new();
            for c in 0..7 {
                row.insert(layout.attn_for_layer(r * 7 + c));
            }
            assert!(row.len() > 1);
        }
    }

    #[test]
    fn aether_align_rejects_mismatched_attn_types() {
        let layout = AetherLayout::aether49();
        let a = CanonicalSlot::named(Some(0), "attn.q_proj");
        // Layer 1 sits at (0,1) -> different type from layer 0 at (0,0).
        assert_ne!(layout.attn_for_layer(0), layout.attn_for_layer(1));
        let b = CanonicalSlot::named(Some(1), "attn.q_proj");
        assert!(!layout.align_ok(&a, &b));
        // Same-type layers align: layer 0 (0,0) and layer 8 (1,1) share (0+0)%5 == (1+1)%5? no —
        // find any same-type pair instead of assuming.
        let t0 = layout.attn_for_layer(0);
        let same = (0..49).find(|l| *l != 0 && layout.attn_for_layer(*l) == t0).unwrap();
        let c = CanonicalSlot::named(Some(same), "attn.q_proj");
        assert!(layout.align_ok(&a, &c));
        // Non-attn roles always align across layers.
        let d = CanonicalSlot::named(Some(0), "ffn.down_proj");
        let e = CanonicalSlot::named(Some(1), "ffn.down_proj");
        assert!(layout.align_ok(&d, &e));
        // Global tensors always align.
        let g = CanonicalSlot::named(None, "lm_head");
        assert!(layout.align_ok(&g, &a));
    }
}
