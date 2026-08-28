#[test]
fn jang_tier_classification() {
    use forge_io::jang_io::JangTier;
    assert_eq!(JangTier::classify("model.layers.0.self_attn.q_proj.weight"), JangTier::Critical);
    assert_eq!(JangTier::classify("model.embed_tokens.weight"), JangTier::Important);
    assert_eq!(JangTier::classify("model.layers.0.mlp.down_proj.weight"), JangTier::Compress);
}
#[test]
fn darwin_genome_roundtrip() {
    let g = forge_darwin::DarwinGenome::random(42);
    let v = g.tensor_ratio("model.layers.5.self_attn.q_proj.weight", Some(5), 32);
    assert!(v >= 0.0 && v <= 1.0);
}
