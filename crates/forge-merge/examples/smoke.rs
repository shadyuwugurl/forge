//! Smoke example over the public `forge-merge` API surface.
//! (Legacy unexported modules are intentionally not exercised here.)

use forge_merge::*;
use forge_merge::orchestrator::MergeOp;
use forge_core::TensorMeta;

fn meta(n: usize) -> TensorMeta {
    TensorMeta { name: "t".into(), shape: vec![n], dtype: forge_core::DType::F32, offset: 0, size: n * 4 }
}

fn main() {
    let m = meta(4);
    let a = vec![1.0, 0.0, 2.0, -1.0];
    let b = vec![0.0, 1.0, 2.0, 1.0];

    // hetero intersection honors explicit weights
    let r = HeteroMerge::with_weights(HeteroMode::Intersection, vec![0.25, 0.75])
        .unwrap()
        .merge(&[vec![4.0], vec![8.0]])
        .unwrap();
    assert!((r[0] - 7.0).abs() < 1e-6, "hetero-weighted {:?}", r);

    // hetero union averages present holders
    let r = HeteroMerge::new(HeteroMode::Union, 2)
        .merge_tensors("t", &meta(2), &[vec![1.0, 3.0], vec![3.0, 5.0]])
        .unwrap();
    assert_eq!(r, vec![2.0, 4.0], "hetero-union {:?}", r);

    // chimera: identical parents average
    let r = ChimeraMerge::new(0.5)
        .merge_tensors("t", &m, &[a.clone(), a.clone()])
        .unwrap();
    assert_eq!(r, a, "chimera-ident {:?}", r);

    // chimera: shape mismatch transplants parent 0
    let r = ChimeraMerge::new(0.0)
        .merge_tensors("t", &m, &[a.clone(), vec![9.0, 9.0]])
        .unwrap();
    assert_eq!(r, a, "chimera-transplant {:?}", r);

    // chimera router trains to the labeled parent
    let mut router = ChimeraRouter::new(2);
    let compat = vec![0.9, 0.1];
    for _ in 0..200 {
        router.train_step(&compat, 1, 0.5);
    }
    assert_eq!(router.pick(&compat), 1, "router {:?}", router.logits);

    // pocket: diverse keep over a tiny expert set
    let experts = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![1.01, 0.01, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    let scores = score_experts(&experts);
    let kept = greedy_diverse_select(&experts, &scores, 2);
    assert_eq!(kept.len(), 2, "pocket {:?}", kept);
    assert!(!(kept.contains(&0) && kept.contains(&1)), "pocket-diversity {:?}", kept);

    // aether: 49-layer remap + uniform average
    let remap = AetherRemap::aether49();
    assert_eq!(remap.remap_plan().len(), 49, "aether-plan");
    let r = remap
        .merge_tensors("t", &meta(2), &[vec![1.0, 3.0], vec![3.0, 5.0]])
        .unwrap();
    assert_eq!(r, vec![2.0, 4.0], "aether-avg {:?}", r);
    let _ = b;

    println!("ALL SMOKE TESTS PASSED");
}
