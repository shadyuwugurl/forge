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
    let c = vec![1.0, 1.0, 0.0, 0.0];
    let inp = vec![a.clone(), b.clone(), c.clone()];

    // task arithmetic: base=a, ft=b,c lambda=1 -> a + (b-a) + (c-a) = b+c-a
    let r = TaskArithmeticMerge::new(1.0).merge_tensors("t", &m, &inp).unwrap();
    assert_eq!(r, vec![0.0, 2.0, 0.0, 2.0], "task_arith {:?}", r);

    // multislerp single model passthrough
    let r = MultiSlerpMerge::new(vec![]).merge_tensors("t", &m, &vec![a.clone()]).unwrap();
    assert_eq!(r, a, "multislerp-1 {:?}", r);

    // multislerp uniform of identical vectors = same
    let r = MultiSlerpMerge::new(vec![]).merge_tensors("t", &m, &vec![a.clone(), a.clone()]).unwrap();
    for (x, y) in r.iter().zip(a.iter()) { assert!((x - y).abs() < 1e-5, "multislerp-ident {:?}", r); }

    // karcher of identical vectors = same (scaled correctly)
    let r = KarcherMerge::new(vec![], 20, 1e-6).merge_tensors("t", &m, &vec![a.clone(), a.clone(), a.clone()]).unwrap();
    for (x, y) in r.iter().zip(a.iter()) { assert!((x - y).abs() < 1e-3, "karcher-ident {:?}", r); }

    // nearswap: identical params blend, distant keep A
    let r = NearSwapMerge::new(0.5, 0.1).merge_tensors("t", &m, &vec![a.clone(), b.clone()]).unwrap();
    assert!((r[2] - 2.0).abs() < 1e-6, "nearswap-near {:?}", r); // both 2.0 -> blended 2.0
    assert_eq!(r[0], 1.0, "nearswap-far keeps A {:?}", r); // 1 vs 0 -> far

    // breadcrumbs with beta=gamma=0 keeps everything -> base + lambda*mean(tau)
    let r = BreadcrumbsMerge::new(1.0, 0.0, 0.0).merge_tensors("t", &m, &inp).unwrap();
    assert_eq!(r, vec![0.0, 1.0, 0.0, 0.5], "breadcrumbs-0 {:?}", r); // mean of nonzero deltas

    // arcee with threshold 0 keeps everything (same as above)
    let r = ArceeFusionMerge::new(1.0, 0.0).merge_tensors("t", &m, &inp).unwrap();
    assert_eq!(r, vec![0.5, 1.0, 1.0, 0.5], "arcee-0 {:?}", r); // full mean incl. zeros

    // sce uniform-ish: check it runs and stays finite
    let r = SceMerge::new(1.0).merge_tensors("t", &m, &inp).unwrap();
    assert!(r.iter().all(|x| x.is_finite()), "sce {:?}", r);

    // model_stock runs finite
    let r = ModelStockMerge::new().merge_tensors("t", &m, &inp).unwrap();
    assert!(r.iter().all(|x| x.is_finite()), "stock {:?}", r);

    // ram deterministic per name, convex combo bounds
    let r1 = RamMerge::new(7).merge_tensors("t", &m, &inp).unwrap();
    let r2 = RamMerge::new(7).merge_tensors("t", &m, &inp).unwrap();
    assert_eq!(r1, r2, "ram-determinism");
    for (i, x) in r1.iter().enumerate() {
        let lo = inp.iter().map(|v| v[i]).fold(f32::INFINITY, f32::min);
        let hi = inp.iter().map(|v| v[i]).fold(f32::NEG_INFINITY, f32::max);
        assert!(*x >= lo - 1e-6 && *x <= hi + 1e-6, "ram-convex {}", x);
    }

    // slerp midpoint of orthogonal unit vectors has norm ~1
    let u = vec![1.0, 0.0, 0.0, 0.0];
    let v = vec![0.0, 1.0, 0.0, 0.0];
    let r = forge_merge::slerp_utils::slerp_pair(&u, &v, 0.5).unwrap();
    let n: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((n - 1.0).abs() < 1e-5, "slerp-norm {}", n);
    assert!((r[0] - r[1]).abs() < 1e-6, "slerp-sym {:?}", r);

    println!("ALL SMOKE TESTS PASSED");
}
