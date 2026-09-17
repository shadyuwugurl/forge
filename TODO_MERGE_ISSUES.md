# TODO_MERGE_ISSUES

Tracked from audit 2026-09-17. Critical first.

## Critical
- [x] `forge merge --method <slerp|ties|dare|...>` all dispatched to `LinearMerge` (`crates/forge-cli/src/commands/merge.rs:create_merge_op`) — fixed to dispatch to real ops
- [x] `forge-merge/src/lib.rs` missing exports for `task_arithmetic, nuslerp, multislerp, karcher, breadcrumbs, sce, model_stock, nearswap, ram, arcee_fusion, mergekit, fusion, frankenmoe` — fixed
- [x] `forge-core::MergeMethod` missing `MultiSlerp, Karcher, Breadcrumbs{...}, BreadcrumbsTies, Sce, ArceeFusion, FrankenMoE, Fusion` variants referenced by `mergekit.rs` — fixed
- [x] Old-style ops (`SlerpMerge, TiesMerge, DareMerge, DellaMerge, FrankenMerge`) only implemented `merge_tensor` (borrowed slices), not streaming `merge_tensors` used by orchestrator — added `merge_tensors`
- [x] `forge info ./local/path` misidentified as HF repo when path contains `/` (`crates/forge-cli/src/commands/info.rs`) — fixed to check `Path::exists()` first, resolve via `resolve_model`
- [x] `forge download` / `HubClient::download` only fetched `config.json` (`crates/forge-hub/src/lib.rs`) — fixed to fetch index + shards + tokenizer + config
- [x] `resolve_model` only handled `model.safetensors`, not sharded (`model.safetensors.index.json`, `model-*.safetensors`) — fixed
- [ ] 7 models missing `model.safetensors.index.json` (need regen via `StreamingWriter::finalize` or hub download)

## Secondary
- [ ] MergeKit Qwen3.5 arch definition missing
- [ ] `forge-hub::model_info::get_model_info` returns stub Unknown/0 params — needs config.json parsing
- [ ] `forge eval` offline fallback uses ppl proxy when datasets empty — document as offline mode, not real eval
- [ ] `forge train` LoRA/GRPO skeleton — needs real loop
- [ ] `forge extract` L2/L3, `forge fuse` orchestration — partial
- [ ] `distributed.rs` mDNS + Thunderbolt ring all-reduce not implemented
- [ ] `ArchitectureMapper`: no head-count mismatch, RoPE theta/scaling adapt, vocab-size align — hetero claim unproven
- [ ] `forge-metal`, GUI/TUI crates exist but unverified on M4

## Verification
- `cargo test -p forge-merge` + `smoke.rs` exact asserts (task-arith, nearswap, slerp midpoint, RAM determinism)
- Real model-level round-trip on tiny safetensors fixture still TODO — convert `smoke.rs` to `#[test]`s
- 27B-scale memory validation outstanding
