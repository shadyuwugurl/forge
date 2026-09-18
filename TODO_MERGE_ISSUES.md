# TODO_MERGE_ISSUES

Audited 2026-09-18 (build green, full suite green). `[x]` fixed+verified, `[ ]` open.

## Critical — fixed
- [x] `forge merge --method <...>` all dispatched to `LinearMerge` → real dispatch
  (`merge.rs:create_merge_op`), 31/31 methods verified on sharded tiny models
  (6fb2a02)
- [x] `forge-merge/lib.rs` missing module exports; `MergeMethod` missing
  MultiSlerp/Karcher/Breadcrumbs{...}/Sce/ArceeFusion/FrankenMoE/Fusion/Nearswap{...}/Ram{...} (6fb2a02)
- [x] Old-style ops lacked streaming `merge_tensors` (6fb2a02)
- [x] `forge info` mis-ID'd local paths with `/` as HF repos → local-exists check (6fb2a02)
- [x] `HubClient::download` fetched only `config.json` → sidecars + index + all
  shards, symlink-to-cache, sharded-without-index fallback (6fb2a02, a635735)
- [x] `resolve_model` single-file only → `TensorStore::open` handles dirs via
  multi-mmap union index; merge/info/quant open dirs directly (a635735, 094b45e)
- [x] `forge quant` opened only first shard (`resolve_model(...).0`) → dir-aware (094b45e)
- [x] GGUF writer emitted placeholder raw bytes under Q8 header → real Q8_0
  block quant (ggml layout, exact sizes), F16/BF16/F32 passthrough, K-quants
  hard-error; CLI writes `<out>/model-<QTYPE>.gguf` (094b45e, round-trip tested)
- [x] `forge fuse` printed "Complete" with empty output → real streaming
  W + mean(B·A) fuse + sidecars; extract→fuse round-trip err 5e-4 (this push,
  `fuse_roundtrip_reconstructs_finetune`)
- [x] jang/dynamic3 `store.path().parent()` copied/read from wrong dir for
  dir-backed stores (jang sprayed `/tmp/*.py` into outputs) → dir-aware
  (this push)
- [x] `batch_extract` skipped bare `.safetensors` files → opens them (this push)

## Still open (honest stubs, explicitly marked in code where feasible)
- [ ] `forge train` loop is a skeleton (`train_step` returns 0.0, says "stub") —
  needs real forward/backward (Metal lora_fwd/bwd kernels exist). Tracked by
  papers backlog (Blockwise SFT 2508.19529, LoPT 2605.04913).
- [ ] `Darwin / Orca / FrankenMoE / Fusion` CLI methods fall back to linear with
  a stderr warning (no silent mismerge). Full evo/darwin engine lives in
  `forge-darwin` (MERGE^3 2502.10436, Mergenetic 2505.11427 in papers backlog).
- [ ] `GgufReader` is a stub → forge cannot merge GGUF inputs (BF16 pipeline only).
- [ ] K-quant GGUF writes (`Q4_K_*`, `Q5_*`, `Q6_*`) bail with clear error — Q8_0 only.
- [ ] `distributed.rs` mDNS + Thunderbolt ring all-reduce not implemented
  (MG-WFBP 1811.11141/1912.09268 in papers backlog).
- [ ] `ArchitectureMapper`: no head-count mismatch / RoPE adapt / vocab align.
- [ ] `forge-hub/model_info.rs`, `download.rs`, `search.rs` are uncompiled dead
  files (lib.rs doesn't declare the modules); `HubClient` in lib.rs is the real path.
- [ ] Workspace-root `tests/` belongs to no package (root Cargo.toml is
  virtual) → never compiled; stale APIs inside. Either wire to a crate or delete.
- [ ] MergeKit has no Qwen3.5 arch definition (carried over; arch_mapper covers
  generic transformer shapes).
- [ ] `forge-metal`, GUI/TUI crates unverified on M4; `forge eval` offline path
  is a ppl proxy by design (documented in eval output).

## Verification
- `cargo test --workspace` green; `smoke.rs` exact asserts pass.
- 31-method CLI matrix + 3/4-way merges + extract→fuse round-trip verified on
  tiny safetensors (this session).
- Real 9B: BF16 merge → Q8_0 quant → eval-both sweep staged (`/tmp/sweep9b.sh`),
  blocked on slow CDN downloads (see session log).
