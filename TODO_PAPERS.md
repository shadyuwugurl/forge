# TODO_PAPERS — Model-Merging / Compression Paper Implementation Roadmap

Source list: 34 unique papers (42 URLs, deduped). Resolved 2026-09-18.
Status legend: `[ ]` todo · `[~]` in README R&D roadmap already · `[x]` done in forge.

Conventions (match this repo):
- New merge ops implement `MergeOp::merge_tensors(&self, name, meta, inputs)` in
  `crates/forge-merge/src/<name>.rs`, exported via `lib.rs`, wired through
  `forge-core::MergeMethod` (+ serde defaults), `mergekit.rs` parse/to_string,
  and `forge-cli` `create_merge_op` dispatch. Streaming rule: one tensor at a
  time, never whole models.
- Every op gets exact-value `smoke.rs` asserts + a tiny-safetensors round-trip
  (`linear`-style harness in `tests/`), determinism check where seeded.
- Name collisions with existing methods must be resolved explicitly (see RAM).

## A. Core merge operators (`forge-merge`)

- [ ] **DRM — Decom-Renorm-Merge** (`2505.23117`)
  `forge-merge/src/drm.rs`. SVD each task delta, merge entry-wise in the aligned
  joint space, renormalize (paper: renorm is the crucial step). Works for full
  FT + LoRA. Local artifacts mirror: `hf://buckets/MC7ever/decom-renorm-merge-artifacts`.
  Effort: M.
- [ ] **DC-Merge — Directional Consistency** (`2603.06242`)
  `forge-merge/src/dc_merge.rs`. Smooth singular values (energy balance) →
  project task vectors onto shared orthogonal subspace → aggregate → project
  back. Ref: https://github.com/Tobeginwith/DC-Merge. Effort: M.
- [ ] **NeuroMerging — neuronal subspace disentangling** (`2503.05320v4`)
  `forge-merge/src/neuro_merge.rs`. Decompose task reps into input-sensitivity
  vs task-adaptability subspaces; mitigate interference per subspace. Effort: L.
- [ ] **R-TSVM — reweighting task singular-vector merging** (`2502.06876`)
  `forge-merge/src/r_tsvm.rs`. Outlier-aware parameter weighting +
  sparsity-adaptive rank selection (heavy-tailed LLM params). Flag: paper is
  mainly a 3H-alignment benchmark — implement R-TSVM op + adopt its
  conflict-resolution findings into `ties`/`dare` docs. Effort: M.
- [ ] **DO-Merging — decouple + orthogonalize (LoRA)** (`2505.15875`)
  `forge-merge/src/do_merge.rs`. Split magnitude/direction, merge independently;
  data-free layer-wise GD with orthogonal constraints on directions. Near
  free-lunch booster for existing methods. Effort: M.
- [ ] **Extra-Merge — rank-1 subspace extrapolation** (`2605.26484`)
  `forge-merge/src/extra_merge.rs`. Fit the ~1-D manifold over consecutive
  merged checkpoints, extrapolate (no grad updates). Needs ≥2 staged merges as
  input — orchestrator support. Effort: M.
- [ ] **CtM — Compress-then-Merge for LoRAs** (`2606.03723`)
  `forge-train` + `forge-merge/src/ctm.rs`. Shared rank-r subspace from LoRA
  factors only → per-adapter r×r coords → merge in reduced space (rank-r by
  construction, no post-hoc SVD truncation). Pairs with existing
  merge-then-compress path. Effort: L.
- [ ] **MC-SMoE — Merge, Then Compress SMoE** (`2310.01334`)
  `forge-merge/src/mc_smoe.rs`. Neuron-permutation alignment → routing-frequency
  weighted expert grouping → merge → low-rank/sparse compress. Local artifacts
  mirror: `hf://buckets/MC7ever/merge-then-compress-artifacts`. Effort: L.
- [ ] **MergeME — homo/hetero MoE merging** (`2502.00997`)
  Extend `frankenmoe.rs` / `moe_bridge.rs`: interference-mitigated expert
  averaging, routing heuristics (less post-merge FT), cross-arch expert merge.
  Effort: L.
- [ ] **KVMerger — adaptive KV-cache merging** (`2407.08454`)
  Inference-time track (`forge-metal` / llama backend): token-similarity merging
  sets + Gaussian-kernel weighted merge under cache budgets. NOT a weight op —
  keep out of `MergeMethod`. Local artifacts mirror:
  `hf://buckets/MC7ever/model-tells-you-where-to-merge-artifacts`. Effort: M.
- [ ] **TIME — temporal merging framework** (`2412.06712`)
  Orchestrator support for sequential merges: init-from-merged vs init-from-base
  axes, per-step technique choice (`forge merge --temporal plan.yaml`).
  Local artifacts mirror:
  `hf://buckets/MC7ever/how-to-merge-your-multimodal-models-over-time-artifacts`.
  Effort: M.
- [ ] **MERGE^3 — IRT fitness estimators** (`2502.10436`)
  `forge-darwin`: reduced-set extraction + IRT ability estimation as cheap
  fitness proxy (claims 50× eval-cost cut). Ref: https://github.com/tommasomncttn/merge3.
  Effort: M.
- [ ] **Mergenetic — evo-merge library API** (`2505.11427`)
  API-design reference for `forge-darwin`: composable (method × evo-algo) +
  lightweight fitness estimators. Ref: https://github.com/tommasomncttn/mergenetic.
  Effort: S (design alignment, not a new op).
- [ ] **Mergeability predictor** (`2601.22285`)
  `forge-cli`: `forge merge --dry-run` advisor scoring gradient-alignment /
  pairwise compatibility metrics per method before burning GPU hours. Effort: M.
- [ ] **Agent-RAM — RL-agent merging** (`2601.13572`)
  `forge-merge/src/agent_ram.rs`. Shared vs unique update disentangling for
  sparse heterogeneous RL task vectors. ⚠️ COLLISION: forge already has `Ram`
  (randomized averaging) — new variant MUST be named `AgentRam`
  (`MergeMethod::AgentRam`, CLI `--method agent_ram`). Effort: M.
- [ ] **SWB-DM — robust trimmed-barycenter aggregation** (`2609.16099`)
  `forge-merge/src/distributed.rs` trust layer: sliced trimmed-Wasserstein
  barycenter + trim calibration; adjacent to darwin MRI-Trust. Effort: L.
- [ ] **MG-WFBP — merged-gradient comms** (`1811.11141` + `1912.09268`, same line)
  `forge-merge/src/distributed.rs` ring all-reduce: batch short per-layer comms
  into single transfers (the unimplemented mDNS/Thunderbolt path). One item,
  two versions (conference → journal). Effort: M.
- [ ] **Branch-Train-Merge workflow** (`2208.03306`)
  Docs + pipeline: branch domain experts → train → merge back (`docs/btm.md`,
  `forge fuse` recipe). No new op. Effort: S.
- [ ] **Disperse-Then-Merge SFT pipeline** (`2405.13432`)
  `forge-train` recipe: shard SFT data → train sub-models → merge (alignment-tax
  reduction). Effort: S.
- [ ] **Objective/language-based merging methodology** (`2410.10801`)
  Eval methodology input (objective-first, then language-grouped merges);
  feed into benchmax suites. No new op. Effort: S.

## B. Eval harness (`forge-eval`)

- [ ] **BSM — Branch-Solve-Merge for eval/generation** (`2310.15123`)
  Decompose → parallel solve → fuse for response eval + constrained generation;
  human-agreement + bias-reduction path for benchmax human CSVs. Effort: M.
- [ ] **PORTIA — position-bias alignment** (`2310.01432`)
  Split-align-merge prompts for pairwise LLM judging; consistency fix for
  benchmax judge comparisons. Effort: S.

## C. Quant / post-merge compress (`forge-quant`)

- [~] **E-PMQ — expert-guided post-merge quantization** (`2605.16882`)
  Already in README R&D roadmap (`forge-quant` + `forge-train`). This list
  confirms priority: expert-guided targets + merged-weight anchoring after
  `forge merge`. Ref: https://github.com/wwjzhy/E-PMQ.

## D. Reference / background (no code)

- [ ] **MergeKit paper** (`2403.13257`) — parity-matrix reference for
  `docs/mergekit_parity.md`. Ref: https://github.com/arcee-ai/MergeKit.
- [ ] **Surveys** — taxonomy alignment only: FUSE (`2603.09938`), MLLM/beyond
  (`2408.07666`, + https://github.com/EnnengYang/Awesome-Model-Merging-Methods-Theories-Applications),
  merge/ensemble/cooperate (`2407.06089`).

## E. Out of scope (recorded so nobody re-trips)

- `2508.06621` BPE merge-list-free inference — tokenizer privacy, not weight merging.
- `2107.05214` SEM table-structure — vision tables, unrelated.
- `2401.11911` generated-vs-retrieved context bias — RAG behavior, not weight merging.
- `2506.09991` Multiverse parallel decoding — inference architecture.
- `2403.06946` UniMoS CLIP UDA — vision-language adaptation; revisit if multimodal track opens.
- `2412.02601` MERGE (GNN histopathology) — name collision only, bio domain.

## Acceptance per item

1. `MergeMethod` variant + mergekit parse/serialize + CLI dispatch (no silent linear fallback).
2. `smoke.rs` exact-value asserts + tiny-safetensors round-trip test.
3. Benchmax delta entry (`forge eval` before/after) where the method is user-facing.
4. Checkbox ticked here with commit hash.
