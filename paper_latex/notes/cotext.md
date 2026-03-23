# Project Context Log

Last updated: 2026-03-20 (Asia/Shanghai)

## Current Status

- Benchmark direction is frozen:
  - strict `v3.0` stays as a **mini benchmark**
  - relaxed primary candidate: `tau=0.80`, `delta=1.0`, `min_C=25`, `min_D=10`
- Attrition root cause is established: the dominant bottleneck is `H_t^disc` scarcity.
- Strict eligible assays from real FS-Mol test fold:
  - `CHEMBL1119333`
  - `CHEMBL1613777`
  - `CHEMBL1614027`
- Aggregated strict-mini model results now exist for:
  - `kNN`
  - `randomForest`
  - `kNN-cliff-aware`
  - `MAML` mid-run
- `kNN` and `RF` include `SQ-PSR` / `SS-SQ-PSR`.
- Legacy `MAML` load path is verified:
  - checkpoint loads in `fsmol-maml-legacy`
  - smoke `standard/adversarial` episodes run
  - legacy runner can emit parquet-compatible rows

## Completed Outputs

- Release bundle: [`outputs/fsmol_cliff_release_run4`](./outputs/fsmol_cliff_release_run4)
- Aggregated baseline results available for:
  - `kNN`
  - `randomForest`
  - `kNN-cliff-aware`
  - `MAML` mid-run aggregate
- `kNN` and `RF` now include `SQ-PSR` / `SS-SQ-PSR`.
- `RF-SQ` aggregate exists at:
  - [`task_results_rf_official_sq.aggregate.json`](./outputs/fsmol_cliff_release_run4/task_results_rf_official_sq.aggregate.json)
- `kNN-cliff-aware` aggregate exists at:
  - [`task_results_knn_cliff_aware.aggregate.json`](./outputs/fsmol_cliff_release_run4/task_results_knn_cliff_aware.aggregate.json)
- `MAML` smoke outputs exist at:
  - [`task_results_maml_smoke.parquet`](./outputs/fsmol_cliff_release_run4/task_results_maml_smoke.parquet)
  - [`task_results_maml_mid.parquet`](./outputs/fsmol_cliff_release_run4/task_results_maml_mid.parquet)
  - [`task_results_maml_mid.aggregate.json`](./outputs/fsmol_cliff_release_run4/task_results_maml_mid.aggregate.json)

## Running Work

- `full MAML` strict-mini run is still active.
- Current observed process:
  - `3 assays × 2 splits × 5 seeds × 400 episodes`
  - implemented through `evaluate_release_with_maml_legacy(...)`
  - current progress is tracked via active `maml_legacy_runner` subprocesses
  - `max_episodes=400`
  - output parquet not written yet

## Latest Verification

- `python -m pytest -q`
- Result: `98 passed, 2 warnings`

## Latest Command Log

1. `git log --oneline --decorate -6`
   - Confirmed current history up to `7107c24`.
2. `python -m pytest -q`
   - Full suite passed: `98 passed, 2 warnings`.
3. Result artifact check under `outputs/fsmol_cliff_release_run4`
   - Confirmed `kNN`, `RF`, `kNN-cliff-aware`, and `MAML mid` aggregate files exist.
4. `ps -ef | rg 'task_results_maml_full|maml_legacy_runner'`
   - Confirmed active full-MAML legacy runner process.
5. `python` file existence check for `task_results_maml_full.parquet`
   - File does not exist yet; full-MAML run has not finished.
6. `ps -ef | rg 'task_results_maml_full|maml_legacy_runner'`
   - Confirmed the full-MAML run is active and delegated through `maml_legacy_runner`.
7. `RF-SQ` aggregation completed.
   - `randomForest` now has full pair-centric strict-mini aggregates.
8. `kNN-cliff-aware` strict-mini run completed.
   - First H3 signal is now available: adversarial cliff metrics improved while collapse decreased.
9. `MAML` legacy smoke/load path verified.
   - checkpoint load passed
   - smoke episodes passed
   - parquet-compatible runner path implemented
10. Read `runner.py`, `adapters.py`, `maml_legacy_runner.py`, `evaluation.py`, `cli.py`
    - confirmed current result chain has no `ProtoNet` runner yet
    - identified reuse points for episode scoring and parquet aggregation
11. Inspected external FS-Mol `ProtoNet` code and checkpoint inventory
    - official code is under `/Volumes/macplus/project/meta/external/FS-Mol`
    - local data dir `fs-mol/` is dataset only, not source code
    - `PN-Support64_best_validation.pt` is present in `checkpoints/`
12. Read `spec.md` support-side scoring clause
    - `SQ-PSR` / `SS-SQ-PSR` require explicit support-side scoring
    - planned implementation uses `support forward pass` for ProtoNet support scores
13. Added and ran ProtoNet TDD tests
    - initial failure: `evaluate_release_with_protonet` was missing from `fsmol_cliff.runner`
    - scope locked to runner rows, adversarial `SQ-PSR`, and CLI dispatch
14. Inspected and aligned existing `src/fsmol_cliff/protonet_runner.py`
    - found official ProtoNet runner was already partially implemented
    - cleaned `runner.py` to use `protonet_runner.py` as the model-specific layer
15. Verification: `python -m pytest tests/test_protonet_runner.py tests/test_cli_commands.py -q`
    - passed: `15 passed, 2 warnings`
16. Real-checkpoint ProtoNet smoke on strict release
    - command used `evaluate_release_with_protonet(...)` with
      - release `outputs/fsmol_cliff_release_run4`
      - data `fs-mol`
      - checkpoint `checkpoints/PN-Support64_best_validation.pt`
      - task `CHEMBL1119333`, seed `0`, `max_episodes=1`
    - succeeded and wrote `/tmp/protonet_smoke.parquet`
    - smoke rows included `sq_psr` and `ss_sq_psr`
17. Full regression: `python -m pytest -q`
    - passed: `102 passed, 2 warnings`
    - ProtoNet integration did not break existing release / CLI / MAML paths
13. ProtoNet checkpoint load smoke in main env
    - official imports work
    - checkpoint load fails via upstream helper because PyTorch 2.6 defaults `weights_only=True`
    - root cause confirmed by direct `torch.load(..., weights_only=False)` success
14. Added `tests/test_protonet_runner.py` and ran it
    - expected RED state reached: `ModuleNotFoundError: fsmol_cliff.protonet_runner`
    - next step is minimal runner implementation plus CLI dispatch
15. Implemented `src/fsmol_cliff/protonet_runner.py`
    - adds legacy-safe checkpoint loading via `torch.load(..., weights_only=False)`
    - maps release molecule ids back to `FSMolTask` samples
    - emits combined support/query score maps for `SQ-PSR` / `SS-SQ-PSR`
16. Fixed runner/CLI integration for ProtoNet
    - aligned `runner.py` wrapper aliases with new `protonet_runner`
    - added CLI regression test for `--backend protonet`
17. Verification
    - `PYTHONPATH=src python -m pytest tests/test_protonet_runner.py tests/test_cli_commands.py -q`
    - result: `15 passed, 2 warnings`
18. Real-data ProtoNet smoke
    - direct Python smoke on `outputs/fsmol_cliff_release_run4` succeeded
    - CLI smoke with `--backend protonet` succeeded and wrote `/tmp/protonet_cli_smoke.parquet`
    - adversarial `sq_psr` / `ss_sq_psr` were non-NA, confirming support-side scores are in the metric chain
19. Full regression suite
    - `PYTHONPATH=src python -m pytest -q`
    - result: `102 passed, 2 warnings`
    - ProtoNet integration did not break existing benchmark, release, MAML, or hypothesis tests
13. Added ProtoNet TDD coverage and ran focused pytest
    - red state confirmed: `evaluate_release_with_protonet` does not exist yet
    - CLI protonet backend is also not wired
14. Implemented ProtoNet release evaluation path
    - wired `evaluate_release_with_protonet(...)` into runner and CLI
    - aligned support-side scoring to `support forward pass`
15. Focused verification
    - `python -m pytest tests/test_protonet_runner.py tests/test_cli_commands.py -q`
    - result: `15 passed, 2 warnings`
16. Real ProtoNet Python API smoke
    - `PYTHONPATH=src python -c ... evaluate_release_with_protonet(...)`
    - strict release + official `PN-Support64_best_validation.pt` succeeded
    - wrote `/tmp/protonet_smoke.parquet`
17. Real ProtoNet CLI smoke
    - `PYTHONPATH=src python -m fsmol_cliff.cli evaluate --backend protonet --max-episodes 1 ...`
    - succeeded and wrote `/tmp/protonet_cli.parquet`
18. Full suite verification
    - `PYTHONPATH=src python -m pytest -q`
    - result: `102 passed, 2 warnings`
19. Inspected real ProtoNet smoke parquet
    - adversarial smoke emitted non-NA `sq_psr` and `ss_sq_psr`
    - confirms support-side scoring enters the unified metric table
20. Started full ProtoNet strict-mini benchmark run
    - command:
      `PYTHONPATH=src python -m fsmol_cliff.cli evaluate --release-dir outputs/fsmol_cliff_release_run4 --data-dir fs-mol --checkpoint checkpoints/PN-Support64_best_validation.pt --output outputs/fsmol_cliff_release_run4/task_results_protonet.parquet --backend protonet`
    - status: running
    - target artifact: `outputs/fsmol_cliff_release_run4/task_results_protonet.parquet`
21. Polled ProtoNet full run twice after launch
    - process remains alive
    - observed only pyarrow CPU-info warnings, no model/runtime exception so far
22. Checked ProtoNet full run status
    - full command process is still alive
    - target parquet is not written yet
    - no new stderr beyond prior startup warnings
23. Estimated ProtoNet full-run duration
    - manifest size confirmed: `6000` standard + `6000` adversarial episodes
    - measured small-batch throughput: `10` episodes in about `9.228s`
    - rough compute estimate: about `3.1h` total for all `12000` episodes, before minor overhead
24. ProtoNet full strict-mini run completed
    - output written: `outputs/fsmol_cliff_release_run4/task_results_protonet.parquet`
    - aggregate written: `outputs/fsmol_cliff_release_run4/task_results_protonet.aggregate.json`
    - full row count: `330` task-level metric rows
25. ProtoNet headline results
    - standard: `delta_auprc≈0.192`, `c_bacc≈0.530`, `q_psr≈0.668`, `scr≈0.883`
    - adversarial: `delta_auprc≈-0.015`, `c_bacc≈0.502`, `sq_psr≈0.664`, `scr≈0.916`, `ss_sq_psr≈0.652`
    - interpretation: ranking-layer表现强，但 decision-layer collapse 仍然明显
26. Built unified strict-mini comparison table
    - files:
      `outputs/fsmol_cliff_release_run4/strict_mini_main_table.csv`
      `outputs/fsmol_cliff_release_run4/strict_mini_main_table.md`
    - included models:
      `kNN`, `RF`, `MAML`, `ProtoNet`, `kNN-cliff-aware`
    - included metrics:
      standard `AP / delta_auprc / c_bacc / q_psr / scr / ss_scr / ss_q_psr`
      adversarial `AP / delta_auprc / c_bacc / sq_psr / scr / ss_scr / ss_sq_psr`
27. Wrote report-style strict-mini conclusion note
    - file:
      `outputs/fsmol_cliff_release_run4/strict_mini_conclusion.md`
    - covers:
      benchmark status, headline findings, H1/H2/H3 interpretation boundaries, ProtoNet runtime note
28. Started v4.0 completion implementation
    - verified current baseline state with `git status --short` and `python -m pytest -q`
    - baseline before new changes: `102 passed, 2 warnings`
29. Added TDD coverage for v4.0 blockers
    - failing tests added for:
      profile-aware release naming
      v4.0 manifest defaults
      task/aggregate result `profile` + `result_tier`
30. Implemented release/profile-aware schema upgrade
    - added strict/relaxed profile definitions
    - upgraded manifest default to `v4.0`
    - release builder now writes `episodes_*_{profile}.parquet`, `fsmol_cliff_{profile}_*.json`
    - assay assets now support `pairs_{profile}.jsonl` and `anchor_to_hardnegs_{profile}.json`
    - task-level and aggregate result rows now include `profile` and `result_tier`
31. Verification after release/schema upgrade
    - targeted: `python -m pytest tests/test_bootstrap.py tests/test_release.py tests/test_release_evaluation.py -q`
    - result: `12 passed, 1 warning`
    - full suite: `python -m pytest -q`
    - result: `104 passed, 2 warnings`
32. Added profile-aware audit support
    - `write_attrition_audit(...)` now accepts `profile`
    - output renamed from `threshold_sweep.parquet` to `threshold_sensitivity.parquet`
    - verification:
      `python -m pytest tests/test_attrition_audit.py -q`
      result: `3 passed`
33. Full suite after audit update
    - `python -m pytest -q`
    - result: `105 passed, 2 warnings`
34. Built new v4.0 release directory
    - strict build:
      `PYTHONPATH=src python -m fsmol_cliff.cli build-release --data-dir fs-mol --output-dir outputs/fsmol_cliff_release_v4 --profile strict --fsmol-data-version fs-mol-local`
    - relaxed build:
      `PYTHONPATH=src python -m fsmol_cliff.cli build-release --data-dir fs-mol --output-dir outputs/fsmol_cliff_release_v4 --profile relaxed --fsmol-data-version fs-mol-local`
    - manifest now shows built profiles: `['relaxed', 'strict']`
35. Relaxed coverage confirmed
    - `fsmol_cliff_relaxed_all.json`: `8` assays
    - `fsmol_cliff_relaxed_adv_eligible.json`: `8` assays
    - relaxed manifests: `16000` standard + `16000` adversarial episodes
36. Built strict/relaxed attrition assets
    - outputs:
      `outputs/fsmol_cliff_release_v4/audit/strict/*`
      `outputs/fsmol_cliff_release_v4/audit/relaxed/*`
    - key funnel result:
      strict `157 -> 3`
      relaxed `157 -> 8`
37. Wrote formal release decision note
    - `outputs/fsmol_cliff_release_v4/benchmark_decision_note.md`
38. Relaxed model runs started
    - running:
      `kNN`, `RF`, `kNN-cliff-aware`, `ProtoNet`
    - `MAML` first failed due legacy runner still reading old manifest names
39. MAML profile-aware bug fixed
    - root cause: `maml_legacy_runner.py` still hardcoded `episodes_{split}.parquet`
    - fix: added `--profile` plumbing and profile-aware manifest resolution
    - verification:
      `python -m pytest tests/test_maml_legacy_runner.py -q`
      result: `5 passed`
40. Relaxed MAML run restarted
    - output target:
      `outputs/fsmol_cliff_release_v4/task_results_maml_relaxed.parquet`
41. Wrote support-valid/model execution metadata
    - file:
      `outputs/fsmol_cliff_release_v4/model_execution_metadata.json`
42. Migrated strict model results into v4 release directory
    - wrote:
      `task_results_knn_strict.parquet`
      `task_results_rf_strict.parquet`
      `task_results_maml_strict.parquet`
      `task_results_protonet_strict.parquet`
      `task_results_knn_cliff_aware_strict.parquet`
    - all strict aggregates regenerated with `profile=strict`, `result_tier=final`
43. Relaxed partial results available
    - completed:
      `task_results_knn_relaxed.parquet`
      `task_results_knn_cliff_aware_relaxed.parquet`
    - aggregates written for both
    - early relaxed signal:
      `kNN` adversarial `scr≈0.929`
      `kNN-cliff-aware` adversarial `scr≈0.900`
44. Verification after MAML/profile fix and release migration
    - `python -m pytest -q`
    - result: `106 passed, 2 warnings`
45. Relaxed benchmark execution status
    - completed:
      `task_results_knn_relaxed.parquet`
      `task_results_knn_cliff_aware_relaxed.parquet`
      plus their aggregate JSON files
    - still running:
      `RF`, `ProtoNet`, `MAML`
    - relaxed early metrics:
      `kNN` standard `q_psr≈0.512`, adversarial `sq_psr≈0.541`, adversarial `scr≈0.929`
      `kNN-cliff-aware` standard `q_psr≈0.528`, adversarial `sq_psr≈0.561`, adversarial `scr≈0.900`
46. Long-run status check at 2026-03-20 22:19 CST
    - still running:
      `RF` relaxed
      `ProtoNet` relaxed
      `MAML` relaxed
    - no relaxed result parquet yet for these three models
47. MAML relaxed current chunk
    - current worker:
      `task_id=CHEMBL1614027`, `profile=relaxed`, `split=standard`, `seed=3`
    - relaxed MAML total chunk count:
      `8 tasks × 5 seeds × 2 splits = 80`
    - approximate progress by chunk order:
      about `16.9%`
48. Status check at 2026-03-20 22:38 CST
    - `RF` relaxed finished and aggregate written
    - `ProtoNet` relaxed still running
    - `MAML` relaxed still running
49. Latest relaxed progress snapshot
    - `RF` relaxed headline:
      standard `q_psr≈0.581`, standard `c_bacc≈0.509`
      adversarial `sq_psr≈0.868`, adversarial `c_bacc≈0.498`, adversarial `scr≈0.932`
    - `MAML` relaxed current chunk moved to:
      `task_id=CHEMBL3888181`, `split=standard`, `seed=2`
    - approximate relaxed MAML progress by chunk order:
      about `34.4%`
50. Status check at 2026-03-20 23:10 CST
    - `ProtoNet` relaxed still running
      runtime now about `99m`
    - `MAML` relaxed still running
      current chunk:
      `task_id=CHEMBL1614027`, `split=adversarial`, `seed=1`
    - relaxed MAML approximate progress by chunk order:
      about `64.4%`
    - no result parquet yet for `ProtoNet` or `MAML`
51. Status check at 2026-03-20 23:56 CST
    - `MAML` relaxed is now complete
      `task_results_maml_relaxed.parquet` exists with `880` rows
      (`8 tasks × 5 seeds × 2 splits × 11 metrics`)
    - `ProtoNet` relaxed is still running
      started around `21:56 CST`
      no result parquet yet
52. Status check at 2026-03-21 00:06 CST
    - `ProtoNet` relaxed still running
    - process CPU time about `182m`
    - output parquet still not written:
      `outputs/fsmol_cliff_release_v4/task_results_protonet_relaxed.parquet`
53. Final relaxed result completion
    - `task_results_protonet_relaxed.parquet` written at `2026-03-21 03:27 CST`
    - `task_results_maml_relaxed.parquet` and `task_results_protonet_relaxed.parquet` aggregates generated
54. Added explicit v4 migration/summary utilities
    - new tests:
      `tests/test_release_migration_compat.py`
      `tests/test_release_artifacts.py`
    - new module:
      `src/fsmol_cliff/release_artifacts.py`
55. Generated relaxed summary artifacts
    - main table:
      `relaxed_main_table.json`
      `relaxed_main_table.csv`
      `relaxed_main_table.md`
    - failure taxonomy:
      `relaxed_failure_taxonomy.json`
      `relaxed_failure_taxonomy.csv`
      `relaxed_failure_taxonomy.md`
    - paired model comparisons:
      `relaxed_model_comparisons.json`
      `relaxed_model_comparisons.csv`
      `relaxed_model_comparisons.md`
56. Final verification after artifact generation
    - `python -m pytest -q`
    - result: `112 passed, 2 warnings`
57. Final relaxed benchmark summary recorded
    - primary artifacts confirmed:
      `relaxed_main_table.{json,csv,md}`
      `relaxed_failure_taxonomy.{json,csv,md}`
      `relaxed_model_comparisons.{json,csv,md}`
    - headline interpretation:
      `ProtoNet` is the strongest vanilla baseline on relaxed
      `RF` shows strong ranking but weak decision robustness
      `MAML` remains boundary-aware but adversarially fragile
      `kNN-cliff-aware` reduces collapse relative to `kNN`
    - remaining work is now mainly release-summary / claim-summary documentation, not benchmark execution

## Update Rule

After each future command batch, append:
- command(s) run
- key result or failure
- impact on benchmark status
