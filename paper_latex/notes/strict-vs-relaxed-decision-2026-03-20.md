# Strict vs Relaxed Benchmark Decision

## Context

Using the real FS-Mol test fold in [`fs-mol`](../fs-mol), the strict current benchmark run at [`outputs/fsmol_cliff_release_run4`](../outputs/fsmol_cliff_release_run4) produced:

- `157` raw test assays
- `157` assays surviving legal-sample processing
- `151` assays surviving active/inactive minimums
- `4` assays surviving high-sim discordant support
- `3` assays surviving `C_t`
- `3` final strict benchmark assays

The dominant attrition stage is `highsim_discordant_support` (`147` assays fail there).

## Strict Decision

Keep the current `v3.0` strict setting unchanged:

- `tau = 0.85`
- `delta = 1.0`
- `min_cliff_pairs = 25`
- `min_noncliff_pairs = 10`

Interpretation:

- strict `v3.0` should be treated as a **mini benchmark**
- it is still useful for high-confidence cliff stress testing
- it is not broad enough to be the only basis for large multi-model claims

Strict result summary:

- eligible assays: `3`
- adversarial-eligible assays: `3`
- total cliff pairs: `107`
- total anchors: `72`

## Relaxed Recommendation

Prepare a follow-on relaxed version rather than mutating strict `v3.0`.

Recommended relaxed candidate:

- `tau = 0.80`
- `delta = 1.0`
- `min_cliff_pairs = 25`
- `min_noncliff_pairs = 10`

Reason:

- eligible assays increase from `3` to `8`
- adversarial-eligible assays increase from `3` to `8`
- total cliff pairs increase from `107` to `426`
- total anchors increase from `72` to `268`
- `delta = 1.0` is preserved, so the cliff definition stays scientifically strong

This is preferred over the wider candidate (`tau = 0.80`, `delta = 0.5`, `min_cliff_pairs = 10`, `min_noncliff_pairs = 5`), which reaches `18` assays but changes the cliff boundary much more aggressively.

## Decision

- **Freeze strict `v3.0` as the main mini benchmark**
- **Prototype a relaxed successor version** using the recommended candidate above
- Use strict for high-precision cliff failure analysis
- Use relaxed for broader multi-model comparison if the study requires more than `3` assays
