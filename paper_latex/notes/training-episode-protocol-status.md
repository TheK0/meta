# Training / Episode Protocol Status

Date: 2026-03-24

## Paper Scope Lock

- working article thesis: episode construction is a meaningful intervention axis for cliff-sensitive few-shot classification, with support-query robustness as the paper-route mechanism
- current evidence state: positive versus vanilla `kNN`, not yet positive versus `kNN-cliff-aware`
- rollback identity: stronger diagnostic benchmark paper
- corrected query-targeted reference artifact: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected`
- corrected paired comparison reference: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected/paired_comparison_knn_and_cliff_aware_vs_query_targeted_support_neg_corrected.json`
- paper-route note: this corrected artifact is the clean episode-construction reference point for the paper route

## Baseline

Current rollback-safe paper identity:

- stronger diagnostic benchmark paper
- final benchmark substrate remains `outputs/fsmol_cliff_release_v4`
- intermediate coverage-extension release remains an appendix / extended-table evidence layer

Current claim anchor:

- `H1`: supported trend
- `H2`: formal claim
- `H3`: supported trend

## Why This Route Exists

This route asks whether the main lever for few-shot cliff robustness is not a stronger backbone, but a better support / episode protocol.

That means the paper would only be upgraded if we can show a reproducible loop:

1. diagnose the ranking--decision split
2. change support or episode construction in a fixed, protocol-like way
3. measure whether cliff-side decision and collapse metrics improve
4. validate the gain with the H3-style checklist

## Rollback Point

If this route fails at any point, the paper should remain:

- benchmark-first
- diagnostic in identity
- strengthened by the intermediate coverage-extension appendix

No future support / episode protocol experiment should overwrite the role of the current final relaxed benchmark.

## Current Execution State

- protocol-paper route: not yet validated
- support-protocol layer: two variants evaluated, both `NO-GO`
- episode-protocol layer: two variants evaluated
- training-time episodic layer: not started
- validation layer: protocol-side paired comparisons completed for current episode variants

## Immediate Next Step

Current default is not to upgrade the paper on this route yet.

Reason:

- no tested support / episode protocol has yet surpassed `kNN-cliff-aware`
- the rollback-safe paper identity remains stronger diagnostic benchmark paper

If experimentation continues:

- continue only with genuinely new episode-construction variants
- do not spend more time polishing the current `query-targeted support negatives` variant as if it were already paper-worthy
- do not continue the current `same_scaffold_query_targeted` rule
- do not continue the current `anchor_coverage_first` rule
- do not continue the current `paired_hardness_balanced` rule
- do not continue the current `query_cluster_separation_by_neg_diversity` rule
- do not continue the current `query_cluster_separation_by_anchor_neg_mix` rule
- do not move to training-time protocol changes unless a deterministic episode-construction rule clears the stronger baseline

## Chosen First Variant

Chosen family:

- support-protocol intervention

Chosen first variant:

- `fixed-support-hard-negative-replacement`

Definition:

- start from the existing support set
- for each support positive, deterministically choose one available hard negative from `anchor_to_hardnegs`
- keep support size fixed
- if replacement candidates exist, replace the least protocol-informative support negatives with hard negatives rather than simply appending extra negatives

Reason for choosing this first:

- it is a true support-construction protocol, not a threshold trick
- it is deterministic and easy to document
- it stays close to the existing `kNN-cliff-aware` baseline, so any change is attributable to support protocol rather than backbone or calibration changes
- it is cheap enough to evaluate first on the intermediate substrate

## Support Protocol Result

Variant tested:

- `fixed-support-hard-negative-replacement`
- backend: `support-replacement`
- substrate: `relaxed_covext_10_10`
- baseline comparator: `kNN-cliff-aware`

Result: `NO-GO`

Why:

- standard `C-BAcc` did not improve cleanly:
  - `+0.001221`, CI `[-0.004526, 0.007758]`
- adversarial `C-BAcc` did not improve:
  - `-0.001771`, CI `[-0.013826, 0.012403]`
- adversarial `SQ-PSR` got worse:
  - `-0.003784`, CI `[-0.005722, -0.001503]`
- adversarial `SCR` got worse:
  - `+0.014204`, CI `[0.000884, 0.028715]`
- same-scaffold adversarial metrics also got worse:
  - `SS-SQ-PSR`: `-0.005661`, CI `[-0.009540, -0.002194]`
  - `SS-SCR`: `+0.022139`, CI `[0.007247, 0.037068]`

Interpretation:

- fixed hard-negative replacement is too aggressive as a support protocol
- it does not preserve the ranking/collapse balance of the stronger `kNN-cliff-aware` baseline
- it should not be promoted into the paper as a successful support protocol

Current recommendation:

- do not continue this exact support-replacement rule
- if the route continues, switch to a more conservative support protocol or move to episode-construction variants

## Chosen Second Variant

Chosen family:

- support-protocol intervention

Chosen second variant:

- `partial-hard-negative-augmentation`

Definition:

- keep the original support positives and support negatives unchanged
- deterministically append only a very small number of hard negatives
- use a global cap rather than per-anchor expansion
- default target: add at most `2` hard negatives total

Reason for choosing this next:

- it is substantially more conservative than both:
  - `kNN-cliff-aware` style one-per-anchor augmentation
  - full support-negative replacement
- it preserves the original support geometry
- it still injects explicit cliff-aware signal
- it is a true support protocol change rather than a decision-rule patch

## Episode Protocol Candidate

Chosen next family:

- episode-construction intervention

Chosen first episode variant:

- `query-targeted support negatives`

Definition:

- keep the current adversarial episode skeleton
- keep support size fixed
- keep query composition fixed
- after the injected anchor/query-neg pairs are selected, preferentially choose support negatives from hard-negative candidates aligned to those injected anchors
- if not enough aligned support negatives exist, fill the rest with the current random support-negative rule

Reason for choosing this next:

- it changes support/query structure at the episode-construction layer, not the decision rule
- it is deterministic and easy to document
- it directly targets support-query perturbation robustness
- it is less likely than support replacement to destroy the entire support geometry because support size and query composition remain fixed

## Episode Protocol Result

Variant tested:

- `query-targeted support negatives`
- substrate: `relaxed_covext_10_10`
- corrected release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Implementation correction note:

- the first local implementation changed the adversarial RNG namespace and therefore resampled the full adversarial episode skeleton
- that did not match the intended protocol definition
- the corrected implementation preserves:
  - `support_pos_ids`
  - `query_pos_ids`
  - `query_neg_ids`
  - `injected_pairs`
- and rewrites only `support_neg_ids`

Result:

- versus vanilla intermediate `kNN`: `GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is still `GO` versus vanilla `kNN`:

- adversarial `C-BAcc` improved:
  - `+0.028222`, CI `[0.000352, 0.063484]`
- adversarial `SCR` improved:
  - `-0.050479`, CI `[-0.087093, -0.019716]`
- adversarial `SS-SCR` improved:
  - `-0.055465`, CI `[-0.099378, -0.020178]`
- control-side non-cliff metrics also moved in the right direction:
  - `NC-BAcc`: `+0.016291`, CI `[0.000626, 0.033399]`
  - `NC-PSR`: `+0.036526`, CI `[0.016899, 0.059819]`
- standard metrics stayed unchanged because the corrected variant does not touch the standard manifests

Why it is still `NO-GO` versus `kNN-cliff-aware`:

- adversarial `C-BAcc` did not improve cleanly:
  - `+0.005988`, CI `[-0.007839, 0.025257]`
- adversarial `SCR` did not improve:
  - `+0.010864`, CI `[-0.008886, 0.030524]`
- adversarial `SQ-PSR` did not improve:
  - `-0.003488`, CI `[-0.008147, 0.001516]`
- same-scaffold collapse also did not improve:
  - `SS-SCR`: `+0.020048`, CI `[-0.008250, 0.053903]`

Interpretation:

- this remains the cleanest positive signal so far at the protocol level against vanilla `kNN`
- but it still does not justify upgrading the paper into a benchmark-plus-protocol paper
- the correct summary remains:
  - episode construction matters
  - current variants still do not beat the strongest simple cliff-aware baseline

## Second Episode Protocol Result

Variant tested:

- `same_scaffold_query_targeted`
- substrate: `relaxed_covext_10_10`
- release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_same_scaffold_query_targeted`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Result:

- versus vanilla intermediate `kNN`: `NO-GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is `NO-GO` even versus vanilla intermediate `kNN`:

- adversarial `C-BAcc` did not improve cleanly:
  - `+0.002586`, CI `[-0.050017, 0.058396]`
- adversarial `SCR` moved in the wrong direction:
  - `+0.008893`, CI `[-0.001459, 0.021869]`
- adversarial `SS-SCR` also moved in the wrong direction:
  - `+0.007664`, CI `[-0.004500, 0.022456]`
- control-side metrics did not provide a compensating signal:
  - `NC-BAcc`: `+0.000333`, CI `[-0.012437, 0.012712]`
  - `NC-PSR`: `+0.006728`, CI `[-0.014708, 0.029623]`

Why it is a hard `NO-GO` versus `kNN-cliff-aware`:

- adversarial `C-BAcc` regressed:
  - `-0.019648`, CI `[-0.073260, 0.027052]`
- adversarial `SCR` got materially worse:
  - `+0.070236`, CI `[0.033092, 0.110647]`
- same-scaffold collapse also got materially worse:
  - `SS-SCR`: `+0.083176`, CI `[0.040950, 0.128621]`
- non-cliff control accuracy also regressed:
  - `NC-BAcc`: `-0.019472`, CI `[-0.037618, -0.000795]`

Interpretation:

- simply forcing same-scaffold injected cliff pairs is not the right support-query robustness rule
- this variant should not be ported or discussed as a near-miss
- the route remains alive only because the corrected `query-targeted support negatives` variant still showed a cleaner positive signal versus vanilla `kNN`

## Third Episode Protocol Result

Variant tested:

- `anchor_coverage_first`
- substrate: `relaxed_covext_10_10`
- release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_anchor_coverage_first`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Result:

- versus vanilla intermediate `kNN`: `NO-GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is `NO-GO` even versus vanilla intermediate `kNN`:

- adversarial `C-BAcc` did not improve cleanly:
  - `+0.008886`, CI `[-0.009426, 0.034788]`
- adversarial `SQ-PSR` degraded:
  - `-0.013828`, CI `[-0.032922, 0.004156]`
- adversarial `SCR` moved in the wrong direction:
  - `+0.010092`, CI `[-0.014236, 0.036220]`
- same-scaffold collapse also moved in the wrong direction:
  - `SS-SCR`: `+0.008884`, CI `[-0.035377, 0.048836]`

Why it is a hard `NO-GO` versus `kNN-cliff-aware`:

- adversarial `C-BAcc` regressed:
  - `-0.013348`, CI `[-0.035446, 0.007235]`
- adversarial `SQ-PSR` regressed clearly:
  - `-0.024153`, CI `[-0.037809, -0.010700]`
- collapse got materially worse:
  - `SCR`: `+0.071435`, CI `[0.028842, 0.114614]`
  - `SS-SCR`: `+0.084397`, CI `[0.018096, 0.148807]`
- control-side non-cliff ranking also regressed:
  - `NC-PSR`: `-0.036284`, CI `[-0.062545, -0.011332]`

Interpretation:

- simply prioritizing anchors with larger cliff-negative coverage is not the right episode-construction rule
- this variant should not be ported or framed as a near-miss
- the route is still alive only because the corrected `query-targeted support negatives` result remains the single cleaner positive signal versus vanilla `kNN`

## Fourth Episode Protocol Result

Variant tested:

- `paired_hardness_balanced`
- substrate: `relaxed_covext_10_10`
- release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_paired_hardness_balanced`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Result:

- versus vanilla intermediate `kNN`: `NO-GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is still `NO-GO` versus vanilla intermediate `kNN`:

- there is some positive movement:
  - `SQ-PSR`: `+0.008525`, CI `[0.001800, 0.016022]`
  - `SCR`: `-0.017176`, CI `[-0.031944, -0.004527]`
- but the key adversarial `C-BAcc` gain is not clean:
  - `+0.018946`, CI `[-0.005416, 0.057247]`
- same-scaffold collapse also does not improve cleanly:
  - `SS-SCR`: `-0.018294`, CI `[-0.042195, 0.003123]`

Why it is still `NO-GO` versus `kNN-cliff-aware`:

- adversarial `C-BAcc` does not improve:
  - `-0.003288`, CI `[-0.031345, 0.026435]`
- collapse still gets worse:
  - `SCR`: `+0.044167`, CI `[0.000824, 0.090941]`
  - `SS-SCR`: `+0.057218`, CI `[-0.000318, 0.115106]`
- control-side metrics do not support a stronger interpretation:
  - `NC-BAcc`: `-0.012911`, CI `[-0.041028, 0.013004]`
  - `NC-PSR`: `-0.029329`, CI `[-0.067971, 0.004881]`

Interpretation:

- this is the closest non-winning variant so far
- but it still fails the only gate that matters for a paper upgrade
- it should be logged as another `NO-GO`, not as a hidden success

## Fifth Episode Protocol Result

Variant tested:

- `query_cluster_separation_by_neg_diversity`
- substrate: `relaxed_covext_10_10`
- release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_neg_diversity`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Result:

- versus vanilla intermediate `kNN`: `NO-GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is still `NO-GO`:

- there is some ranking-side movement versus vanilla `kNN`:
  - `SQ-PSR`: `+0.009369`, CI `[0.001159, 0.018597]`
- but adversarial `C-BAcc` is still not clean:
  - `+0.016485`, CI `[-0.004774, 0.051254]`
- and relative to `kNN-cliff-aware`, collapse gets worse:
  - `SCR`: `+0.051420`, CI `[0.002789, 0.103138]`
  - `SS-SCR`: `+0.066286`, CI `[0.001244, 0.133241]`

Interpretation:

- this is another directional but insufficient signal
- it does not change the paper-level conclusion

## Sixth Episode Protocol Result

Variant tested:

- `query_cluster_separation_by_anchor_neg_mix`
- substrate: `relaxed_covext_10_10`
- release directory: `outputs/fsmol_cliff_release_v4_covext_intermediate_query_cluster_separation_by_anchor_neg_mix`
- baseline comparators:
  - vanilla intermediate `kNN`
  - `kNN-cliff-aware`

Result:

- versus vanilla intermediate `kNN`: `NO-GO`
- versus `kNN-cliff-aware`: `NO-GO`

Why it is a hard `NO-GO`:

- adversarial `C-BAcc` already regresses versus vanilla `kNN`:
  - `-0.009232`, CI `[-0.027852, 0.006624]`
- collapse does not improve:
  - `SCR`: `+0.003177`, CI `[-0.007274, 0.014780]`
  - `SS-SCR`: `+0.007910`, CI `[-0.010200, 0.030710]`
- and relative to `kNN-cliff-aware`, the regression is clear:
  - `C-BAcc`: `-0.031466`, CI `[-0.051745, -0.013005]`
  - `SCR`: `+0.064520`, CI `[0.024419, 0.108454]`
  - `SS-SCR`: `+0.083422`, CI `[0.029609, 0.143822]`

Interpretation:

- the anchor/negative mixing rule is not a promising continuation of this route
- the sequential episode-construction sweep is now exhausted without a stronger-baseline win

## Route Conclusion

Current bottom line:

- only the corrected `query-targeted support negatives` variant showed a cleaner positive signal versus vanilla `kNN`
- no tested episode-construction variant beat `kNN-cliff-aware`
- therefore this route does not currently justify a benchmark-plus-protocol paper

Recommended paper handling:

- keep the stronger diagnostic benchmark paper identity
- treat episode-construction findings as future-work or appendix-level protocol exploration
- do not port any current episode variant to ProtoNet as if it had already passed the promotion gate
