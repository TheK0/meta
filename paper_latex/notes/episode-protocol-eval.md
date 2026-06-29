# Episode Protocol Evaluation

Date: 2026-03-24

## Thesis

This note tracks whether episode construction can support a paper framed around support-query robustness in cliff-sensitive few-shot molecular classification.

Target claim:

- episode construction is a meaningful intervention axis
- its gains must survive paired comparison gates
- beating vanilla `kNN` is necessary but not sufficient
- beating `kNN-cliff-aware` is the real promotion gate

Rollback boundary:

- if no episode variant beats `kNN-cliff-aware`, the paper remains a stronger diagnostic benchmark paper

## Fixed Evaluation Contract

Benchmark anchor:

- final substrate remains `outputs/fsmol_cliff_release_v4`

Method-development substrate:

- `outputs/fsmol_cliff_release_v4_covext_intermediate`

Primary comparators:

- vanilla intermediate `kNN`
- `kNN-cliff-aware`

Primary metrics:

- adversarial `C-BAcc`
- adversarial `SQ-PSR`
- adversarial `SCR`
- adversarial `SS-SCR`
- adversarial `NC-BAcc`
- adversarial `NC-PSR`

Promotion rule:

- article `GO` requires a clean win over `kNN-cliff-aware`
- article `NO-GO` if the variant only beats vanilla `kNN`

## Corrected Reference Variant

Current cleanest episode-construction reference:

- `query-targeted support negatives`
- corrected release:
  - `outputs/fsmol_cliff_release_v4_covext_intermediate_query_targeted_support_neg_corrected`

Implementation note:

- the earlier local artifact resampled the full adversarial episode skeleton and should not be treated as the clean reference implementation
- the corrected implementation preserves:
  - `support_pos_ids`
  - `query_pos_ids`
  - `query_neg_ids`
  - `injected_pairs`
- it rewrites only `support_neg_ids`

## Current Results

### `query-targeted support negatives`

Result versus vanilla intermediate `kNN`:

- `GO`
- adversarial `C-BAcc`: `+0.028222`, CI `[0.000352, 0.063484]`
- adversarial `SCR`: `-0.050479`, CI `[-0.087093, -0.019716]`
- adversarial `SS-SCR`: `-0.055465`, CI `[-0.099378, -0.020178]`
- adversarial `NC-BAcc`: `+0.016291`, CI `[0.000626, 0.033399]`
- adversarial `NC-PSR`: `+0.036526`, CI `[0.016899, 0.059819]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `+0.005988`, CI `[-0.007839, 0.025257]`
- adversarial `SQ-PSR`: `-0.003488`, CI `[-0.008147, 0.001516]`
- adversarial `SCR`: `+0.010864`, CI `[-0.008886, 0.030524]`
- adversarial `SS-SCR`: `+0.020048`, CI `[-0.008250, 0.053903]`

Interpretation:

- episode construction matters
- the current clean variant is still below the stronger simple cliff-aware baseline
- this route stays alive only for new episode variants, not for polishing this exact one

### `same_scaffold_query_targeted`

Result versus vanilla intermediate `kNN`:

- `NO-GO`
- adversarial `C-BAcc`: `+0.002586`, CI `[-0.050017, 0.058396]`
- adversarial `SQ-PSR`: `+0.013816`, CI `[-0.017866, 0.055753]`
- adversarial `SCR`: `+0.008893`, CI `[-0.001459, 0.021869]`
- adversarial `SS-SCR`: `+0.007664`, CI `[-0.004500, 0.022456]`
- adversarial `NC-BAcc`: `+0.000333`, CI `[-0.012437, 0.012712]`
- adversarial `NC-PSR`: `+0.006728`, CI `[-0.014708, 0.029623]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.019648`, CI `[-0.073260, 0.027052]`
- adversarial `SQ-PSR`: `+0.003491`, CI `[-0.025162, 0.035566]`
- adversarial `SCR`: `+0.070236`, CI `[0.033092, 0.110647]`
- adversarial `SS-SCR`: `+0.083176`, CI `[0.040950, 0.128621]`
- adversarial `NC-BAcc`: `-0.019472`, CI `[-0.037618, -0.000795]`
- adversarial `NC-PSR`: `-0.030691`, CI `[-0.062321, 0.001544]`

Interpretation:

- preferring same-scaffold injected cliff pairs did not produce a cleaner boundary signal
- collapse got worse rather than better
- this variant should be treated as a hard `NO-GO`

### `anchor_coverage_first`

Result versus vanilla intermediate `kNN`:

- `NO-GO`
- adversarial `C-BAcc`: `+0.008886`, CI `[-0.009426, 0.034788]`
- adversarial `SQ-PSR`: `-0.013828`, CI `[-0.032922, 0.004156]`
- adversarial `SCR`: `+0.010092`, CI `[-0.014236, 0.036220]`
- adversarial `SS-SCR`: `+0.008884`, CI `[-0.035377, 0.048836]`
- adversarial `NC-BAcc`: `+0.004356`, CI `[-0.010345, 0.023281]`
- adversarial `NC-PSR`: `+0.001136`, CI `[-0.022361, 0.025397]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.013348`, CI `[-0.035446, 0.007235]`
- adversarial `SQ-PSR`: `-0.024153`, CI `[-0.037809, -0.010700]`
- adversarial `SCR`: `+0.071435`, CI `[0.028842, 0.114614]`
- adversarial `SS-SCR`: `+0.084397`, CI `[0.018096, 0.148807]`
- adversarial `NC-BAcc`: `-0.015449`, CI `[-0.036366, 0.003388]`
- adversarial `NC-PSR`: `-0.036284`, CI `[-0.062545, -0.011332]`

Interpretation:

- prioritizing anchors with larger cliff-negative coverage did not produce a cleaner adversarial episode
- the variant sacrifices ranking and collapse behavior relative to `kNN-cliff-aware`
- this variant should also be treated as a hard `NO-GO`

### `paired_hardness_balanced`

Result versus vanilla intermediate `kNN`:

- `NO-GO`
- adversarial `C-BAcc`: `+0.018946`, CI `[-0.005416, 0.057247]`
- adversarial `SQ-PSR`: `+0.008525`, CI `[0.001800, 0.016022]`
- adversarial `SCR`: `-0.017176`, CI `[-0.031944, -0.004527]`
- adversarial `SS-SCR`: `-0.018294`, CI `[-0.042195, 0.003123]`
- adversarial `NC-BAcc`: `+0.006894`, CI `[-0.001047, 0.017423]`
- adversarial `NC-PSR`: `+0.008090`, CI `[-0.015627, 0.034295]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.003288`, CI `[-0.031345, 0.026435]`
- adversarial `SQ-PSR`: `-0.001800`, CI `[-0.016025, 0.012394]`
- adversarial `SCR`: `+0.044167`, CI `[0.000824, 0.090941]`
- adversarial `SS-SCR`: `+0.057218`, CI `[-0.000318, 0.115106]`
- adversarial `NC-BAcc`: `-0.012911`, CI `[-0.041028, 0.013004]`
- adversarial `NC-PSR`: `-0.029329`, CI `[-0.067971, 0.004881]`

Interpretation:

- balancing injected cliff-pair hardness gives a cleaner signal than `same_scaffold_query_targeted` or `anchor_coverage_first`
- but it still fails the stronger-baseline gate because collapse worsens relative to `kNN-cliff-aware`
- this is not a paper-upgrade result

### `query_cluster_separation_by_neg_diversity`

Result versus vanilla intermediate `kNN`:

- `NO-GO`
- adversarial `C-BAcc`: `+0.016485`, CI `[-0.004774, 0.051254]`
- adversarial `SQ-PSR`: `+0.009369`, CI `[0.001159, 0.018597]`
- adversarial `SCR`: `-0.009923`, CI `[-0.023472, 0.001877]`
- adversarial `SS-SCR`: `-0.009226`, CI `[-0.036427, 0.016747]`
- adversarial `NC-BAcc`: `+0.005192`, CI `[-0.002903, 0.013884]`
- adversarial `NC-PSR`: `+0.017823`, CI `[-0.004811, 0.038973]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.005749`, CI `[-0.034720, 0.023516]`
- adversarial `SQ-PSR`: `-0.000956`, CI `[-0.014353, 0.012662]`
- adversarial `SCR`: `+0.051420`, CI `[0.002789, 0.103138]`
- adversarial `SS-SCR`: `+0.066286`, CI `[0.001244, 0.133241]`
- adversarial `NC-BAcc`: `-0.014613`, CI `[-0.039649, 0.008274]`
- adversarial `NC-PSR`: `-0.019597`, CI `[-0.053618, 0.006612]`

Interpretation:

- avoiding hub negatives produces a slightly cleaner signal than the earlier same-scaffold or anchor-coverage rules
- but the stronger-baseline gate still fails because collapse worsens relative to `kNN-cliff-aware`
- this variant remains a `NO-GO`

### `query_cluster_separation_by_anchor_neg_mix`

Result versus vanilla intermediate `kNN`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.009232`, CI `[-0.027852, 0.006624]`
- adversarial `SQ-PSR`: `-0.001912`, CI `[-0.024263, 0.015919]`
- adversarial `SCR`: `+0.003177`, CI `[-0.007274, 0.014780]`
- adversarial `SS-SCR`: `+0.007910`, CI `[-0.010200, 0.030710]`
- adversarial `NC-BAcc`: `+0.008035`, CI `[-0.005064, 0.026277]`
- adversarial `NC-PSR`: `+0.010859`, CI `[-0.012236, 0.033012]`

Result versus `kNN-cliff-aware`:

- `NO-GO`
- adversarial `C-BAcc`: `-0.031466`, CI `[-0.051745, -0.013005]`
- adversarial `SQ-PSR`: `-0.012237`, CI `[-0.031531, 0.005103]`
- adversarial `SCR`: `+0.064520`, CI `[0.024419, 0.108454]`
- adversarial `SS-SCR`: `+0.083422`, CI `[0.029609, 0.143822]`
- adversarial `NC-BAcc`: `-0.011770`, CI `[-0.030932, 0.005283]`
- adversarial `NC-PSR`: `-0.026561`, CI `[-0.051750, -0.005222]`

Interpretation:

- mixing high-coverage and low-coverage anchors does not stabilize the query-side perturbation story
- this variant is worse than `query_cluster_separation_by_neg_diversity`
- this is also a hard `NO-GO`
