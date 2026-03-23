# Coverage-Extension Decision Note

Date: 2026-03-23

Status:
- `Plan outcome A` for the intermediate coverage-extension release, with one boundary:
  - coverage-extension should be retained as an appendix / extended-table evidence-strengthening layer
  - the current final relaxed release remains the only `final` benchmark substrate
  - H3 already shows stronger evidence on the intermediate release
  - H1 full-strength reassessment remains pending completion of the intermediate `ProtoNet` run

## 1. Coverage Gate

Current final relaxed release:
- eligible assays: `6`
- adversarial-eligible assays: `6`
- total cliff pairs: `325`
- total anchors: `200`
- same-scaffold cliff pairs: `171`

Intermediate coverage-extension release (`relaxed_covext_10_10`):
- eligible assays: `10`
- adversarial-eligible assays: `10`
- total cliff pairs: `407`
- total anchors: `268`
- same-scaffold cliff pairs: `229`

Interpretation:
- coverage increased materially
- adversarial-eligible support increased in lockstep
- cliff signal was not diluted; the release gained cliff pairs, anchors, and same-scaffold coverage rather than merely adding weaker assays

## 2. H3 Signal

Compared with vanilla `kNN`, the intermediate `kNN-cliff-aware` release shows stronger paired evidence than the current final relaxed release on the most important decision/collapse metrics:

- standard `C-BAcc`: `+0.006549`, CI `[0.002650, 0.010647]`
- standard `SCR`: `-0.022791`, CI `[-0.041185, -0.005342]`
- adversarial `C-BAcc`: `+0.022234`, CI `[0.002132, 0.045002]`
- adversarial `SCR`: `-0.061343`, CI `[-0.108198, -0.021282]`

Task-direction counts on the intermediate release:
- standard `C-BAcc`: `7/10` better, `1/10` worse, `2/10` tie
- adversarial `C-BAcc`: `8/10` better, `2/10` worse
- adversarial `SCR`: `9/10` better, `1/10` worse
- adversarial `SQ-PSR`: `5/10` better, `5/10` worse

Interpretation:
- the intervention signal clearly strengthens on the intermediate substrate
- the improvement remains driven by decision/collapse behavior rather than by a clean ranking gain on adversarial `SQ-PSR`
- this is exactly the direction we wanted the coverage-extension profile to test

## 3. H2 Signal

Intermediate `kNN -> randomForest` remains consistent with the ranking-vs-decision split:

- standard `delta_auprc`: `+0.076935`, CI `[0.064948, 0.089757]`
- adversarial `SQ-PSR`: `+0.355122`, CI `[0.313384, 0.385869]`
- adversarial `C-BAcc`: `+0.012759`, CI `[-0.014416, 0.056074]`
- adversarial `SCR`: `+0.006153`, CI `[-0.026304, 0.042388]`

Task-direction counts:
- adversarial `SQ-PSR`: `10/10` better
- adversarial `C-BAcc`: `4/10` better, `5/10` worse, `1/10` tie
- adversarial `SCR`: `6/10` better, `4/10` worse

Interpretation:
- the intermediate release does not weaken the H2 story
- ranking remains uniformly stronger than decision improvement for `RF`

## 4. H1 Boundary

H1 should not be upgraded yet.

Reason:
- the intermediate release has already improved coverage and strengthened H3
- however, the full-strength intermediate model set is not complete yet because the `ProtoNet` intermediate run is still pending
- H1 is the most sensitive to full-model ordering and cliff-vs-control stability, so it should be reassessed only after the intermediate `ProtoNet` rows are available

## 5. Current Paper Decision

Recommended current paper handling:

- keep `outputs/fsmol_cliff_release_v4` as the only `final` release used for main claims
- include the intermediate coverage-extension release as an appendix / extended-table robustness layer
- explicitly state that it strengthens the H3 intervention story
- do not upgrade H1 wording yet
- re-check H1 only after the intermediate `ProtoNet` run and aggregate are complete
