# Training / Episode Protocol Paper Go/No-Go Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether the paper can be upgraded from a diagnostic benchmark paper into a benchmark-plus-training/episode-protocol paper by showing that support and episode construction systematically change cliff-side robustness and collapse behavior.

**Architecture:** Keep the current final relaxed release as the fixed benchmark anchor and use the intermediate coverage-extension release as the only method-development substrate. Route the paper through a protocol hierarchy: diagnosis on the final release, substrate strengthening on the intermediate release, one small inference-time support protocol family, one small episode-construction family, and only then optional training-time episodic learning if earlier layers clear their gates.

**Tech Stack:** Python 3.12, pytest, pandas/parquet, existing `fsmol_cliff` CLI/release pipeline, markdown notes in `paper_latex/notes`, release artifacts under `outputs/`.

---

## File Structure

**Modify**
- `paper_latex/notes/coverage-extension-decision.md`
  Purpose: keep the current benchmark evidence boundary correct.
- `paper_latex/main.tex`
  Purpose: upgrade paper framing only if the protocol route passes.
- `src/fsmol_cliff/adapters.py`
  Purpose: host small support-side protocol variants on local classical backends.
- `src/fsmol_cliff/runner.py`
  Purpose: wire new support / episode protocol backends into release evaluation.
- `src/fsmol_cliff/cli.py`
  Purpose: expose protocol backends in a controlled way.
- `src/fsmol_cliff/manifests.py`
  Purpose: if needed, add deterministic episode-construction protocol variants.
- `src/fsmol_cliff/episodes.py`
  Purpose: if needed, add support/query-aware episode augmentation rules.
- `tests/test_baseline_adapter_runtime.py`
  Purpose: verify support-protocol rules.
- `tests/test_release_evaluation.py`
  Purpose: verify release-mode backend behavior.
- `tests/test_cli_commands.py`
  Purpose: verify CLI exposure of protocol variants.
- `tests/test_episodes.py`
  Purpose: verify any new episode-construction protocol.

**Create**
- `docs/superpowers/plans/2026-03-24-training-episode-protocol-go-no-go-plan.md`
  Purpose: this plan.
- `paper_latex/notes/training-episode-protocol-status.md`
  Purpose: live go/no-go status record.
- `paper_latex/notes/support-protocol-eval.md`
  Purpose: results for inference-time support protocol variants.
- `paper_latex/notes/episode-protocol-eval.md`
  Purpose: results for episode-construction protocol variants.
- `paper_latex/notes/protocol-checklist-summary.md`
  Purpose: unified H3-style validation summary for all protocol variants.

**Do not modify**
- `spec.md`
  Reason: this route must use the existing benchmark protocol, not redefine it.
- `outputs/fsmol_cliff_release_v4`
  Reason: keep the current final substrate fixed.

---

## Core Claim Target

This route only succeeds if the paper can support a statement like:

> Few-shot cliff robustness depends not only on model family, but also on whether support construction and episode construction expose the right local boundary information. We provide a benchmark-guided protocol for designing and validating such support/episode interventions.

This route does **not** require a new state-of-the-art backbone. It requires a protocol-level contribution with reproducible effects under a fixed validation checklist.

---

## Chunk 1: Freeze the Benchmark Anchor

### Task 1: Record the current rollback-safe identity

**Files:**
- Create: `paper_latex/notes/training-episode-protocol-status.md`

- [ ] **Step 1: Write the current benchmark baseline**

Record:
- final substrate = `outputs/fsmol_cliff_release_v4`
- intermediate substrate = `outputs/fsmol_cliff_release_v4_covext_intermediate`
- current claim anchor remains:
  - `H1`: supported trend
  - `H2`: formal claim
  - `H3`: supported trend

- [ ] **Step 2: Record the rollback-safe paper identity**

Write explicitly:
- if this route fails, the paper remains a valid stronger diagnostic benchmark paper with an appendix evidence layer

- [ ] **Step 3: Commit**

```bash
git add paper_latex/notes/training-episode-protocol-status.md
git commit -m "docs: record training protocol baseline status"
```

---

## Chunk 2: Support Protocol Layer

### Task 2: Define the smallest inference-time support protocol family

**Files:**
- Modify: `paper_latex/notes/training-episode-protocol-status.md`

- [ ] **Step 1: Pick at most two support protocol variants**

Recommended family:

- **Variant S1: Cliff-aware support selection**
  - deterministic support set augmentation or replacement toward anchor-hard-negative structure
  - no score post-processing

- **Variant S2: Support-conditioned reweighting**
  - deterministic weighting of support molecules based on cliff-side relevance
  - no encoder change

Constraint:
- these are support protocol changes, not threshold repair, not representation-learning changes

- [ ] **Step 2: Exclude scope creep**

Do not include:
- learned calibrators
- generic post-hoc threshold tuning
- backbone swaps

### Task 3: Evaluate support protocol variants on the intermediate substrate

**Files:**
- Create: `paper_latex/notes/support-protocol-eval.md`

- [ ] **Step 1: Run each support protocol variant only on the intermediate substrate**

Required baselines:
- `kNN`
- `ProtoNet` only if the support protocol can be meaningfully ported without changing the method family

- [ ] **Step 2: Report the fixed protocol checklist**

For each variant, report:
- standard `C-BAcc`
- standard `SCR`
- adversarial `C-BAcc`
- adversarial `SQ-PSR`
- adversarial `SCR`
- same-scaffold `SS-SCR`
- `NC-BAcc`
- `NC-PSR`

- [ ] **Step 3: Support protocol go/no-go**

GO only if:
- adversarial `C-BAcc` improves directionally and preferably cleanly
- `SCR` / `SS-SCR` fall directionally
- `SQ-PSR` is not obviously sacrificed
- controls do not degrade clearly

NO-GO if:
- gains appear only in one slice
- support change looks like ad-hoc data engineering
- effects vanish under same-scaffold or adversarial evaluation

---

## Chunk 3: Episode Protocol Layer

### Task 4: Define the smallest episode-construction family

**Files:**
- Modify likely: `src/fsmol_cliff/manifests.py`, `src/fsmol_cliff/episodes.py`
- Modify: `paper_latex/notes/training-episode-protocol-status.md`

- [ ] **Step 1: Pick one deterministic episode family**

Recommended candidates:
- anchor-hard-negative guided support augmentation
- adversarial support curriculum
- query-targeted support augmentation

Constraint:
- the episode rule must be deterministic and documentable
- it must be describable as protocol, not as opaque data massage

- [ ] **Step 2: Specify a fixed release tier**

Episode-protocol variants must be run as:
- `intermediate` or `exploratory`
- never overwrite the current final release

### Task 5: Evaluate episode protocol variants

**Files:**
- Create: `paper_latex/notes/episode-protocol-eval.md`

- [ ] **Step 1: Build protocol-specific manifests or release variant**

Only if needed:
- create a physically separate intermediate release tree
- do not mix with the current final release

- [ ] **Step 2: Evaluate at least one local/classical path**

Required:
- `kNN` or `kNN-cliff-aware`

Optional:
- `ProtoNet` only if the episode protocol is compatible with the runner and still interpretable

- [ ] **Step 3: Episode protocol go/no-go**

GO only if:
- adversarial cliff-side decision metrics improve
- collapse metrics improve
- control-side remains acceptable
- effect survives same-scaffold restriction

NO-GO if:
- the protocol only makes the query easier
- effects disappear under adversarial or same-scaffold slices

---

## Chunk 4: Training-Time Episodic Learning Layer

### Task 6: Only if Chunk 2 or 3 is GO, port one protocol into training

**Files:**
- Modify likely:
  - `src/fsmol_cliff/protonet_runner.py`
  - `src/fsmol_cliff/maml_legacy_runner.py` only if clearly feasible
- Test likely:
  - `tests/test_protonet_runner.py`

- [ ] **Step 1: Pick one protocol that already worked at inference/episode level**

Do not invent a new family here. Use:
- the best support protocol
  or
- the best episode protocol

- [ ] **Step 2: Add it to ProtoNet first**

Reason:
- ProtoNet is already the strongest balanced baseline
- if a protocol survives on ProtoNet, the contribution is much more credible

- [ ] **Step 3: Training-layer go/no-go**

GO if:
- the protocol still reduces collapse and improves cliff-side decisions on a learned episodic model

NO-GO if:
- the effect disappears once training is involved

---

## Chunk 5: Unified Validation Layer

### Task 7: Turn H3 into the fixed protocol validator

**Files:**
- Create: `paper_latex/notes/protocol-checklist-summary.md`

- [ ] **Step 1: For every protocol variant, fill the same checklist**

Required fields:
- `\Delta Official`
- `\Delta C-BAcc`
- `\Delta Q-PSR`
- `\Delta SQ-PSR`
- `\Delta SCR`
- `\Delta SS-SCR`
- `\Delta NC-BAcc`
- `\Delta NC-PSR`

- [ ] **Step 2: Interpret each protocol result**

Allowed verdicts:
- genuine protocol improvement
- partial improvement
- failed protocol

- [ ] **Step 3: Validation go/no-go**

GO if:
- the checklist reliably distinguishes good protocol variants from fake improvements

NO-GO if:
- the protocol cannot tell whether a “gain” is real or artifact-like

---

## Chunk 6: Paper Identity Decision

### Task 8: Final go/no-go for the protocol-paper route

**Files:**
- Modify: `paper_latex/main.tex`
- Modify: `paper_latex/notes/training-episode-protocol-status.md`

- [ ] **Step 1: Upgrade paper identity only if the loop closes**

Required:
- support or episode protocol layer = GO
- validation layer = GO
- the result is explainable as protocol, not just one-off method hacking

- [ ] **Step 2: If GO, rewrite the paper around the loop**

New paper identity:
- benchmark + training/episode protocol paper
- few-shot training protocol paper
- benchmark-guided episode design paper

- [ ] **Step 3: If NO-GO, keep benchmark identity**

Fallback:
- stronger diagnostic benchmark paper
- intermediate coverage-extension as appendix evidence
- protocol attempts reported, if at all, only as partial or failed cases

---

## Go/No-Go Summary

**GO for this route only if:**
- at least one support or episode protocol variant produces real cliff-side gains
- collapse decreases in adversarial and preferably same-scaffold slices
- control-side remains acceptable
- the H3-style checklist can certify that the gain is not fake

**NO-GO and stay benchmark-first if:**
- protocol changes behave like data engineering tricks
- gains are slice-specific and unstable
- same-scaffold or control-side checks break the story
- no protocol survives once evaluated under the full checklist
