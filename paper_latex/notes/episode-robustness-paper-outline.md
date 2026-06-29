# Episode-Construction / Support-Query Robustness Paper Outline

Date: 2026-03-24

## Working Title

- episodic design for cliff-sensitive few-shot molecular classification

## Core Thesis

- few-shot cliff robustness depends not only on model family
- support-query robustness is an intervention axis at the episode-construction level
- protocol variants should be judged by fixed paired gates, not by isolated score gains

## Boundary Conditions

- final benchmark anchor remains `outputs/fsmol_cliff_release_v4`
- all protocol development stays on `outputs/fsmol_cliff_release_v4_covext_intermediate`
- current evidence is positive versus vanilla `kNN`
- current evidence is not yet positive versus `kNN-cliff-aware`
- if this does not change, the paper does not upgrade

## Proposed Sections

### 1. Problem Framing

- average few-shot metrics hide cliff-side failure
- model family alone does not exhaust the design space
- episode construction is a plausible mechanism-level intervention axis

### 2. Benchmark-Guided Protocol Design

- keep the benchmark anchor fixed
- treat protocol changes as release-level episode variants
- require paired comparison against both vanilla and stronger cliff-aware baselines

### 3. Evaluation Contract

- intermediate substrate only
- paired bootstrap comparison
- primary gate is not "beats vanilla `kNN`"
- primary gate is "beats `kNN-cliff-aware` without obvious collateral damage"

### 4. Current Reference Result

- corrected `query-targeted support negatives` is the clean reference artifact
- it supports the weaker statement:
  - episode construction matters
- it does not yet support the stronger statement:
  - current protocol family beats the best simple cliff-aware baseline

### 5. Next Experimental Branch

- continue only with genuinely new episode-construction variants
- port to ProtoNet only after a stronger-baseline `GO`
- stop the article upgrade route if all new variants remain below `kNN-cliff-aware`

### 6. Failure-Stable Closeout

- if no variant clears the stronger baseline gate, retain this as future work or appendix discussion
- keep the main paper identity as stronger diagnostic benchmark paper
