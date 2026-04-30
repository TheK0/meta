# Stronger-Baseline Gate Failure Patterns

Date: 2026-04-30

This document defines a reusable taxonomy for understanding why method families fail the stronger-baseline gate. The gate is applied against `kNN-cliff-aware` (minimum) or `ProtoNet` (paper-upgrade).

## Weak Directional Signal With Standard-Side Harm

**Pattern**: The method shows a numerically positive but small primary signal on adversarial cliff metrics while introducing clean negative deltas on standard-side control metrics.

**Example**: A1 (query-only local score refit). Slight adversarial improvement but standard C-BAcc or SCR moved in the wrong direction.

**Rule**: If any standard-side metric shows a clean negative delta, close the family immediately — the method is not learning cliff-sensitive calibration; it is overfitting to adversarial structure at the cost of general decision quality.

## Broad Wrong-Way Degradation

**Pattern**: The method moves the majority of primary decision metrics (C-BAcc, SCR, SS-SCR) in the wrong direction, even if one metric shows a marginal positive delta.

**Example**: B0 (coarse cliff-margin loss injection). The margin penalty parameter sweep failed to find a region where primary metrics improved coherently.

**Rule**: A single marginal positive among broad degradation does not keep a family alive. Close it and do not iterate on the same loss form (lambda + regularizer) with different hyperparameters.

## Unstable Or Non-Scaling Mechanism Signal

**Pattern**: A mechanism signal (e.g., cliff-vs-control variance gap, dropout sensitivity difference) appears in a small-scale pilot but disappears or becomes noise when scaled to full benchmark evaluation.

**Example**: Expanded C0 (support-subset dropout audit). The initial cliff-vs-control variance gap was promising in a single-task test but did not hold across the full 10-assay intermediate substrate.

**Rule**: Mechanism signals that do not scale to full task coverage are not valid training entries. Close the family and do not convert the mechanism into a training objective.

## Beats Vanilla But Not The Stronger Baseline

**Pattern**: The method shows a clean improvement over vanilla `kNN` but fails to beat `kNN-cliff-aware` on primary or safety metrics.

**Example**: Query-targeted support negatives (corrected episode construction variant). Positive deltas vs vanilla kNN on C-BAcc and SCR, but the superiority disappears when compared against kNN-cliff-aware.

**Rule**: Beating vanilla `kNN` is no longer sufficient for any method claim. A method that only beats the weakest baseline stays closed or is retained as historical evidence only, not as an active research direction.

## Policy For Closing A Family

1. **The gate is not "is there any positive number."** The gate is "is there a coherent improvement pattern without safety leakage."

2. **Near-miss results do not keep a family alive by default.** A family with a promising but non-significant primary signal and clean safety metrics may be documented as a near-miss, but remains closed unless a fundamentally different implementation approach is proposed.

3. **A family that only beats vanilla but not the stronger baseline stays closed or historical-only.** This applies even if the improvement over vanilla is statistically clean.

4. **Exhausted episode-construction exact families do not reopen under informal reframing.** Changing the narrative ("we were really testing X all along") does not reopen a closed family. A genuinely new approach requires a separate justification memo.

5. **Iterating hyperparameters within a closed family counts as extending the same family.** For example, searching different λ_cliff values in B0 is still B0 and stays closed.

6. **Audit-only work does not authorize training.** An audit finding (e.g., perturbation sensitivity gap) does not by itself justify a training intervention. Training requires a separate justification artifact.
