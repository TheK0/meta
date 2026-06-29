# Protocol Checklist Summary

Date: 2026-03-24

This note records the paired-comparison gate status for protocol variants relative to both vanilla intermediate `kNN` and `kNN-cliff-aware`.

| variant | vs baseline | adv C-BAcc | adv SQ-PSR | adv SCR | adv SS-SCR | decision |
| --- | --- | --- | --- | --- | --- | --- |
| `query-targeted support negatives` | `kNN` | `+0.028222` | `+0.006837` | `-0.050479` | `-0.055465` | `GO` |
| `query-targeted support negatives` | `kNN-cliff-aware` | `+0.005988` | `-0.003488` | `+0.010864` | `+0.020048` | `NO-GO` |
| `same_scaffold_query_targeted` | `kNN` | `+0.002586` | `+0.013816` | `+0.008893` | `+0.007664` | `NO-GO` |
| `same_scaffold_query_targeted` | `kNN-cliff-aware` | `-0.019648` | `+0.003491` | `+0.070236` | `+0.083176` | `NO-GO` |
| `anchor_coverage_first` | `kNN` | `+0.008886` | `-0.013828` | `+0.010092` | `+0.008884` | `NO-GO` |
| `anchor_coverage_first` | `kNN-cliff-aware` | `-0.013348` | `-0.024153` | `+0.071435` | `+0.084397` | `NO-GO` |
| `paired_hardness_balanced` | `kNN` | `+0.018946` | `+0.008525` | `-0.017176` | `-0.018294` | `NO-GO` |
| `paired_hardness_balanced` | `kNN-cliff-aware` | `-0.003288` | `-0.001800` | `+0.044167` | `+0.057218` | `NO-GO` |
| `query_cluster_separation_by_neg_diversity` | `kNN` | `+0.016485` | `+0.009369` | `-0.009923` | `-0.009226` | `NO-GO` |
| `query_cluster_separation_by_neg_diversity` | `kNN-cliff-aware` | `-0.005749` | `-0.000956` | `+0.051420` | `+0.066286` | `NO-GO` |
| `query_cluster_separation_by_anchor_neg_mix` | `kNN` | `-0.009232` | `-0.001912` | `+0.003177` | `+0.007910` | `NO-GO` |
| `query_cluster_separation_by_anchor_neg_mix` | `kNN-cliff-aware` | `-0.031466` | `-0.012237` | `+0.064520` | `+0.083422` | `NO-GO` |

Current reading:

- the only cleanly positive protocol signal so far is still the corrected `query-targeted support negatives` variant versus vanilla `kNN`
- no tested protocol variant has yet passed the stronger `kNN-cliff-aware` gate
- the current sequential episode-construction sweep should therefore be treated as exhausted
