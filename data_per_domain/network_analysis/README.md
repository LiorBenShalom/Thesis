# Citation network — saved files

This directory contains the citation network used throughout the sentencing-range
prediction work. **All files reference the canonical Hebrew verdict IDs** (no
English IDs, no duplicates).

## Files

| file | description |
|---|---|
| `citation_edges_in_set.csv` | Directed edges A→B between the 4118 verdicts that have a sentencing range (drugs+weapon, sentencing_confidence=גבוהה). Same-domain only. 5238 edges. **This is the network used for prediction.** |
| `citation_edges_all.csv` | Directed edges A→B between any two canonical verdicts in master_inventory (across drugs/weapon/appeals/unknown). 15127 edges. Includes cross-domain edges and edges to verdicts without a range. |
| `citation_pair_types.csv` | For each of the 85,093 same-domain pairs in `similarity_scores.csv`: which relation types apply (`1hop`, `2hop`, `cocite`, or combinations) plus the GPT-4.1 similarity score. |
| `node_metrics.csv` | Per-node statistics: in-degree, out-degree, sentencing range, year, court — for in_set verdicts. |
| `summary.json` | Hub list, triadic closure stats, reciprocity. |

## Reconstruction

The network is reconstructed from:
1. `master_inventory.csv` — canonical verdicts with sentencing range
2. `innovation_submission/output/all_domains_unified.csv` — `citations_json` column listing each verdict's outbound citations
3. `innovation_submission/data_master_final/verdict_alias.csv` — old→canonical ID alias

Reconstruction script: `innovation_submission/scripts/export_citation_network.py`.
