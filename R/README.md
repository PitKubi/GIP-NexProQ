# R analysis scripts

The R scripts used for the NexProQ assay characterisation and for generating the
manuscript figures. They read the exported TMT quantitation tables and write
figures to `plots/`.

Paths are relative to the working directory, so run the scripts from the root of
this repository with the exported quantitation tables placed in `data/`:

```
Rscript R/manuscript_figures.R
```

Expected inputs in `data/`:

| File | Used by |
|---|---|
| `wide_table_with_bloq.csv` | all CV and plex analyses, `manuscript_figures.R`, `dbs_analysis_v2.R`, `dbs_protein_vsDIA.R` |
| `validation cohort - first 10 samples - normalized_wide_table.csv` | `manuscript_figures.R`, `dbs_protein_rankplot.R` |
| `report.pr_matrix.tsv` | `dbs_protein_vsDIA.R` (DIA-NN protein matrix from the discovery run) |

## Manuscript figures

| Script | Produces |
|---|---|
| `manuscript_figures.R` | Figure 5A/B (intra- and inter-assay CV histograms), Figure 5C (pairwise day-to-day correlation grid) and Figure 6 (ranked protein concentration reference), rendered at 600 dpi as PNG and PDF |
| `dbs_protein_rankplot.R` | Ranked protein concentration plot from the validation cohort wide table |
| `fig_plex_main.R` | Plex-level summary panels, including the variance partition |
| `fig_plex_comparison.R` | Comparison of quantitation between TMT plexes |

## Assay characterisation and CV analysis

| Script | Purpose |
|---|---|
| `dbs_analysis_v2.R` | Primary analysis of the DBS wide table with below-limit-of-quantitation handling |
| `dbs_protein_vsDIA.R` | Compares NexProQ protein quantitation against the DIA-PASEF discovery results |
| `cv_diagnostics.R` | Coefficient-of-variation diagnostics across the reproducibility axes |
| `cv_diagnostics2.R` | Follow-up CV diagnostics |
| `cv_plex_analysis.R` | Within- versus across-plex CV breakdown |
| `cv_structure_analysis.R` | Decomposes CV into within-plex, intra-day and inter-day components |

## Requirements

`tidyverse`, `patchwork` and `scales`.
