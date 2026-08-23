# R analysis scripts

The R scripts used for the NexProQ assay characterisation and for generating the
manuscript figures. They read the exported TMT quantitation tables and write
figures to `plots/`.

Each script sets its input and output locations at the top (`proj_dir`, `data_dir`,
`csv_path`); adjust those paths to point at your own copy of the data.

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
