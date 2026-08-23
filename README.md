# GIP-NexProQ

Data analysis code for **NexProQ**, a targeted mass spectrometry workflow for the
absolute quantitation of proteins from dried blood spots (DBS) without stable
isotope-labelled internal standards.

NexProQ multiplexes a five-point internal calibration curve and the unknown samples
into a single TMTpro-labelled acquisition, using non-adjacent reporter ion channels
and PRM-PASEF on a timsTOF HT. Every plex therefore carries its own matrix-matched
standard curve, which is what makes absolute quantitation possible without
synthesising heavy peptide standards.

This repository accompanies:

> Richard VR, Ebrahimi Ghahnavieh L, Kubiniok P, Geib T, Willetts M, Hornburg D,
> Kruppa G, Borchers CH. *Multiplexed targeted protein quantitation using isobaric
> labelling-based internal calibration and PRM-PASEF for the quantification of
> proteins from dried blood spots.*

## Contents

| Path | Description |
|---|---|
| `app_v6.py` | Current Streamlit application: builds the acquisition template, fits the per-plex calibration curves and exports the quantitation tables |
| `app_v2.py`, `app_v3.py`, `app_v4.py` | Earlier releases of the application, retained for reproducibility of previously generated results |
| `R/` | R scripts for assay characterisation and for generating the manuscript figures (see `R/README.md`) |
| `requirements.txt` | Python dependencies |

## Web application

The application is hosted and needs no local installation:

- Large-cohort version: https://gip-nexproq-largecohort.streamlit.app/
- Standard version: https://gip-nexproq.streamlit.app/

It takes the reporter ion intensities exported from the acquisition software
together with a template describing which TMT channel holds which calibration
level or sample, fits a weighted least-squares regression per peptide, and returns
absolute concentrations with the below-limit-of-quantitation values flagged.

## Running the application locally

Requires Python 3.10 or newer.

```
pip install -r requirements.txt
streamlit run app_v6.py
```

## Running the R analyses

Requires R 4.2 or newer with `tidyverse`, `patchwork` and `scales`.

The scripts read their inputs from `data/` and write figures to `plots/`, both
relative to the working directory, so run them from the root of this repository
after placing the exported quantitation tables in `data/`. `R/README.md` lists
which script produces which figure and which input files it expects.

```
Rscript R/manuscript_figures.R
```

## Data availability

The mass spectrometry proteomics data are deposited in the ProteomeXchange
Consortium via the PRIDE partner repository. Accession numbers are given in the
Data Availability statement of the manuscript.

## Licence

Distributed under the terms of the [LICENSE](LICENSE) file in this repository.
