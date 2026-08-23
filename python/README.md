# Analysis scripts

`fig_calibration_r2.py` produces the supplemental figure showing how calibration
curve quality is distributed across the target peptide panel.

It refits every internal calibration curve rather than reading values back from
an export, because the application does not persist per-curve R2. The fit
mirrors `calibrate_group()` in `app_v6.py`: weighted least squares of
concentration on reporter-ion intensity with w = 1/x^2, the lowest below-LOD
standard clamped to zero and any other below-LOD standard dropped, and a
weighted R2. **If the application's fit changes, this script must change with
it** or the figure will drift from the data the application reports.

Run it from the root of the repository, with the Skyline export in `data/`:

```
python3 python/fig_calibration_r2.py
```

Input: `data/2025-0825_TMT-quant-template_gene.csv`
Output: `plots/SupplementalFigure_calibration_R2.{png,pdf}` and
`plots/calibration_r2_per_curve.csv`

Requires `numpy`, `pandas` and `matplotlib`.
