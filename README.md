# Joint Likelihood of Local Dark-Matter VDF from Baryonic Surface Density and Vertical Kinematics

Code and figures for the red-clump (RC) chemical-plane selection and the
Cheng-style panel-wise vertical-Jeans surface-density analysis used in this
project.

## `code/`

| file | role |
|------|------|
| `chemical_plane_contours.py` | Shared module: abundance de-quantization (`dither`), smoothed/balanced density maps, Cheng thin/thick/halo region masks, and the overdensity contour + sequence-ridge ("crest") drawing used by every figure. |
| `plot_four_pooled_chemical_planes_clean_quality.py` | Source of truth for the survey-specific abundance **quality cuts** (`quality_mask`) and the four pooled single-plane chemical figures. |
| `plot_four_quality_motivated_rz_panels.py` | The 5×9 R–\|Z\| grid of chemical planes; recomputes Galactocentric R, \|Z\| and fits the per-panel L1 (fixed) / L2 (thin–thick separator) lines. Writes the combined-panel L2 table consumed downstream. |
| `panelwise_combined_cheng_surface_density.py` | Per-panel L1/L2 classification of the clean combined catalogue and the Cheng+2024 velocity-dispersion → h_σ → Σ(R,\|Z\|) surface-density run, including the per-panel folders. |
| `surface_density_cheng.py` | The Cheng+2024 Jeans machinery (velocity-dispersion profiles, h_σ fit, Σ/K_z), imported unchanged by the pipeline above. |

### L2 (thin/thick) policy
Each combined R–|Z| panel is classified with its **directly fitted** L2 where the
two-ridge/valley detection converges, and with the **category-average fallback
L2** elsewhere (the same line drawn on the grid figure), so every panel is
classified and used. The valley fit can fail on dense panels (e.g. the solar
7<R<9 kpc bin) because the dominant survey floods the pooled histogram and
smears the thin/thick valley — a survey-imbalance effect, not missing data.
Each panel records `l2_kind` (`direct` or `fallback_category_average`).

## `figures/`

- `four_pooled_chemical_planes/` — the pooled chemical-plane figures: single-plane
  clean-quality hexbins and the R–|Z| quality-motivated panel grids (with and
  without contours).
- `panelwise_surface_density/` — per-panel chemical-selection and Σ(R)/Σ(|Z|)
  figures plus the paper-style Σ(R) / K_z / h_σ figure set.

## Not included
Large binaries are intentionally excluded: the strict point-cache
(`*.npz`, ~300 MB), the augmented FITS catalogues, and the bulky audit CSVs.
Only source code and PNG figures are tracked here.
