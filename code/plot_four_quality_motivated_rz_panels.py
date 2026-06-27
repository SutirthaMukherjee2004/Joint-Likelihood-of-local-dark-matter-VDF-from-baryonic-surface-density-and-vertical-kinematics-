#!/usr/bin/env python3
"""Four pooled chemical planes in R-|Z| panels with quality-motivated cuts.

The grey background is the original finite sample in the plotted R-|Z| panel
and abundance display window. The colored layer is the cleaned sample after
source-specific flags/S/N/error/QRF cuts plus exact sentinel/grid-edge removal.
The abundance display windows are not quality cuts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import AutoMinorLocator

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover - plotting still works without smoothing.
    gaussian_filter = None

from plot_four_pooled_chemical_planes_clean_quality import (
    CATEGORIES,
    DEFAULT_INPUT_DIR,
    Pair,
    bitmask_clear,
    err_max,
    finite_between,
    gaia_flags_good,
    int_eq,
    not_near_values,
    qrf_width_ok,
    quality_mask,
)
from chemical_plane_contours import draw_chemical_contours_and_crest, dither

# Dark-violet background (viridis zero) so empty regions are violet, not white.
VIOLET_BG = plt.cm.viridis(0.0)


DEFAULT_OUTPUT_DIR = Path(
    "/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/output_abundance_crossmatch/"
    "four_pooled_chemical_planes_basic6_kinematics"
)

R_BINS = (
    (0.0, 3.0),
    (3.0, 5.0),
    (5.0, 7.0),
    (7.0, 9.0),
    (9.0, 11.0),
    (11.0, 13.0),
    (13.0, 15.0),
    (15.0, 20.0),
    (20.0, 25.0),
)
Z_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 8.0))
COMBINED_KEY = "combined_xfe_or_alpham_vs_metallicity"
COMBINED_GROUP_ORDER = ("alpha_m", "mgfe", "alphafe", "gaia")
COMBINED_GROUP_BY_QUALITY = {
    "hattori_alpham": "alpha_m",
    "apogee_alpham": "alpha_m",
    "apogee_mgfe": "mgfe",
    "galah_mgfe": "mgfe",
    "lamost_mgfe": "mgfe",
    "ges_mgfe": "mgfe",
    "desi_sp_elem_mgfe": "mgfe",
    "desi_rvs_alphafe": "alphafe",
    "desi_sp_alphafe": "alphafe",
    "rave_alphafe": "alphafe",
    "gaia_alphafe": "gaia",
    "gaia_mgfe": "gaia",
}

THIN_THICK_LINE_M = -0.05000836526147814
THIN_THICK_LINE_B = 0.2000005330440794
HALO_LINE_M = -0.8002225074868987
HALO_LINE_B = -0.6006222012635331
GUIDE_MAGENTA = "magenta"
L2_FIT_BINS_X = 150
L2_FIT_BINS_Y = 150
L2_FIT_SIGMA = 1.2
L2_MIN_PANEL_POINTS = 300
L2_MIN_SEPARATOR_POINTS = 7

DISPLAY_LIMITS = {
    "alpha_m_vs_mh": {"xlim": (-3.0, 0.6), "ylim": (-0.20, 0.60)},
    "mgfe_vs_feh": {"xlim": (-3.5, 0.8), "ylim": (-0.50, 1.00)},
    "alphafe_vs_feh": {"xlim": (-3.5, 0.8), "ylim": (-0.35, 0.80)},
    "combined_xfe_or_alpham_vs_metallicity": {"xlim": (-3.5, 0.8), "ylim": (-0.50, 1.00)},
}

LATEX_LABELS = {
    "alpha_m_vs_mh": (r"$\mathbf{[M/H]}$", r"$\mathbf{[\alpha/M]}$"),
    "mgfe_vs_feh": (r"$\mathbf{[Fe/H]}$", r"$\mathbf{[Mg/Fe]}$"),
    "alphafe_vs_feh": (r"$\mathbf{[Fe/H]}$", r"$\mathbf{[\alpha/Fe]}$"),
    "combined_xfe_or_alpham_vs_metallicity": (
        r"$\mathbf{[Fe/H]\ or\ [M/H]}$",
        r"$\mathbf{[\alpha/M]\ or\ [\alpha/Fe]\ or\ [Mg/Fe]}$",
    ),
}

MOTION_COLUMNS = ("RV_final", "pmra_final", "pmdec_final")


@dataclass(frozen=True)
class L2Fit:
    m: float
    b: float
    fitted: bool
    n_separator_points: int
    weight_sum: float
    reason: str


def base_fits_files(input_dir: Path, include_duplicates: bool) -> list[Path]:
    paths = sorted(input_dir.glob("Entire_catalogue_chunk*_chemistry.fits"))
    if not include_duplicates:
        paths = [p for p in paths if ".1_" not in p.name]
    if not paths:
        raise FileNotFoundError(f"No augmented FITS files found in {input_dir}")
    return paths


def finite_motion_kinematics(data: fits.FITS_rec) -> np.ndarray:
    mask = np.ones(len(data), dtype=bool)
    for col in MOTION_COLUMNS:
        mask &= np.isfinite(np.asarray(data[col], dtype=np.float64))
    return mask


def pair_xy(data: fits.FITS_rec, pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data[pair.x_col], dtype=np.float64)
    if pair.y_minus_x is None:
        y = np.asarray(data[pair.y_col], dtype=np.float64)
    else:
        num, den = pair.y_minus_x
        y = np.asarray(data[num], dtype=np.float64) - np.asarray(data[den], dtype=np.float64)
    return x, y


def apogee_artifact_good(data: fits.FITS_rec) -> np.ndarray:
    aspcap_bad_bits = (19, 20, 23, 24, 27, 31, 32, 33, 34, 35, 36, 40, 41)
    star_bad_bits = (0, 3, 4, 18, 22)
    return (
        finite_between(data, "APOGEE_DR17_SNR", 50.0, np.inf)
        & bitmask_clear(data, "APOGEE_DR17_ASPCAPFLAG", aspcap_bad_bits)
        & bitmask_clear(data, "APOGEE_DR17_STARFLAG", star_bad_bits)
    )


def artifact_quality_mask(data: fits.FITS_rec, pair: Pair, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Inclusive artefact-cleaning mask for the R-|Z| panels.

    This keeps physically plausible measurements unless catalogue flags,
    exact grid/sentinel values, or extremely broad abundance posteriors
    indicate a likely measurement artefact.
    """
    key = pair.quality_key

    if key == "hattori_alpham":
        return (
            int_eq(data, "HATTORI_XP_CMD_GOOD", 1)
            & qrf_width_ok(data, "HATTORI_XP_MH_16_QRF", "HATTORI_XP_MH_84_QRF", 0.80)
            & qrf_width_ok(data, "HATTORI_XP_ALPHAM_16_QRF", "HATTORI_XP_ALPHAM_84_QRF", 0.40)
        )

    if key == "apogee_alpham":
        return apogee_artifact_good(data)

    if key == "apogee_mgfe":
        return (
            apogee_artifact_good(data)
            & err_max(data, "APOGEE_DR17_FE_H_ERR", 0.20)
            & err_max(data, "APOGEE_DR17_MG_FE_ERR", 0.20)
        )

    if key == "galah_mgfe":
        return (
            int_eq(data, "GALAH_DR4_SP_FLAG", 0)
            & int_eq(data, "GALAH_DR4_MG_FE_FLAG", 0)
            & err_max(data, "GALAH_DR4_FE_H_ERR", 0.25)
            & err_max(data, "GALAH_DR4_MG_FE_ERR", 0.25)
        )

    if key == "lamost_mgfe":
        return (
            finite_between(data, "LAMOST_DR11_MRS_SNR", 20.0, np.inf)
            & finite_between(data, "LAMOST_DR11_MRS_LOGG", 0.0, 5.5)
        )

    if key == "ges_mgfe":
        return (
            err_max(data, "GES_DR5_FE_H_ERR", 0.25)
            & err_max(data, "GES_DR5_MG1_A_ERR", 0.25)
        )

    if key == "desi_sp_elem_mgfe":
        return (
            int_eq(data, "DESI_SP_SUCCESS", 1)
            & finite_between(data, "DESI_SP_SNR_MED", 20.0, np.inf)
            & err_max(data, "DESI_SP_ELEM_FE_H_ERR", 0.30)
            & err_max(data, "DESI_SP_ELEM_MG_H_ERR", 0.30)
            & not_near_values(x, (-5.0, -4.0, -3.5, 0.5, 1.0))
        )

    if key == "desi_rvs_alphafe":
        return (
            np.asarray(data["DESI_RVS_SUCCESS"], dtype=bool)
            & np.asarray(data["DESI_RVS_PRIMARY"], dtype=bool)
            & int_eq(data, "DESI_RVS_WARN", 0)
            & err_max(data, "DESI_RVS_FEH_ERR", 0.35)
            & err_max(data, "DESI_RVS_ALPHAFE_ERR", 0.35)
            & not_near_values(x, (-4.0, 1.0))
            & not_near_values(y, (-0.4, 1.2))
        )

    if key == "desi_sp_alphafe":
        return (
            int_eq(data, "DESI_SP_SUCCESS", 1)
            & finite_between(data, "DESI_SP_SNR_MED", 20.0, np.inf)
            & not_near_values(x, (-10.0,))
            & not_near_values(y, (-0.2, 1.0))
        )

    if key == "rave_alphafe":
        return (
            err_max(data, "RAVE_DR6_FE_H_ERR", 0.35)
            & err_max(data, "RAVE_DR6_ALPHA_FE_ERR", 0.35)
            & not_near_values(y, (-0.2999,))
        )

    if key == "gaia_alphafe":
        return (
            gaia_flags_good(data)
            & qrf_width_ok(data, "GAIA_DR3_MH_GSPSPEC_LO", "GAIA_DR3_MH_GSPSPEC_HI", 0.50)
            & qrf_width_ok(data, "GAIA_DR3_ALPHAFE_GSPSPEC_LO", "GAIA_DR3_ALPHAFE_GSPSPEC_HI", 0.35)
        )

    if key == "gaia_mgfe":
        return (
            gaia_flags_good(data, extra_indices_zero_based=(15, 16))
            & qrf_width_ok(data, "GAIA_DR3_MH_GSPSPEC_LO", "GAIA_DR3_MH_GSPSPEC_HI", 0.50)
            & qrf_width_ok(data, "GAIA_DR3_MGFE_GSPSPEC_LO", "GAIA_DR3_MGFE_GSPSPEC_HI", 0.35)
            & finite_between(data, "GAIA_DR3_MGFE_GSPSPEC_NLINES", 1.0, np.inf)
            & finite_between(data, "GAIA_DR3_MGFE_GSPSPEC_LSCAT", 0.0, 0.20)
        )

    raise ValueError(f"Unknown quality key: {key}")


def galactocentric_xyz_from_final_kinematics(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_kpc: np.ndarray,
    pmra_masyr: np.ndarray,
    pmdec_masyr: np.ndarray,
    rv_kms: np.ndarray,
    r0_kpc: float,
    z_sun_kpc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = SkyCoord(
        ra=ra_deg.astype(np.float64, copy=False) * u.deg,
        dec=dec_deg.astype(np.float64, copy=False) * u.deg,
        distance=distance_kpc.astype(np.float64, copy=False) * u.kpc,
        pm_ra_cosdec=pmra_masyr.astype(np.float64, copy=False) * u.mas / u.yr,
        pm_dec=pmdec_masyr.astype(np.float64, copy=False) * u.mas / u.yr,
        radial_velocity=rv_kms.astype(np.float64, copy=False) * u.km / u.s,
        frame="icrs",
    )
    frame = Galactocentric(galcen_distance=r0_kpc * u.kpc, z_sun=z_sun_kpc * u.kpc)
    galcen = coords.transform_to(frame)
    x_gc = galcen.x.to_value(u.kpc)
    y_gc = galcen.y.to_value(u.kpc)
    z_gc = galcen.z.to_value(u.kpc)
    return x_gc, y_gc, z_gc, np.hypot(x_gc, y_gc)


def galactocentric_xyz_from_galactic_lb(
    l_deg: np.ndarray,
    b_deg: np.ndarray,
    distance_kpc: np.ndarray,
    r0_kpc: float,
    z_sun_kpc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coords = SkyCoord(
        l=l_deg.astype(np.float64, copy=False) * u.deg,
        b=b_deg.astype(np.float64, copy=False) * u.deg,
        distance=distance_kpc.astype(np.float64, copy=False) * u.kpc,
        frame="galactic",
    )
    frame = Galactocentric(galcen_distance=r0_kpc * u.kpc, z_sun=z_sun_kpc * u.kpc)
    galcen = coords.transform_to(frame)
    x_gc = galcen.x.to_value(u.kpc)
    y_gc = galcen.y.to_value(u.kpc)
    z_gc = galcen.z.to_value(u.kpc)
    return x_gc, y_gc, z_gc, np.hypot(x_gc, y_gc)


def distance_with_parallax_fallback(data: fits.FITS_rec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = np.asarray(data["distance_final"], dtype=np.float64).copy()
    final_distance = np.isfinite(distance) & (distance > 0.0)
    parallax = np.asarray(data["parallax_final"], dtype=np.float64)
    parallax_distance = (~final_distance) & np.isfinite(parallax) & (parallax > 0.0)
    distance[parallax_distance] = 1.0 / parallax[parallax_distance]
    return distance, final_distance, parallax_distance


def galactocentric_rz_with_coordinate_fallbacks(
    data: fits.FITS_rec,
    motion_mask: np.ndarray,
    r0_kpc: float,
    z_sun_kpc: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    n = len(data)
    r_gc = np.full(n, np.nan, dtype=np.float64)
    z_gc = np.full(n, np.nan, dtype=np.float64)

    distance, distance_final_mask, parallax_distance_mask = distance_with_parallax_fallback(data)
    finite_distance = np.isfinite(distance) & (distance > 0.0)
    ra = np.asarray(data["RA_final"], dtype=np.float64)
    dec = np.asarray(data["DEC_final"], dtype=np.float64)
    l_deg = np.asarray(data["l"], dtype=np.float64)
    b_deg = np.asarray(data["b"], dtype=np.float64)
    pmra = np.asarray(data["pmra_final"], dtype=np.float64)
    pmdec = np.asarray(data["pmdec_final"], dtype=np.float64)
    rv = np.asarray(data["RV_final"], dtype=np.float64)

    icrs_mask = motion_mask & finite_distance & np.isfinite(ra) & np.isfinite(dec)
    lb_mask = motion_mask & finite_distance & ~icrs_mask & np.isfinite(l_deg) & np.isfinite(b_deg)

    if np.any(icrs_mask):
        _x_use, _y_use, z_use, r_use = galactocentric_xyz_from_final_kinematics(
            ra[icrs_mask],
            dec[icrs_mask],
            distance[icrs_mask],
            pmra[icrs_mask],
            pmdec[icrs_mask],
            rv[icrs_mask],
            r0_kpc,
            z_sun_kpc,
        )
        z_gc[icrs_mask] = z_use
        r_gc[icrs_mask] = r_use

    if np.any(lb_mask):
        _x_use, _y_use, z_use, r_use = galactocentric_xyz_from_galactic_lb(
            l_deg[lb_mask],
            b_deg[lb_mask],
            distance[lb_mask],
            r0_kpc,
            z_sun_kpc,
        )
        z_gc[lb_mask] = z_use
        r_gc[lb_mask] = r_use

    stats = {
        "distance_final_positive": int((motion_mask & distance_final_mask).sum()),
        "distance_parallax_fallback": int((motion_mask & parallax_distance_mask).sum()),
        "usable_distance": int((motion_mask & finite_distance).sum()),
        "coord_icrs": int(icrs_mask.sum()),
        "coord_lb_fallback": int(lb_mask.sum()),
        "usable_position_distance": int((icrs_mask | lb_mask).sum()),
    }
    return r_gc, z_gc, stats


def empty_point_store() -> list[list[dict[str, object]]]:
    return [
        [
            {
                "raw_x": [],
                "raw_y": [],
                "clean_x": [],
                "clean_y": [],
                "group_clean_x": {name: [] for name in COMBINED_GROUP_ORDER},
                "group_clean_y": {name: [] for name in COMBINED_GROUP_ORDER},
            }
            for _ in R_BINS
        ]
        for _ in Z_BINS
    ]


def append_panel_points(
    point_store: list[list[dict[str, object]]],
    counts: np.ndarray,
    key_prefix: str,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    r_gc: np.ndarray,
    abs_z: np.ndarray,
    group_name: str | None = None,
) -> None:
    if not np.any(mask):
        return
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        zmask = mask & (abs_z >= zlo) & (abs_z < zhi)
        if not np.any(zmask):
            continue
        for ir, (rlo, rhi) in enumerate(R_BINS):
            pmask = zmask & (r_gc >= rlo) & (r_gc < rhi)
            n = int(pmask.sum())
            if n == 0:
                continue
            px = x[pmask].astype(np.float32, copy=True)
            py = y[pmask].astype(np.float32, copy=True)
            point_store[iz][ir][f"{key_prefix}_x"].append(px)
            point_store[iz][ir][f"{key_prefix}_y"].append(py)
            if group_name is not None and key_prefix == "clean":
                point_store[iz][ir]["group_clean_x"][group_name].append(px)
                point_store[iz][ir]["group_clean_y"][group_name].append(py)
            counts[iz, ir] += n


def update_category_panels(
    point_store: list[list[dict[str, object]]],
    clean_panel_counts: np.ndarray,
    raw_panel_counts: np.ndarray,
    detail_rows: list[dict[str, object]],
    cat_key: str,
    pair: Pair,
    data: fits.FITS_rec,
    base_mask: np.ndarray,
    r_gc: np.ndarray,
    abs_z: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    x, y = pair_xy(data, pair)
    finite_xy = base_mask & np.isfinite(x) & np.isfinite(y)
    displayed_raw = finite_xy & (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])

    # Strict "clean-quality" selection (same as the single-plane figures): remove
    # catalogue warning/failed solutions, exact sentinel/grid-edge synthetic values,
    # and clearly unconstrained measurements. This is the stricter, systematics-removing
    # selection, not the earlier over-relaxed one.
    qmask = quality_mask(data, pair, x, y)
    quality = finite_xy & qmask
    displayed_clean = displayed_raw & qmask

    detail_rows.append(
        {
            "category": cat_key,
            "contributing_pair": pair.label,
            "quality_key": pair.quality_key,
            "original_points_in_rz_volume": int(finite_xy.sum()),
            "original_displayed_points_in_rz_volume": int(displayed_raw.sum()),
            "quality_points_in_rz_volume": int(quality.sum()),
            "displayed_points_in_rz_volume": int(displayed_clean.sum()),
        }
    )

    append_panel_points(point_store, raw_panel_counts, "raw", x, y, displayed_raw, r_gc, abs_z)
    group_name = COMBINED_GROUP_BY_QUALITY.get(pair.quality_key) if cat_key == COMBINED_KEY else None
    append_panel_points(point_store, clean_panel_counts, "clean", x, y, displayed_clean, r_gc, abs_z, group_name)


def l1_l2_intersection_x(l2_m: float, l2_b: float) -> float:
    denom = l2_m - HALO_LINE_M
    if abs(denom) < 1e-10:
        return np.nan
    return (HALO_LINE_B - l2_b) / denom


def weighted_line_fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float] | None:
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0.0)
    if int(ok.sum()) < 2:
        return None
    x_use = x[ok].astype(np.float64, copy=False)
    y_use = y[ok].astype(np.float64, copy=False)
    w_use = weight[ok].astype(np.float64, copy=False)
    x_bar = np.average(x_use, weights=w_use)
    y_bar = np.average(y_use, weights=w_use)
    dx = x_use - x_bar
    denom = float(np.sum(w_use * dx * dx))
    if denom <= 1e-12:
        return None
    m = float(np.sum(w_use * dx * (y_use - y_bar)) / denom)
    b = float(y_bar - m * x_bar)
    return m, b


def robust_weighted_line_fit(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[float, float] | None:
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight > 0.0)
    if int(keep.sum()) < L2_MIN_SEPARATOR_POINTS:
        return None
    result = None
    for _ in range(4):
        result = weighted_line_fit(x[keep], y[keep], weight[keep])
        if result is None:
            return None
        m, b = result
        residual = y - (m * x + b)
        local = keep & np.isfinite(residual)
        if int(local.sum()) < L2_MIN_SEPARATOR_POINTS:
            break
        mad = float(np.median(np.abs(residual[local] - np.median(residual[local]))))
        clip = max(0.045, 3.0 * 1.4826 * mad)
        new_keep = local & (np.abs(residual) <= clip)
        if int(new_keep.sum()) < L2_MIN_SEPARATOR_POINTS or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return result


def fit_l2_separator(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> L2Fit:
    finite = np.isfinite(x) & np.isfinite(y)
    finite &= (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    if int(finite.sum()) < L2_MIN_PANEL_POINTS:
        return L2Fit(np.nan, np.nan, False, 0, 0.0, "too_few_clean_points")

    x_use = x[finite].astype(np.float64, copy=False)
    y_use = y[finite].astype(np.float64, copy=False)
    old_x_intersection = l1_l2_intersection_x(THIN_THICK_LINE_M, THIN_THICK_LINE_B)
    fit_x_min = xlim[0] if not np.isfinite(old_x_intersection) else max(xlim[0], old_x_intersection - 0.10)
    disk_side = x_use >= fit_x_min
    if int(disk_side.sum()) < L2_MIN_PANEL_POINTS:
        return L2Fit(np.nan, np.nan, False, 0, 0.0, "too_few_disk_side_points")
    x_use = x_use[disk_side]
    y_use = y_use[disk_side]

    hist, x_edges, y_edges = np.histogram2d(
        dither(x_use),
        dither(y_use),
        bins=(L2_FIT_BINS_X, L2_FIT_BINS_Y),
        range=[list(xlim), list(ylim)],
    )
    density = gaussian_filter(hist, sigma=L2_FIT_SIGMA, mode="nearest") if gaussian_filter is not None else hist
    positive = density[density > 0.0]
    if positive.size == 0:
        return L2Fit(np.nan, np.nan, False, 0, 0.0, "empty_density")

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    peak_floor = max(float(np.percentile(positive, 45.0)), 0.015 * float(density.max()))
    sep_min = max(0.045, 0.050 * (ylim[1] - ylim[0]))
    midpoint_x: list[float] = []
    midpoint_y: list[float] = []
    midpoint_w: list[float] = []

    for ix, xc in enumerate(x_centers):
        if xc < fit_x_min:
            continue
        col = density[ix, :]
        if float(col.max()) < peak_floor:
            continue
        split_y = THIN_THICK_LINE_M * xc + THIN_THICK_LINE_B
        lower_idx = np.flatnonzero(y_centers < split_y)
        upper_idx = np.flatnonzero(y_centers >= split_y)
        if lower_idx.size < 4 or upper_idx.size < 4:
            continue

        j_low = int(lower_idx[np.argmax(col[lower_idx])])
        j_high = int(upper_idx[np.argmax(col[upper_idx])])
        y_low = float(y_centers[j_low])
        y_high = float(y_centers[j_high])
        low_peak = float(col[j_low])
        high_peak = float(col[j_high])
        if low_peak < peak_floor or high_peak < peak_floor or y_high <= y_low:
            continue
        if (y_high - y_low) < sep_min:
            continue

        lo, hi = sorted((j_low, j_high))
        valley = float(col[lo:hi + 1].min())
        min_peak = min(low_peak, high_peak)
        if min_peak <= 0.0 or valley > 0.92 * min_peak:
            continue

        midpoint_x.append(float(xc))
        midpoint_y.append(0.5 * (y_low + y_high))
        valley_contrast = max(0.05, 1.0 - valley / min_peak)
        midpoint_w.append(float(np.sqrt(low_peak * high_peak) * valley_contrast))

    n_midpoints = len(midpoint_x)
    if n_midpoints < L2_MIN_SEPARATOR_POINTS:
        return L2Fit(np.nan, np.nan, False, n_midpoints, float(np.sum(midpoint_w)), "did_not_find_two_ridges")

    mx = np.asarray(midpoint_x, dtype=np.float64)
    my = np.asarray(midpoint_y, dtype=np.float64)
    mw = np.asarray(midpoint_w, dtype=np.float64)
    result = robust_weighted_line_fit(mx, my, mw)
    if result is None:
        return L2Fit(np.nan, np.nan, False, n_midpoints, float(mw.sum()), "line_fit_failed")

    m, b = result
    y_left = m * max(xlim[0], fit_x_min) + b
    y_right = m * xlim[1] + b
    y_pad = 0.25 * (ylim[1] - ylim[0])
    if not (-0.30 <= m <= 0.20) or not (ylim[0] - y_pad <= y_left <= ylim[1] + y_pad) or not (
        ylim[0] - y_pad <= y_right <= ylim[1] + y_pad
    ):
        return L2Fit(np.nan, np.nan, False, n_midpoints, float(mw.sum()), "unstable_line_fit")
    return L2Fit(m, b, True, n_midpoints, float(mw.sum()), "fitted")


def category_l2_fits(
    cat_key: str,
    point_store: list[list[dict[str, object]]],
    clean_panel_counts: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> tuple[list[list[L2Fit]], list[dict[str, object]]]:
    raw_fits: list[list[L2Fit]] = []
    fitted_m: list[float] = []
    fitted_b: list[float] = []
    fitted_w: list[float] = []

    for iz in range(len(Z_BINS)):
        row: list[L2Fit] = []
        for ir in range(len(R_BINS)):
            clean_x, clean_y = panel_vectors(point_store, iz, ir, "clean")
            fit = fit_l2_separator(clean_x, clean_y, xlim, ylim)
            row.append(fit)
            if fit.fitted:
                fitted_m.append(fit.m)
                fitted_b.append(fit.b)
                fitted_w.append(max(1.0, fit.weight_sum))
        raw_fits.append(row)

    if fitted_m:
        w = np.asarray(fitted_w, dtype=np.float64)
        avg_m = float(np.average(np.asarray(fitted_m, dtype=np.float64), weights=w))
        avg_b = float(np.average(np.asarray(fitted_b, dtype=np.float64), weights=w))
        fallback_reason = "category_average"
    else:
        avg_m = THIN_THICK_LINE_M
        avg_b = THIN_THICK_LINE_B
        fallback_reason = "default_cheng_l2"

    final_fits: list[list[L2Fit]] = []
    rows: list[dict[str, object]] = []
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        out_row: list[L2Fit] = []
        for ir, (rlo, rhi) in enumerate(R_BINS):
            fit = raw_fits[iz][ir]
            if fit.fitted:
                final = fit
                used_fallback = False
            else:
                final = L2Fit(avg_m, avg_b, False, fit.n_separator_points, fit.weight_sum, fallback_reason)
                used_fallback = True
            out_row.append(final)
            x_intersection = l1_l2_intersection_x(final.m, final.b)
            rows.append(
                {
                    "category": cat_key,
                    "R_min_kpc": rlo,
                    "R_max_kpc": rhi,
                    "absZ_min_kpc": zlo,
                    "absZ_max_kpc": zhi,
                    "clean_displayed_points": int(clean_panel_counts[iz, ir]),
                    "l1_m_fixed": HALO_LINE_M,
                    "l1_b_fixed": HALO_LINE_B,
                    "l2_m": final.m,
                    "l2_b": final.b,
                    "l2_panel_fit": bool(fit.fitted),
                    "l2_used_fallback": bool(used_fallback),
                    "l2_fit_reason": fit.reason if fit.fitted else f"{fit.reason}; using {fallback_reason}",
                    "l2_separator_points": int(fit.n_separator_points),
                    "l2_weight_sum": float(fit.weight_sum),
                    "l2_l1_intersection_x": x_intersection,
                    "l2_draw_x_min": max(xlim[0], x_intersection) if np.isfinite(x_intersection) else xlim[0],
                    "l2_draw_x_max": xlim[1],
                }
            )
        final_fits.append(out_row)
    return final_fits, rows


def overlay_guides(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    l2_fit: L2Fit,
) -> None:
    x = np.linspace(xlim[0], xlim[1], 500)
    l1_y = HALO_LINE_M * x + HALO_LINE_B
    l1_mask = (l1_y >= ylim[0]) & (l1_y <= ylim[1])
    ax.plot(x[l1_mask], l1_y[l1_mask], color=GUIDE_MAGENTA, lw=1.3, ls="-", alpha=0.95, zorder=4)

    l2_y = l2_fit.m * x + l2_fit.b
    x_intersection = l1_l2_intersection_x(l2_fit.m, l2_fit.b)
    draw_x_min = xlim[0] if not np.isfinite(x_intersection) else max(xlim[0], x_intersection)
    l2_mask = (x >= draw_x_min) & (l2_y >= ylim[0]) & (l2_y <= ylim[1])
    ax.plot(x[l2_mask], l2_y[l2_mask], color=GUIDE_MAGENTA, lw=1.2, ls=":", alpha=0.98, zorder=4)


def panel_vectors(
    point_store: list[list[dict[str, object]]],
    iz: int,
    ir: int,
    key_prefix: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs = point_store[iz][ir][f"{key_prefix}_x"]
    ys = point_store[iz][ir][f"{key_prefix}_y"]
    if not xs:
        empty = np.array([], dtype=np.float32)
        return empty, empty
    return np.concatenate(xs), np.concatenate(ys)


def panel_groups(
    point_store: list[list[dict[str, object]]],
    iz: int,
    ir: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    panel = point_store[iz][ir]
    out = []
    for name in COMBINED_GROUP_ORDER:
        xs = panel["group_clean_x"][name]
        ys = panel["group_clean_y"][name]
        if xs:
            out.append((np.concatenate(xs), np.concatenate(ys)))
    return out


def draw_hex_layer(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    cmap: str,
    alpha: float,
    zorder: int,
    norm: LogNorm | None = None,
):
    if len(x) == 0:
        return None
    hist, _, _ = np.histogram2d(
        dither(x),
        dither(y),
        bins=gridsize,
        range=[list(xlim), list(ylim)],
    )
    positive = hist[hist > 0]
    if positive.size == 0:
        return None
    if norm is None:
        norm = LogNorm(vmin=1.0, vmax=max(2.0, float(np.percentile(positive, 99.7))))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    return ax.imshow(
        np.ma.masked_where(hist.T <= 0, hist.T),
        origin="lower",
        extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
        aspect="auto",
        cmap=cmap_obj,
        norm=norm,
        alpha=alpha,
        interpolation="bilinear",
        rasterized=True,
        zorder=zorder,
    )


def clean_density_norm(
    point_store: list[list[dict[str, object]]],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    gridsize: int,
) -> LogNorm:
    positive_counts: list[np.ndarray] = []
    hist_bins = max(50, gridsize)
    for iz in range(len(Z_BINS)):
        for ir in range(len(R_BINS)):
            x, y = panel_vectors(point_store, iz, ir, "clean")
            if len(x) == 0:
                continue
            hist, _, _ = np.histogram2d(x, y, bins=hist_bins, range=[xlim, ylim])
            vals = hist[hist > 0]
            if vals.size:
                positive_counts.append(vals)
    if not positive_counts:
        return LogNorm(vmin=1.0, vmax=2.0)
    vals = np.concatenate(positive_counts)
    vmax = max(2.0, float(np.percentile(vals, 99.5)))
    return LogNorm(vmin=1.0, vmax=vmax)


# Contour + crest drawing now lives in the shared module
# `chemical_plane_contours.draw_chemical_contours_and_crest`, which builds
# disjoint per-region (thin / thick / halo) contours and a clipped crest.


def plot_category(
    cat,
    point_store: list[list[dict[str, object]]],
    clean_panel_counts: np.ndarray,
    raw_panel_counts: np.ndarray,
    output_dir: Path,
    gridsize: int,
    show_contours: bool = True,
) -> list[dict[str, object]]:
    limits = DISPLAY_LIMITS[cat.key]
    xlim = limits["xlim"]
    ylim = limits["ylim"]
    x_label, y_label = LATEX_LABELS[cat.key]
    clean_norm = clean_density_norm(point_store, xlim, ylim, gridsize)
    l2_fits, l2_rows = category_l2_fits(cat.key, point_store, clean_panel_counts, xlim, ylim)

    fig, axes = plt.subplots(
        len(Z_BINS),
        len(R_BINS),
        figsize=(28.0, 15.5),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.05, right=0.935, bottom=0.075, top=0.91, wspace=0.14, hspace=0.54)

    last_clean_hb = None
    for row_index, iz in enumerate(range(len(Z_BINS))):
        zlo, zhi = Z_BINS[iz]
        for ir, (rlo, rhi) in enumerate(R_BINS):
            ax = axes[row_index, ir]
            raw_x, raw_y = panel_vectors(point_store, iz, ir, "raw")
            clean_x, clean_y = panel_vectors(point_store, iz, ir, "clean")

            ax.set_facecolor(VIOLET_BG)
            draw_hex_layer(ax, raw_x, raw_y, gridsize, xlim, ylim, "Greys", 0.55, 1)
            clean_hb = draw_hex_layer(ax, clean_x, clean_y, gridsize, xlim, ylim, "viridis", 1.0, 2, clean_norm)
            if clean_hb is not None:
                last_clean_hb = clean_hb
            draw_chemical_contours_and_crest(
                ax, clean_x, clean_y, xlim, ylim,
                show_contours=show_contours, crest=show_contours,
            )

            ax.grid(False)
            overlay_guides(ax, xlim, ylim, l2_fits[iz][ir])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_axisbelow(False)
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(
                axis="both",
                which="major",
                labelsize=10,
                direction="in",
                top=True,
                right=True,
                length=5.5,
                width=1.2,
                color="white",
            )
            ax.tick_params(
                axis="both",
                which="minor",
                direction="in",
                top=True,
                right=True,
                length=3.0,
                width=0.9,
                color="white",
            )
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_color("0.15")
                spine.set_linewidth(1.25)
            ax.set_title(
                rf"${rlo:g}<R<{rhi:g}$" + "\n" + rf"${zlo:g}<|Z|<{zhi:g}$" + "\n"
                + rf"$N_{{\rm clean}}={int(clean_panel_counts[iz, ir]):,}$",
                fontsize=10.5,
                fontweight="bold",
                pad=2,
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.label_outer()

    # Single shared axis labels for the whole figure (not per panel).
    fig.supxlabel(x_label, fontsize=22, fontweight="bold")
    fig.supylabel(y_label, fontsize=22, fontweight="bold")

    # Green boundary around the R in [3,15] kpc, |Z| in [0,2] kpc panels.
    col_idx = [i for i, (rlo, rhi) in enumerate(R_BINS) if rlo >= 3.0 - 1e-6 and rhi <= 15.0 + 1e-6]
    row_idx = [i for i, (zlo, zhi) in enumerate(Z_BINS) if zlo >= 0.0 - 1e-6 and zhi <= 2.0 + 1e-6]
    if col_idx and row_idx:
        p_tl = axes[row_idx[0], col_idx[0]].get_position()
        p_tr = axes[row_idx[0], col_idx[-1]].get_position()
        p_bl = axes[row_idx[-1], col_idx[0]].get_position()
        pad = 0.006
        rx0, rx1 = p_tl.x0 - pad, p_tr.x1 + pad
        ry0, ry1 = p_bl.y0 - pad, p_tl.y1 + pad
        fig.add_artist(
            Rectangle(
                (rx0, ry0), rx1 - rx0, ry1 - ry0, transform=fig.transFigure,
                fill=False, edgecolor="lime", linewidth=3.0, zorder=20, clip_on=False,
            )
        )

    legend_handles = [
        Patch(facecolor="0.6", edgecolor="none", label="Uncleaned"),
        Patch(facecolor=plt.get_cmap("viridis")(0.82), edgecolor="none", label="Cleaned density"),
    ]
    if show_contours:
        legend_handles.append(Line2D([0], [0], color="black", lw=1.2, label="Shape contours"))
        legend_handles.append(Line2D([0], [0], color="blue", lw=1.2, label="Core contours"))
    legend_handles += [
        Line2D([0], [0], color="red", lw=1.8, label="Sequence ridge"),
        Line2D([0], [0], color=GUIDE_MAGENTA, lw=1.3, ls="-", label="L1 fixed"),
        Line2D([0], [0], color=GUIDE_MAGENTA, lw=1.2, ls=":", label="L2 fitted"),
        Line2D([0], [0], color="lime", lw=2.4, label=r"$3<R<15,\ |Z|<2$ kpc"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=len(legend_handles),
        frameon=False,
        prop={"size": 15, "weight": "bold"},
        handlelength=2.2,
        columnspacing=1.6,
    )
    if last_clean_hb is not None:
        cax = fig.add_axes([0.947, 0.18, 0.012, 0.64])
        cb = fig.colorbar(last_clean_hb, cax=cax)
        cb.set_label(r"$\mathbf{Clean\ density\ (N/bin)}$", fontsize=14, fontweight="bold")
        cb.ax.tick_params(labelsize=11, width=1.2, length=4.5)
        for tick in cb.ax.get_yticklabels():
            tick.set_fontweight("bold")

    suffix = "" if show_contours else "_no_contours"
    fig.savefig(output_dir / f"{cat.key}_quality_motivated_rz_panels{suffix}.png", dpi=300)
    fig.savefig(output_dir / f"{cat.key}_quality_motivated_rz_panels{suffix}.pdf")
    plt.close(fig)
    return l2_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--bins", type=int, default=220, help="Hexbin gridsize per small panel.")
    parser.add_argument("--r0-kpc", type=float, default=8.122)
    parser.add_argument("--z-sun-kpc", type=float, default=0.025)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = base_fits_files(args.input_dir, args.include_duplicates)

    point_stores = {cat.key: empty_point_store() for cat in CATEGORIES}
    clean_panel_counts = {cat.key: np.zeros((len(Z_BINS), len(R_BINS)), dtype=np.int64) for cat in CATEGORIES}
    raw_panel_counts = {cat.key: np.zeros((len(Z_BINS), len(R_BINS)), dtype=np.int64) for cat in CATEGORIES}
    detail_rows: list[dict[str, object]] = []
    coord_summary = {
        "rows": 0,
        "valid_motion3": 0,
        "distance_final_positive": 0,
        "distance_parallax_fallback": 0,
        "usable_distance": 0,
        "coord_icrs": 0,
        "coord_lb_fallback": 0,
        "usable_position_distance": 0,
        "in_rz_panel_volume": 0,
    }

    print(f"[info] using {len(files)} unique FITS chunks", flush=True)
    for path in files:
        print(f"[file] {path.name}", flush=True)
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            nrows = len(data)
            for start in range(0, nrows, args.chunksize):
                stop = min(start + args.chunksize, nrows)
                block = data[start:stop]
                coord_summary["rows"] += len(block)
                kin = finite_motion_kinematics(block)
                coord_summary["valid_motion3"] += int(kin.sum())
                r_gc, z_gc, fallback_stats = galactocentric_rz_with_coordinate_fallbacks(
                    block, kin, args.r0_kpc, args.z_sun_kpc
                )
                for key, value in fallback_stats.items():
                    coord_summary[key] += value
                abs_z = np.abs(z_gc)
                base_mask = (
                    kin
                    & np.isfinite(r_gc)
                    & np.isfinite(abs_z)
                    & (r_gc >= R_BINS[0][0])
                    & (r_gc < R_BINS[-1][1])
                    & (abs_z >= Z_BINS[0][0])
                    & (abs_z < Z_BINS[-1][1])
                )
                coord_summary["in_rz_panel_volume"] += int(base_mask.sum())

                for cat in CATEGORIES:
                    limits = DISPLAY_LIMITS[cat.key]
                    for pair in cat.pairs:
                        update_category_panels(
                            point_stores[cat.key],
                            clean_panel_counts[cat.key],
                            raw_panel_counts[cat.key],
                            detail_rows,
                            cat.key,
                            pair,
                            block,
                            base_mask,
                            r_gc,
                            abs_z,
                            limits["xlim"],
                            limits["ylim"],
                        )

    l2_line_rows: list[dict[str, object]] = []
    for cat in CATEGORIES:
        for show_contours in (True, False):
            rows = plot_category(
                cat,
                point_stores[cat.key],
                clean_panel_counts[cat.key],
                raw_panel_counts[cat.key],
                args.output_dir,
                args.bins,
                show_contours=show_contours,
            )
            if show_contours:
                l2_line_rows.extend(rows)
        print(
            f"[plot] {cat.key}: clean={int(clean_panel_counts[cat.key].sum()):,}; "
            f"background={int(raw_panel_counts[cat.key].sum()):,} (with + no-contour copies)",
            flush=True,
        )

    panel_rows = []
    for cat in CATEGORIES:
        for iz, (zlo, zhi) in enumerate(Z_BINS):
            for ir, (rlo, rhi) in enumerate(R_BINS):
                panel_rows.append(
                    {
                        "category": cat.key,
                        "R_min_kpc": rlo,
                        "R_max_kpc": rhi,
                        "absZ_min_kpc": zlo,
                        "absZ_max_kpc": zhi,
                        "original_displayed_points": int(raw_panel_counts[cat.key][iz, ir]),
                        "clean_displayed_points": int(clean_panel_counts[cat.key][iz, ir]),
                    }
                )
    panel_df = pd.DataFrame(panel_rows)
    panel_df.to_csv(args.output_dir / "four_category_quality_motivated_rz_panel_counts.csv", index=False)
    l2_lines_df = pd.DataFrame(l2_line_rows)
    l2_lines_df.to_csv(args.output_dir / "four_category_quality_motivated_rz_l2_lines.csv", index=False)
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(args.output_dir / "four_category_quality_motivated_rz_pair_counts.csv", index=False)
    if not detail_df.empty:
        aggregate_cols = ["category", "contributing_pair", "quality_key"]
        count_cols = [
            "original_points_in_rz_volume",
            "original_displayed_points_in_rz_volume",
            "quality_points_in_rz_volume",
            "displayed_points_in_rz_volume",
        ]
        (
            detail_df.groupby(aggregate_cols, as_index=False)[count_cols]
            .sum()
            .sort_values(aggregate_cols)
            .to_csv(args.output_dir / "four_category_quality_motivated_rz_pair_counts_aggregated.csv", index=False)
        )
    pd.DataFrame([{**coord_summary, "r0_kpc": args.r0_kpc, "z_sun_kpc": args.z_sun_kpc}]).to_csv(
        args.output_dir / "four_category_quality_motivated_rz_coordinate_summary.csv", index=False
    )

    reduction_rows = []
    single_summary_path = args.output_dir / "four_category_clean_quality_counts_summary.csv"
    if single_summary_path.exists():
        single_df = pd.read_csv(single_summary_path)
        panel_summary = (
            panel_df.groupby("category", as_index=False)[["original_displayed_points", "clean_displayed_points"]]
            .sum()
            .rename(
                columns={
                    "original_displayed_points": "rz_original_background",
                    "clean_displayed_points": "rz_clean_count",
                }
            )
        )
        reduction_df = single_df[["category", "points_in_clean_plot"]].rename(
            columns={"points_in_clean_plot": "single_clean_count"}
        )
        reduction_df = reduction_df.merge(panel_summary, on="category", how="left").fillna(0)
        reduction_df["missing_from_rz_vs_single"] = (
            reduction_df["single_clean_count"] - reduction_df["rz_clean_count"]
        )
        reduction_df["rz_clean_fraction_of_single"] = np.where(
            reduction_df["single_clean_count"] > 0,
            reduction_df["rz_clean_count"] / reduction_df["single_clean_count"],
            np.nan,
        )
        reduction_df.to_csv(args.output_dir / "four_category_quality_motivated_rz_reduction_diagnostic.csv", index=False)
        reduction_rows = reduction_df.to_dict("records")

    with (args.output_dir / "README_quality_motivated_rz_panels.md").open("w", encoding="ascii") as f:
        f.write("# Quality-Motivated R-|Z| Chemical Panels\n\n")
        f.write("These plots use the same four pooled chemical categories as the single-plane figures.\n")
        f.write("Each category panel has a dark-violet background (the viridis zero colour) so empty regions are violet, not white, and the cleaned density stands out clearly.\n")
        f.write("The grey layer is the pre-quality-cut finite-abundance sample, and the viridis binned-density image layer is the cleaned sample. The density is drawn as a raster image rather than hexagonal patches to avoid close-zoom moire/interference from the hex lattice.\n")
        f.write("Contours and ridges, when enabled, are computed from the cleaned sample. Red lines are sequence ridges.\n")
        f.write("The fixed steep guide line L1 is drawn solid magenta. The thin/thick-disc separator L2 is drawn dotted magenta, fitted per panel from the two cleaned-density ridges when two ridges are detected, and replaced by the category-average L2 when a panel is too sparse or has only one ridge. L2 is clipped so its metal-poor/left side is not shown beyond L1.\n")
        f.write("Axis labels are shown only on the outer panel edges. The side colorbar is the shared clean-sample binned-density scale for the category figure.\n\n")
        f.write("The panel cleaning uses the same strict clean-quality selection as the single-plane figures (the `quality_mask` cuts): it removes catalogue warning/failed solutions, exact sentinel/grid-edge synthetic values, and clearly unconstrained or broad-posterior measurements. The goal is to remove systematics and unphysical/warning-flag stars, not to keep marginal data.\n")
        f.write("For Hattori XP this means CMD_GOOD, the low-extinction selection 0 < E(B-V) <= 0.1, and tight QRF posterior-width limits (Delta[M/H]_16-84 <= 0.30, Delta[alpha/M]_16-84 <= 0.15). The broad-posterior horizontal pile-up near [alpha/M] ~ +0.1 is a synthetic systematic and is removed by the QRF-width cut.\n")
        f.write("Axis limits are display limits only; they are not quality cuts.\n")
        f.write("Contours use the standard smoothed enclosed-mass density levels with an absolute noise floor, plus the sequence ridge line from the cleaned density field.\n\n")
        f.write("The spatial grid extends to 0 < R < 25 kpc and 0 < |Z| < 8 kpc.\n")
        f.write("Motion validity means finite `RV_final`, `pmra_final`, and `pmdec_final`.\n")
        f.write("Fresh X, Y, Z, and R are recomputed with Astropy `SkyCoord` transformed to `Galactocentric`.\n")
        f.write("Coordinate fallback order: use `RA_final`/`DEC_final` when available; otherwise use `l`/`b` directly as Galactic coordinates.\n")
        f.write("Distance fallback order: use positive `distance_final`; otherwise use positive `parallax_final` as inverse-parallax distance in kpc.\n")
        f.write("For the plotted position bins, R and Z depend on sky position and distance; RV and proper motions are required for the selected kinematic sample but do not change position-only R and Z.\n\n")
        f.write("Audit tables include per-panel counts, detailed per-chunk pair counts, aggregated pair counts by source, and `four_category_quality_motivated_rz_l2_lines.csv` with the per-panel L1/L2 separator parameters for later analysis.\n\n")
        f.write("Count diagnostic:\n")
        f.write(
            f"- This Astropy Galactocentric run has {coord_summary['rows']:,} rows, "
            f"{coord_summary['valid_motion3']:,} rows with finite RV/PM, "
            f"{coord_summary['usable_position_distance']:,} rows with usable position and distance after fallbacks, "
            f"and {coord_summary['in_rz_panel_volume']:,} rows in the requested R-|Z| volume.\n"
        )
        f.write(
            f"- Coordinate sources after fallbacks: ICRS RA/Dec = {coord_summary['coord_icrs']:,}; "
            f"Galactic l/b fallback = {coord_summary['coord_lb_fallback']:,}.\n"
        )
        f.write(
            f"- Distance sources after fallbacks: `distance_final` = {coord_summary['distance_final_positive']:,}; "
            f"inverse positive `parallax_final` fallback = {coord_summary['distance_parallax_fallback']:,}.\n"
        )
        for row in reduction_rows:
            if row["category"] == "combined_xfe_or_alpham_vs_metallicity":
                f.write(
                    f"- Combined clean panel count is {int(row['rz_clean_count']):,}, compared with "
                    f"{int(row['single_clean_count']):,} in the single clean chemical plane "
                    f"({row['rz_clean_fraction_of_single']:.1%}).\n"
                )
                break
        f.write("- See `four_category_quality_motivated_rz_reduction_diagnostic.csv` for category-by-category comparison.\n\n")
        f.write(f"R0 = {args.r0_kpc} kpc, z_sun = {args.z_sun_kpc} kpc. Duplicate `.1` FITS chunks are excluded by default.\n")


if __name__ == "__main__":
    main()
