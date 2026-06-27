#!/usr/bin/env python3
"""Four pooled chemical planes with source-specific abundance quality cuts."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm

from chemical_plane_contours import draw_chemical_contours_and_crest, dither

# Dark-violet background (viridis zero) so empty regions are violet, not white.
VIOLET_BG = plt.cm.viridis(0.0)


DEFAULT_INPUT_DIR = Path(
    "/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/output_abundance_crossmatch/"
    "augmented_fits_gaia_dr3_desi_dr1_apogee_dr17_galah_dr4_hattori_xp_all_ebv0p0_1p0_rave_dr6_ges_dr5_lamost_dr11"
)
DEFAULT_OUTPUT_DIR = Path(
    "/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/output_abundance_crossmatch/"
    "four_pooled_chemical_planes_basic6_kinematics"
)

KINEMATIC_COLUMNS = ("RA_final", "DEC_final", "distance_final", "RV_final", "pmra_final", "pmdec_final")
INT_SENTINEL_LIMIT = -10**18


@dataclass(frozen=True)
class Pair:
    label: str
    x_col: str
    y_col: str
    quality_key: str
    y_minus_x: tuple[str, str] | None = None


@dataclass(frozen=True)
class Category:
    key: str
    title: str
    x_label: str
    y_label: str
    pairs: tuple[Pair, ...]
    xlim: tuple[float, float]
    ylim: tuple[float, float]


ALPHA_M_PAIRS = (
    Pair("Hattori XP", "HATTORI_XP_MH", "HATTORI_XP_ALPHAM", "hattori_alpham"),
    Pair("APOGEE DR17", "APOGEE_DR17_M_H", "APOGEE_DR17_ALPHA_M", "apogee_alpham"),
)

MGFE_FEH_PAIRS = (
    Pair("APOGEE DR17", "APOGEE_DR17_FE_H", "APOGEE_DR17_MG_FE", "apogee_mgfe"),
    Pair("GALAH DR4", "GALAH_DR4_FE_H", "GALAH_DR4_MG_FE", "galah_mgfe"),
    Pair("LAMOST DR11 MRS", "LAMOST_DR11_MRS_FE_H", "LAMOST_DR11_MRS_MG_FE", "lamost_mgfe"),
    Pair("GES DR5", "GES_DR5_FE_H", "GES_DR5_MG_FE_FROM_MG1", "ges_mgfe"),
    Pair(
        "DESI DR1 SP elemental",
        "DESI_SP_ELEM_FE_H",
        "",
        "desi_sp_elem_mgfe",
        ("DESI_SP_ELEM_MG_H", "DESI_SP_ELEM_FE_H"),
    ),
)

ALPHAFE_FEH_PAIRS = (
    Pair("DESI DR1 RVS", "DESI_RVS_FEH", "DESI_RVS_ALPHAFE", "desi_rvs_alphafe"),
    Pair("DESI DR1 SP", "DESI_SP_FEH", "DESI_SP_ALPHAFE", "desi_sp_alphafe"),
    Pair("RAVE DR6", "RAVE_DR6_FE_H", "RAVE_DR6_ALPHA_FE", "rave_alphafe"),
)

COMBINED_EXTRA_PAIRS = (
    Pair("Gaia DR3 alpha", "GAIA_DR3_MH_GSPSPEC", "GAIA_DR3_ALPHAFE_GSPSPEC", "gaia_alphafe"),
    Pair("Gaia DR3 Mg", "GAIA_DR3_MH_GSPSPEC", "GAIA_DR3_MGFE_GSPSPEC", "gaia_mgfe"),
)

CATEGORIES = (
    Category(
        "alpha_m_vs_mh",
        "Clean [alpha/M] vs [M/H]",
        "[M/H]",
        "[alpha/M]",
        ALPHA_M_PAIRS,
        (-3.0, 0.8),
        (-0.15, 0.55),
    ),
    Category(
        "mgfe_vs_feh",
        "Clean [Mg/Fe] vs [Fe/H]",
        "[Fe/H]",
        "[Mg/Fe]",
        MGFE_FEH_PAIRS,
        (-3.5, 0.8),
        (-0.5, 1.0),
    ),
    Category(
        "alphafe_vs_feh",
        "Clean [alpha/Fe] vs [Fe/H]",
        "[Fe/H]",
        "[alpha/Fe]",
        ALPHAFE_FEH_PAIRS,
        (-3.5, 0.8),
        (-0.35, 0.8),
    ),
    Category(
        "combined_xfe_or_alpham_vs_metallicity",
        "Clean combined [alpha/M], [alpha/Fe], and [Mg/Fe] vs metallicity",
        "[Fe/H] or [M/H]",
        "[alpha/M] or [alpha/Fe] or [Mg/Fe]",
        ALPHA_M_PAIRS + MGFE_FEH_PAIRS + ALPHAFE_FEH_PAIRS + COMBINED_EXTRA_PAIRS,
        (-3.5, 0.8),
        (-0.5, 1.0),
    ),
)


def finite_basic_kinematics(data: fits.FITS_rec) -> np.ndarray:
    mask = np.ones(len(data), dtype=bool)
    for col in KINEMATIC_COLUMNS:
        mask &= np.isfinite(np.asarray(data[col], dtype=np.float64))
    return mask


def fcol(data: fits.FITS_rec, col: str) -> np.ndarray:
    return np.asarray(data[col], dtype=np.float64)


def icol(data: fits.FITS_rec, col: str) -> np.ndarray:
    return np.asarray(data[col], dtype=np.int64)


def finite_between(data: fits.FITS_rec, col: str, lo: float, hi: float) -> np.ndarray:
    arr = fcol(data, col)
    return np.isfinite(arr) & (arr >= lo) & (arr <= hi)


def err_max(data: fits.FITS_rec, col: str, max_err: float) -> np.ndarray:
    arr = fcol(data, col)
    return np.isfinite(arr) & (arr >= 0.0) & (arr <= max_err)


def int_eq(data: fits.FITS_rec, col: str, value: int) -> np.ndarray:
    arr = icol(data, col)
    return (arr > INT_SENTINEL_LIMIT) & (arr == value)


def bitmask_clear(data: fits.FITS_rec, col: str, bits: tuple[int, ...]) -> np.ndarray:
    arr = icol(data, col)
    valid = arr > INT_SENTINEL_LIMIT
    mask_value = np.int64(0)
    for bit in bits:
        mask_value |= np.int64(1) << np.int64(bit)
    return valid & ((arr & mask_value) == 0)


def qrf_width_ok(data: fits.FITS_rec, lo_col: str, hi_col: str, max_width: float) -> np.ndarray:
    lo = fcol(data, lo_col)
    hi = fcol(data, hi_col)
    width = hi - lo
    return np.isfinite(width) & (width >= 0.0) & (width <= max_width)


def not_near_values(arr: np.ndarray, values: tuple[float, ...], tol: float = 0.003) -> np.ndarray:
    good = np.ones(len(arr), dtype=bool)
    for value in values:
        good &= np.abs(arr - value) > tol
    return good


def gaia_flags_good(data: fits.FITS_rec, extra_indices_zero_based: tuple[int, ...] = ()) -> np.ndarray:
    flags = np.asarray(data["GAIA_DR3_FLAGS_GSPSPEC"], dtype="U41")
    if len(flags) == 0:
        return np.zeros(0, dtype=bool)
    chars = flags.view("U1").reshape(len(flags), 41)
    # First 13 characters describe parameter-level problems. Accept 0/1 only;
    # 2+ and 9 indicate warnings/bad cases that can bias chemistry.
    good = np.ones(len(flags), dtype=bool)
    for idx in range(13):
        good &= np.isin(chars[:, idx], ("0", "1"))
    for idx in extra_indices_zero_based:
        good &= chars[:, idx] == "0"
    return good


def apogee_good(data: fits.FITS_rec) -> np.ndarray:
    aspcap_bad_bits = (19, 20, 23, 24, 27, 31, 32, 33, 34, 35, 36, 40, 41)
    star_bad_bits = (0, 3, 4, 18, 22)
    return (
        finite_between(data, "APOGEE_DR17_SNR", 70.0, np.inf)
        & bitmask_clear(data, "APOGEE_DR17_ASPCAPFLAG", aspcap_bad_bits)
        & bitmask_clear(data, "APOGEE_DR17_STARFLAG", star_bad_bits)
    )


def quality_mask(data: fits.FITS_rec, pair: Pair, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    key = pair.quality_key

    if key == "hattori_alpham":
        return (
            int_eq(data, "HATTORI_XP_CMD_GOOD", 1)
            & finite_between(data, "HATTORI_XP_EBV_DUSTMAPS_086", 0.0, 0.1)
            & qrf_width_ok(data, "HATTORI_XP_MH_16_QRF", "HATTORI_XP_MH_84_QRF", 0.30)
            & qrf_width_ok(data, "HATTORI_XP_ALPHAM_16_QRF", "HATTORI_XP_ALPHAM_84_QRF", 0.15)
        )

    if key == "apogee_alpham":
        return apogee_good(data)

    if key == "apogee_mgfe":
        return (
            apogee_good(data)
            & err_max(data, "APOGEE_DR17_FE_H_ERR", 0.10)
            & err_max(data, "APOGEE_DR17_MG_FE_ERR", 0.10)
        )

    if key == "galah_mgfe":
        return (
            int_eq(data, "GALAH_DR4_SP_FLAG", 0)
            & int_eq(data, "GALAH_DR4_MG_FE_FLAG", 0)
            & err_max(data, "GALAH_DR4_FE_H_ERR", 0.15)
            & err_max(data, "GALAH_DR4_MG_FE_ERR", 0.15)
        )

    if key == "lamost_mgfe":
        return (
            finite_between(data, "LAMOST_DR11_MRS_SNR", 50.0, np.inf)
            & finite_between(data, "LAMOST_DR11_MRS_LOGG", 0.0, 5.5)
        )

    if key == "ges_mgfe":
        return (
            err_max(data, "GES_DR5_FE_H_ERR", 0.15)
            & err_max(data, "GES_DR5_MG1_A_ERR", 0.15)
        )

    if key == "desi_sp_elem_mgfe":
        return (
            int_eq(data, "DESI_SP_SUCCESS", 1)
            & finite_between(data, "DESI_SP_SNR_MED", 50.0, np.inf)
            & err_max(data, "DESI_SP_ELEM_FE_H_ERR", 0.10)
            & err_max(data, "DESI_SP_ELEM_MG_H_ERR", 0.15)
            & not_near_values(x, (-5.0, -4.0, -3.5, 0.5, 1.0))
        )

    if key == "desi_rvs_alphafe":
        return (
            np.asarray(data["DESI_RVS_SUCCESS"], dtype=bool)
            & np.asarray(data["DESI_RVS_PRIMARY"], dtype=bool)
            & int_eq(data, "DESI_RVS_WARN", 0)
            & err_max(data, "DESI_RVS_FEH_ERR", 0.20)
            & err_max(data, "DESI_RVS_ALPHAFE_ERR", 0.20)
            & not_near_values(x, (-4.0, 1.0))
            & not_near_values(y, (-0.4, 1.2))
        )

    if key == "desi_sp_alphafe":
        return (
            int_eq(data, "DESI_SP_SUCCESS", 1)
            & finite_between(data, "DESI_SP_SNR_MED", 50.0, np.inf)
            & not_near_values(x, (-10.0,))
            & not_near_values(y, (-0.2, 1.0))
        )

    if key == "rave_alphafe":
        return (
            err_max(data, "RAVE_DR6_FE_H_ERR", 0.20)
            & err_max(data, "RAVE_DR6_ALPHA_FE_ERR", 0.20)
            & not_near_values(y, (-0.2999,))
        )

    if key == "gaia_alphafe":
        return (
            gaia_flags_good(data)
            & qrf_width_ok(data, "GAIA_DR3_MH_GSPSPEC_LO", "GAIA_DR3_MH_GSPSPEC_HI", 0.30)
            & qrf_width_ok(data, "GAIA_DR3_ALPHAFE_GSPSPEC_LO", "GAIA_DR3_ALPHAFE_GSPSPEC_HI", 0.20)
        )

    if key == "gaia_mgfe":
        return (
            gaia_flags_good(data, extra_indices_zero_based=(15, 16))
            & qrf_width_ok(data, "GAIA_DR3_MH_GSPSPEC_LO", "GAIA_DR3_MH_GSPSPEC_HI", 0.30)
            & qrf_width_ok(data, "GAIA_DR3_MGFE_GSPSPEC_LO", "GAIA_DR3_MGFE_GSPSPEC_HI", 0.20)
            & finite_between(data, "GAIA_DR3_MGFE_GSPSPEC_NLINES", 1.0, np.inf)
            & finite_between(data, "GAIA_DR3_MGFE_GSPSPEC_LSCAT", 0.0, 0.10)
        )

    raise ValueError(f"Unknown quality key: {key}")


def pair_xy(data: fits.FITS_rec, pair: Pair) -> tuple[np.ndarray, np.ndarray]:
    x = fcol(data, pair.x_col)
    if pair.y_minus_x is None:
        y = fcol(data, pair.y_col)
    else:
        num, den = pair.y_minus_x
        y = fcol(data, num) - fcol(data, den)
    return x, y


def overlay_guides(ax: plt.Axes, xlim: tuple[float, float]) -> None:
    x = np.linspace(xlim[0], xlim[1], 400)
    ax.plot(x, -0.05000836526147814 * x + 0.2000005330440794, color="0.55", lw=1.0, ls="--", alpha=0.8)
    xh = x[x <= -0.5]
    ax.plot(xh, -0.8002225074868987 * xh - 0.6006222012635331, color="0.55", lw=0.8, ls="--", alpha=0.7)


def draw_density_image(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int,
    extent: tuple[float, float, float, float],
    cmap: str,
    alpha: float,
    zorder: int,
):
    if len(x) == 0:
        return None
    hist, _, _ = np.histogram2d(
        dither(x),
        dither(y),
        bins=gridsize,
        range=[(extent[0], extent[1]), (extent[2], extent[3])],
    )
    positive = hist[hist > 0]
    if positive.size == 0:
        return None
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0.0)
    return ax.imshow(
        np.ma.masked_where(hist.T <= 0, hist.T),
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap=cmap_obj,
        norm=LogNorm(vmin=1.0, vmax=max(2.0, float(np.percentile(positive, 99.7)))),
        alpha=alpha,
        interpolation="bilinear",
        rasterized=True,
        zorder=zorder,
    )


def save_hexbin(out_dir: Path, cat: Category, x: np.ndarray, y: np.ndarray,
                xu: np.ndarray, yu: np.ndarray, gridsize: int, groups=None) -> None:
    # Two copies per category: one with overdensity contours, one without.
    extent = (cat.xlim[0], cat.xlim[1], cat.ylim[0], cat.ylim[1])
    for show_contours in (True, False):
        fig, ax = plt.subplots(figsize=(7.4, 5.5), constrained_layout=True)
        ax.set_facecolor(VIOLET_BG)
        # Uncleaned (pre-quality-cut) sample as a grey background (dithered to
        # remove abundance-quantization interference stripes).
        if len(xu):
            draw_density_image(ax, xu, yu, gridsize, extent, "Greys", 0.55, 1)
        # Cleaned sample on top, with the colourbar (also dithered for display).
        if len(x):
            im = draw_density_image(ax, x, y, gridsize, extent, "viridis", 1.0, 2)
            if im is not None:
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("cleaned density (N/bin)")
        draw_chemical_contours_and_crest(
            ax, x, y, cat.xlim, cat.ylim, show_contours=show_contours, crest=show_contours,
            groups=groups,
        )
        overlay_guides(ax, cat.xlim)
        ax.set_xlim(*cat.xlim)
        ax.set_ylim(*cat.ylim)
        ax.set_xlabel(cat.x_label)
        ax.set_ylabel(cat.y_label)
        ax.set_title(f"{cat.title}\ngrey = uncleaned, colour = cleaned, valid basic kinematics, N={len(x):,}")
        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", color="white", top=True, right=True)
        suffix = "" if show_contours else "_no_contours"
        fig.savefig(out_dir / f"{cat.key}_clean_quality_hexbin{suffix}.png", dpi=260)
        fig.savefig(out_dir / f"{cat.key}_clean_quality_hexbin{suffix}.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--gridsize", type=int, default=280)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.input_dir.glob("Entire_catalogue_chunk*_chemistry.fits"))
    if not args.include_duplicates:
        paths = [p for p in paths if ".1_" not in p.name]
    if not paths:
        raise FileNotFoundError(f"No chemistry FITS files found in {args.input_dir}")

    x_lists: dict[str, list[np.ndarray]] = {cat.key: [] for cat in CATEGORIES}
    y_lists: dict[str, list[np.ndarray]] = {cat.key: [] for cat in CATEGORIES}
    # Uncleaned (pre-quality-cut) sample, shown as a grey background.
    xu_lists: dict[str, list[np.ndarray]] = {cat.key: [] for cat in CATEGORIES}
    yu_lists: dict[str, list[np.ndarray]] = {cat.key: [] for cat in CATEGORIES}
    summary = {cat.key: {"rows": 0, "valid_basic_kinematics": 0, "points": 0} for cat in CATEGORIES}
    detail_rows: list[dict[str, object]] = []

    # For the combined plane, collect cleaned points per abundance category so
    # contours can use the old balanced normalized-map calculation; the visible
    # viridis layer remains the ordinary pooled cleaned hexbin.
    COMBINED_KEY = "combined_xfe_or_alpham_vs_metallicity"
    GROUP_PAIRLISTS = [
        ("alpha_m", ALPHA_M_PAIRS),
        ("mgfe", MGFE_FEH_PAIRS),
        ("alphafe", ALPHAFE_FEH_PAIRS),
        ("gaia", COMBINED_EXTRA_PAIRS),
    ]
    cgx: dict[str, list[np.ndarray]] = {name: [] for name, _ in GROUP_PAIRLISTS}
    cgy: dict[str, list[np.ndarray]] = {name: [] for name, _ in GROUP_PAIRLISTS}

    def _pair_group(pair):
        for name, pair_list in GROUP_PAIRLISTS:
            if pair in pair_list:
                return name
        return None

    for path in paths:
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            kin = finite_basic_kinematics(data)
            n_kin = int(kin.sum())
            print(f"[read] {path.name}: valid_basic_kinematics={n_kin:,}", flush=True)
            for cat in CATEGORIES:
                summary[cat.key]["rows"] += len(data)
                summary[cat.key]["valid_basic_kinematics"] += n_kin
                for pair in cat.pairs:
                    x, y = pair_xy(data, pair)
                    base = (
                        kin
                        & np.isfinite(x)
                        & np.isfinite(y)
                        & (x >= cat.xlim[0])
                        & (x <= cat.xlim[1])
                        & (y >= cat.ylim[0])
                        & (y <= cat.ylim[1])
                    )
                    qmask = quality_mask(data, pair, x, y)
                    mask = base & qmask
                    n = int(mask.sum())
                    detail_rows.append(
                        {
                            "category": cat.key,
                            "contributing_pair": pair.label,
                            "quality_key": pair.quality_key,
                            "file": path.name,
                            "points": n,
                        }
                    )
                    if n:
                        x_lists[cat.key].append(x[mask].astype(np.float32))
                        y_lists[cat.key].append(y[mask].astype(np.float32))
                        summary[cat.key]["points"] += n
                        if cat.key == COMBINED_KEY:
                            gname = _pair_group(pair)
                            if gname is not None:
                                cgx[gname].append(x[mask].astype(np.float32))
                                cgy[gname].append(y[mask].astype(np.float32))
                    if base.any():
                        xu_lists[cat.key].append(x[base].astype(np.float32))
                        yu_lists[cat.key].append(y[base].astype(np.float32))

    summary_rows = []
    for cat in CATEGORIES:
        if x_lists[cat.key]:
            x = np.concatenate(x_lists[cat.key])
            y = np.concatenate(y_lists[cat.key])
        else:
            x = np.array([], dtype=np.float32)
            y = np.array([], dtype=np.float32)
        if xu_lists[cat.key]:
            xu = np.concatenate(xu_lists[cat.key])
            yu = np.concatenate(yu_lists[cat.key])
        else:
            xu = np.array([], dtype=np.float32)
            yu = np.array([], dtype=np.float32)
        groups = None
        if cat.key == COMBINED_KEY:
            groups = [
                (np.concatenate(cgx[name]), np.concatenate(cgy[name]))
                for name, _ in GROUP_PAIRLISTS if cgx[name]
            ]
        save_hexbin(args.output_dir, cat, x, y, xu, yu, args.gridsize, groups=groups)
        print(f"[plot] {cat.key}: {len(x):,}", flush=True)
        summary_rows.append(
            {
                "category": cat.key,
                "rows": summary[cat.key]["rows"],
                "valid_basic_kinematics": summary[cat.key]["valid_basic_kinematics"],
                "points_in_clean_plot": summary[cat.key]["points"],
                "x_label": cat.x_label,
                "y_label": cat.y_label,
            }
        )

    with (args.output_dir / "four_category_clean_quality_counts_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
                "rows",
                "valid_basic_kinematics",
                "points_in_clean_plot",
                "x_label",
                "y_label",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.output_dir / "four_category_clean_quality_counts_by_pair.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "contributing_pair", "quality_key", "file", "points"])
        writer.writeheader()
        writer.writerows(detail_rows)

    with (args.output_dir / "README_clean_quality.md").open("w") as f:
        f.write("# Cleaned Quality Four Pooled Chemical Planes\n\n")
        f.write("The plots are still pooled by chemical plane; no survey is visually separated.\n\n")
        f.write("Valid kinematics means exactly finite `RA_final`, `DEC_final`, `distance_final`, `RV_final`, `pmra_final`, and `pmdec_final`.\n\n")
        f.write("Additional cleaning applies survey-specific abundance quality cuts:\n\n")
        f.write("- Hattori XP: `CMD_GOOD == 1`, `0 < E(B-V) <= 0.1`, and QRF width cuts.\n")
        f.write("- Gaia DR3 GSP-Spec: good `FLAGS_GSPSPEC` parameter flags, abundance uncertainty bounds, Mg line checks where Mg is used.\n")
        f.write("- DESI: `SUCCESS`, `PRIMARY`/`WARN == 0` where available, S/N and abundance-error cuts, and removal only of exact sentinel/grid-edge abundance values.\n")
        f.write("- APOGEE: S/N >= 70, no selected bad ASPCAP/STARFLAG bits, and Mg/Fe error cuts where Mg is used.\n")
        f.write("- GALAH: `SP_FLAG == 0`, `MG_FE_FLAG == 0`, and abundance-error cuts. `FE_H_FLAG` is intentionally not used because GALAH DR4 documents a bug in that flag.\n")
        f.write("- LAMOST, RAVE, and GES: S/N/error cuts from the columns available in the augmented FITS.\n\n")
        f.write("References checked while choosing these cuts:\n")
        f.write("- https://zenodo.org/records/10902172\n")
        f.write("- https://arxiv.org/abs/2404.01269\n")
        f.write("- https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_astrophysical_parameter_tables/ssec_dm_astrophysical_parameters.html\n")
        f.write("- https://arxiv.org/abs/2206.05541\n")
        f.write("- https://www.galah-survey.org/dr4/using_the_data/\n")
        f.write("- https://www.galah-survey.org/dr4/flags/\n")
        f.write("- https://www.sdss4.org/dr17/irspec/apogee-bitmasks/\n")
        f.write("- https://arxiv.org/abs/2407.06280\n")
        f.write("- https://arxiv.org/abs/2305.05854\n")
        f.write("- https://arxiv.org/abs/2002.04512\n")


if __name__ == "__main__":
    main()
