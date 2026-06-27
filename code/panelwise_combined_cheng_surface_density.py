#!/usr/bin/env python3
"""Strict Cheng-style surface-density run for the clean combined chemical panels.

This script keeps the existing Cheng+2024 Jeans machinery in
``/user/sutirtha/surface_density_cheng.py`` unchanged, and only supplies a
new tracer selection:

* start from the same clean/viridis combined abundance points used by
  ``plot_four_quality_motivated_rz_panels.py``;
* classify each point in its own R-|Z| chemical panel using fixed L1 and the
  combined-panel L2: the directly fitted L2 where the valley/two-ridge fit
  succeeded, and the category-average fallback L2 (the same line drawn on the
  grid figure) elsewhere, so every panel is classified and used;
* run the Cheng-style velocity-dispersion, h_sigma, and Sigma(R, |Z|)
  calculation on the resulting thin/thick tracer samples.

Each panel records whether its L2 boundary was a direct fit or the
category-average fallback, so the distinction stays auditable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
from astropy.coordinates import Galactocentric, SkyCoord
from astropy.io import fits
from astropy import units as u

sys.path.insert(0, "/user/sutirtha")

import surface_density_cheng as sdc
from plot_four_pooled_chemical_planes_clean_quality import (
    CATEGORIES,
    DEFAULT_INPUT_DIR,
    quality_mask,
)
from plot_four_quality_motivated_rz_panels import (
    COMBINED_KEY,
    DISPLAY_LIMITS,
    HALO_LINE_B,
    HALO_LINE_M,
    R_BINS,
    Z_BINS,
    base_fits_files,
    distance_with_parallax_fallback,
    pair_xy,
)
from chemical_plane_contours import draw_chemical_contours_and_crest, dither


DEFAULT_RZ_OUTPUT_DIR = Path(
    "/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/output_abundance_crossmatch/"
    "four_pooled_chemical_planes_basic6_kinematics"
)
DEFAULT_OUTPUT_DIR = Path(
    "/user/sutirtha/cheng2024_panelwise_combined_l1l2_strict_surface_density"
)

POP_UNCLASSIFIED = np.uint8(0)
POP_THIN = np.uint8(1)
POP_THICK = np.uint8(2)
POP_HALO = np.uint8(3)
POP_NAMES = {
    int(POP_UNCLASSIFIED): "unclassified_no_l2_or_outside_disk",
    int(POP_THIN): "thin",
    int(POP_THICK): "thick",
    int(POP_HALO): "halo_l1_left",
}

FIG8_Z = np.array([0.3, 1.0, 3.0], dtype=np.float64)
OVERPLOT_Z = np.array([0.3, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0], dtype=np.float64)
PAPER_R0_KPC = 8.122
PAPER_Z_SUN_KPC = 0.0208
PAPER_VSUN_KMS = (12.9, 245.6, 7.78)
SURFACE_REJECT_ACCEPTED = "accepted"
SURFACE_REJECT_NONFINITE = "nonfinite_sigma"
SURFACE_REJECT_NEGATIVE_MEDIAN = "negative_median"
SURFACE_REJECT_NEGATIVE_BAND = "uncertainty_crosses_negative"
HSIGMA_DIRECT_PROFILE = "direct_interpolated_profile_fixed_Z"
HSIGMA_ODD_LINEAR_MODEL = "paper_odd_linear_sigma_RZ_model_fixed_Z"
HSIGMA_DIRECT_UPPER_BOUND_GUARD = 49.0
PANEL_GRID_DENSE_SAMPLES_PER_KPC = 80
PANEL_GRID_MIN_DENSE_SAMPLES = 80


def combined_category():
    for cat in CATEGORIES:
        if cat.key == COMBINED_KEY:
            return cat
    raise RuntimeError(f"Could not find combined category {COMBINED_KEY!r}")


def bin_edges(bins: tuple[tuple[float, float], ...]) -> np.ndarray:
    return np.array([bins[0][0]] + [hi for _, hi in bins], dtype=np.float64)


def panel_indices(r_gc: np.ndarray, abs_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_edges = bin_edges(R_BINS)
    z_edges = bin_edges(Z_BINS)
    ir = np.searchsorted(r_edges, r_gc, side="right") - 1
    iz = np.searchsorted(z_edges, abs_z, side="right") - 1
    ok = (
        np.isfinite(r_gc)
        & np.isfinite(abs_z)
        & (ir >= 0)
        & (ir < len(R_BINS))
        & (iz >= 0)
        & (iz < len(Z_BINS))
    )
    return iz.astype(np.int16), ir.astype(np.int16), ok


def load_strict_l2(l2_csv: Path) -> dict[tuple[int, int], tuple[float, float, bool]]:
    """Load every combined-panel L2 boundary as (slope, intercept, is_direct).

    The grid figure fills `l2_m`/`l2_b` for ALL panels: a direct per-panel fit
    where the two-ridge/valley detection succeeded (`l2_panel_fit == True`), and
    the category-average fallback line elsewhere. Both are returned here so every
    panel can be classified; the boolean records which kind was used.
    """
    if not l2_csv.exists():
        raise FileNotFoundError(
            f"Missing L2 table: {l2_csv}. Regenerate plot_four_quality_motivated_rz_panels.py first."
        )
    df = pd.read_csv(l2_csv)
    need = {
        "category",
        "R_min_kpc",
        "R_max_kpc",
        "absZ_min_kpc",
        "absZ_max_kpc",
        "l2_m",
        "l2_b",
        "l2_panel_fit",
    }
    missing = sorted(need.difference(df.columns))
    if missing:
        raise RuntimeError(f"L2 table {l2_csv} is missing columns: {missing}")

    out: dict[tuple[int, int], tuple[float, float, bool]] = {}
    sub = df[df["category"] == COMBINED_KEY]
    for _, row in sub.iterrows():
        if not (np.isfinite(row["l2_m"]) and np.isfinite(row["l2_b"])):
            continue
        iz = next(
            (
                i
                for i, (zlo, zhi) in enumerate(Z_BINS)
                if np.isclose(row["absZ_min_kpc"], zlo) and np.isclose(row["absZ_max_kpc"], zhi)
            ),
            None,
        )
        ir = next(
            (
                i
                for i, (rlo, rhi) in enumerate(R_BINS)
                if np.isclose(row["R_min_kpc"], rlo) and np.isclose(row["R_max_kpc"], rhi)
            ),
            None,
        )
        if iz is None or ir is None:
            continue
        out[(iz, ir)] = (float(row["l2_m"]), float(row["l2_b"]), bool(row["l2_panel_fit"]))
    if not out:
        raise RuntimeError(f"No combined-panel L2 rows found in {l2_csv}")
    return out


def transform_to_galcen_cyl(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    distance_kpc: np.ndarray,
    pmra_masyr: np.ndarray,
    pmdec_masyr: np.ndarray,
    rv_kms: np.ndarray,
    batch_size: int = 200_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame = Galactocentric(
        galcen_distance=PAPER_R0_KPC * u.kpc,
        z_sun=PAPER_Z_SUN_KPC * u.kpc,
        galcen_v_sun=u.Quantity(PAPER_VSUN_KMS, unit=u.km / u.s),
    )
    n = len(ra_deg)
    r_gc = np.full(n, np.nan, dtype=np.float64)
    z_gc = np.full(n, np.nan, dtype=np.float64)
    v_r = np.full(n, np.nan, dtype=np.float64)
    v_z = np.full(n, np.nan, dtype=np.float64)
    v_phi = np.full(n, np.nan, dtype=np.float64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl = slice(start, end)
        ok = (
            np.isfinite(ra_deg[sl])
            & np.isfinite(dec_deg[sl])
            & np.isfinite(distance_kpc[sl])
            & (distance_kpc[sl] > 0.0)
            & np.isfinite(pmra_masyr[sl])
            & np.isfinite(pmdec_masyr[sl])
            & np.isfinite(rv_kms[sl])
        )
        if not np.any(ok):
            continue
        coords = SkyCoord(
            ra=ra_deg[sl][ok] * u.deg,
            dec=dec_deg[sl][ok] * u.deg,
            distance=distance_kpc[sl][ok] * u.kpc,
            pm_ra_cosdec=pmra_masyr[sl][ok] * u.mas / u.yr,
            pm_dec=pmdec_masyr[sl][ok] * u.mas / u.yr,
            radial_velocity=rv_kms[sl][ok] * u.km / u.s,
            frame="icrs",
        )
        gc = coords.transform_to(frame)
        x = gc.x.to_value(u.kpc)
        y = gc.y.to_value(u.kpc)
        z = gc.z.to_value(u.kpc)
        vx = gc.v_x.to_value(u.km / u.s)
        vy = gc.v_y.to_value(u.km / u.s)
        vz = gc.v_z.to_value(u.km / u.s)
        r = np.hypot(x, y)
        r_safe = np.where(r > 1e-9, r, 1e-9)
        global_idx = np.flatnonzero(ok) + start
        r_gc[global_idx] = r
        z_gc[global_idx] = z
        v_r[global_idx] = (x * vx + y * vy) / r_safe
        v_z[global_idx] = vz
        v_phi[global_idx] = (y * vx - x * vy) / r_safe

    return r_gc, z_gc, v_r, v_z, v_phi


def source_pair_table(cat) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pair_id": i,
                "label": pair.label,
                "quality_key": pair.quality_key,
                "x_col": pair.x_col,
                "y_col": pair.y_col,
                "y_minus_x": "" if pair.y_minus_x is None else f"{pair.y_minus_x[0]}-{pair.y_minus_x[1]}",
            }
            for i, pair in enumerate(cat.pairs)
        ]
    )


def clean_pair_mask(data, pair, x, y, xlim, ylim, velocity_input) -> np.ndarray:
    finite_xy = np.isfinite(x) & np.isfinite(y)
    displayed = finite_xy & (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    return velocity_input & displayed & quality_mask(data, pair, x, y)


def classify_points(
    x: np.ndarray,
    y: np.ndarray,
    iz: np.ndarray,
    ir: np.ndarray,
    strict_l2: dict[tuple[int, int], tuple[float, float, bool]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pop = np.full(len(x), POP_UNCLASSIFIED, dtype=np.uint8)
    l2_direct = np.zeros(len(x), dtype=bool)
    l2_m = np.full(len(x), np.nan, dtype=np.float32)
    l2_b = np.full(len(x), np.nan, dtype=np.float32)

    # Left side of the steep L1 boundary is the halo partition.
    halo = y <= HALO_LINE_M * x + HALO_LINE_B
    pop[halo] = POP_HALO

    for (pz, pr), (m, b, is_direct) in strict_l2.items():
        panel = (iz == pz) & (ir == pr)
        if not np.any(panel):
            continue
        l2_direct[panel] = bool(is_direct)
        l2_m[panel] = m
        l2_b[panel] = b
        disk = panel & ~halo
        thick = disk & (y >= m * x + b)
        thin = disk & ~thick
        pop[thin] = POP_THIN
        pop[thick] = POP_THICK

    return pop, l2_direct, l2_m, l2_b


def build_strict_cache(args, strict_l2: dict[tuple[int, int], tuple[float, float, bool]]) -> None:
    cat = combined_category()
    limits = DISPLAY_LIMITS[COMBINED_KEY]
    xlim = limits["xlim"]
    ylim = limits["ylim"]
    files = base_fits_files(args.input_dir, args.include_duplicates)
    if args.max_chunks is not None:
        files = files[: args.max_chunks]

    acc: dict[str, list[np.ndarray]] = {
        "R": [],
        "Z": [],
        "vR": [],
        "vZ": [],
        "vp": [],
        "xchem": [],
        "ychem": [],
        "panel_iz": [],
        "panel_ir": [],
        "pop": [],
        "l2_direct": [],
        "l2_m": [],
        "l2_b": [],
        "pair_id": [],
    }
    chunk_rows = []

    for file_index, path in enumerate(files, 1):
        print(f"[cache] {file_index}/{len(files)} {path.name}", flush=True)
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data
            n = len(data)
            distance, distance_final, parallax_fallback = distance_with_parallax_fallback(data)
            velocity_input = (
                np.isfinite(np.asarray(data["RA_final"], dtype=np.float64))
                & np.isfinite(np.asarray(data["DEC_final"], dtype=np.float64))
                & np.isfinite(distance)
                & (distance > 0.0)
                & np.isfinite(np.asarray(data["pmra_final"], dtype=np.float64))
                & np.isfinite(np.asarray(data["pmdec_final"], dtype=np.float64))
                & np.isfinite(np.asarray(data["RV_final"], dtype=np.float64))
            )

            any_clean = np.zeros(n, dtype=bool)
            pre_counts = []
            for pair_id, pair in enumerate(cat.pairs):
                x, y = pair_xy(data, pair)
                pm = clean_pair_mask(data, pair, x, y, xlim, ylim, velocity_input)
                any_clean |= pm
                pre_counts.append(
                    {
                        "chunk": path.name,
                        "pair_id": pair_id,
                        "quality_key": pair.quality_key,
                        "pre_transform_clean_velocity_input_points": int(pm.sum()),
                    }
                )

            selected = np.flatnonzero(any_clean)
            if selected.size == 0:
                chunk_rows.extend(pre_counts)
                continue

            r_gc, z_gc, v_r, v_z, v_phi = transform_to_galcen_cyl(
                np.asarray(data["RA_final"], dtype=np.float64)[selected],
                np.asarray(data["DEC_final"], dtype=np.float64)[selected],
                distance[selected],
                np.asarray(data["pmra_final"], dtype=np.float64)[selected],
                np.asarray(data["pmdec_final"], dtype=np.float64)[selected],
                np.asarray(data["RV_final"], dtype=np.float64)[selected],
                batch_size=args.transform_batch_size,
            )
            abs_z = np.abs(z_gc)
            iz_all, ir_all, in_panel = panel_indices(r_gc, abs_z)
            finite_phase = (
                in_panel
                & np.isfinite(v_r)
                & np.isfinite(v_z)
                & np.isfinite(v_phi)
                & np.isfinite(r_gc)
                & np.isfinite(z_gc)
            )

            for row in pre_counts:
                row["post_transform_any_clean_unique_rows"] = int(selected.size)
                row["post_transform_unique_rows_in_rz_volume"] = int(finite_phase.sum())
            chunk_rows.extend(pre_counts)

            for pair_id, pair in enumerate(cat.pairs):
                x_full, y_full = pair_xy(data, pair)
                pm_full = clean_pair_mask(data, pair, x_full, y_full, xlim, ylim, velocity_input)
                pm = pm_full[selected] & finite_phase
                if not np.any(pm):
                    continue

                x = x_full[selected][pm].astype(np.float32)
                y = y_full[selected][pm].astype(np.float32)
                iz = iz_all[pm]
                ir = ir_all[pm]
                pop, l2_direct, l2_m, l2_b = classify_points(x, y, iz, ir, strict_l2)

                acc["R"].append(r_gc[pm].astype(np.float32))
                acc["Z"].append(z_gc[pm].astype(np.float32))
                acc["vR"].append(v_r[pm].astype(np.float32))
                acc["vZ"].append(v_z[pm].astype(np.float32))
                acc["vp"].append(v_phi[pm].astype(np.float32))
                acc["xchem"].append(x)
                acc["ychem"].append(y)
                acc["panel_iz"].append(iz.astype(np.int16))
                acc["panel_ir"].append(ir.astype(np.int16))
                acc["pop"].append(pop)
                acc["l2_direct"].append(l2_direct)
                acc["l2_m"].append(l2_m)
                acc["l2_b"].append(l2_b)
                acc["pair_id"].append(np.full(int(pm.sum()), pair_id, dtype=np.int16))

    if not acc["R"]:
        raise RuntimeError("No clean combined viridis points survived velocity transformation and R-|Z| volume cuts.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_pair_table(cat).to_csv(args.output_dir / "combined_source_pair_ids.csv", index=False)
    pd.DataFrame(chunk_rows).to_csv(args.output_dir / "strict_cache_chunk_pair_audit.csv", index=False)

    arrays = {key: np.concatenate(values) for key, values in acc.items()}
    np.savez_compressed(args.cache, **arrays)
    print(f"[cache] wrote {args.cache} with {len(arrays['R']):,} clean combined points", flush=True)


def load_cache(cache: Path) -> dict[str, np.ndarray]:
    z = np.load(cache)
    return {key: z[key] for key in z.files}


def write_classification_counts(data: dict[str, np.ndarray], strict_l2, outd: Path) -> pd.DataFrame:
    rows = []
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            in_panel = (data["panel_iz"] == iz) & (data["panel_ir"] == ir)
            l2_entry = strict_l2.get((iz, ir))
            is_direct = bool(l2_entry[2]) if l2_entry is not None else False
            row = {
                "R_min_kpc": rlo,
                "R_max_kpc": rhi,
                "absZ_min_kpc": zlo,
                "absZ_max_kpc": zhi,
                "has_direct_l2": is_direct,
                "l2_kind": ("direct" if is_direct else "fallback_category_average")
                if l2_entry is not None
                else "none",
                "clean_combined_points": int(in_panel.sum()),
            }
            for code, name in POP_NAMES.items():
                row[f"n_{name}"] = int((in_panel & (data["pop"] == code)).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outd / "panel_l1_l2_population_counts.csv", index=False)
    return df


def plot_population_counts(counts: pd.DataFrame, outd: Path) -> None:
    fig, axes = plt.subplots(len(Z_BINS), len(R_BINS), figsize=(24, 12), sharex=False, sharey=False)
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            ax = axes[iz, ir]
            row = counts[
                np.isclose(counts["R_min_kpc"], rlo)
                & np.isclose(counts["R_max_kpc"], rhi)
                & np.isclose(counts["absZ_min_kpc"], zlo)
                & np.isclose(counts["absZ_max_kpc"], zhi)
            ].iloc[0]
            vals = [
                row["n_thin"],
                row["n_thick"],
                row["n_halo_l1_left"],
                row["n_unclassified_no_l2_or_outside_disk"],
            ]
            labels = ["thin", "thick", "halo", "no L2"]
            colors = ["#2166ac", "#b2182b", "#4daf4a", "0.65"]
            ax.bar(labels, vals, color=colors, width=0.72)
            title = f"{rlo:g}<R<{rhi:g}\n{zlo:g}<|Z|<{zhi:g}"
            if not bool(row["has_direct_l2"]):
                title += "\n(fallback L2)"
            ax.set_title(title, fontsize=8)
            ax.set_yscale("log")
            ax.set_ylim(0.8, max(10.0, max(vals) * 1.8))
            ax.tick_params(axis="x", labelrotation=90, labelsize=7)
            ax.tick_params(axis="y", labelsize=7)
    fig.supylabel("clean combined points")
    fig.tight_layout()
    fig.savefig(outd / "panel_l1_l2_population_counts.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def estimate_panel_group_geometry(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    group_size: int,
    z_targets: list[float],
    rlo: float,
    rhi: float,
    rbw: float = 1.0,
    ztol: float = 0.45,
) -> dict[str, object]:
    """Estimate whether a single original R-|Z| panel can support Cheng h_sigma.

    This mirrors the geometry of sdc.cprof/sdc.crz_at_z without bootstrapping:
    in 1 kpc R bins, sort stars by signed Z, chunk into Cheng group sizes, then
    count R bins that bracket each required z target within the paper-code
    interpolation tolerance. It is a feasibility audit, not a fallback fit.
    """
    rr = np.asarray(data["R"][mask], dtype=float)
    zz = np.asarray(data["Z"][mask], dtype=float)
    finite = np.isfinite(rr) & np.isfinite(zz)
    rr = rr[finite]
    zz = zz[finite]

    profile_groups = 0
    r_bins_with_groups = 0
    medians_by_r: list[tuple[float, np.ndarray]] = []
    for rl in np.arange(rlo, rhi, rbw):
        rh = min(float(rl + rbw), float(rhi))
        ii = np.where((rr >= rl) & (rr < rh))[0]
        n_groups = len(ii) // group_size if len(ii) >= group_size else 0
        if n_groups <= 0:
            continue
        ii = ii[np.argsort(zz[ii])]
        z_medians = []
        for gi in range(n_groups):
            jj = ii[gi * group_size:(gi + 1) * group_size]
            z_medians.append(float(np.median(zz[jj])))
        medians = np.array(z_medians, dtype=float)
        medians_by_r.append((float(0.5 * (rl + rh)), medians))
        profile_groups += int(n_groups)
        r_bins_with_groups += 1

    target_counts = {}
    target_r_values = {}
    total = 0
    for zt in z_targets:
        r_values = []
        for rm, medians in medians_by_r:
            if len(medians) < 2:
                continue
            below = medians[medians <= zt]
            above = medians[medians >= zt]
            if not len(below) or not len(above):
                continue
            if abs(float(below.max()) - zt) <= ztol and abs(float(above.min()) - zt) <= ztol:
                r_values.append(rm)
        target_counts[float(zt)] = int(len(r_values))
        target_r_values[float(zt)] = ",".join(f"{rv:.1f}" for rv in r_values)
        total += len(r_values)

    return {
        "n_finite_points": int(len(rr)),
        "r_bins_with_groups": int(r_bins_with_groups),
        "profile_groups": int(profile_groups),
        "h_sigma_target_points": int(total),
        "target_counts": target_counts,
        "target_r_values": target_r_values,
    }


def write_panel_cheng_feasibility(
    data: dict[str, np.ndarray],
    strict_l2,
    outd: Path,
    rbw: float = 1.0,
    thin_group_size: int | None = None,
    halo_group_size: int | None = None,
) -> pd.DataFrame:
    rows = []
    if thin_group_size is None:
        thin_group_size = int(sdc.TN["ng"])
    if halo_group_size is None:
        halo_group_size = 100
    specs = [
        ("thin", POP_THIN, int(thin_group_size), [-1.0, 1.0], True),
        ("thick", POP_THICK, int(sdc.TK["ng"]), [-2.0, -1.0, 1.0, 2.0], True),
        ("halo", POP_HALO, int(halo_group_size), [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0], False),
    ]
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            in_panel = (data["panel_iz"] == iz) & (data["panel_ir"] == ir)
            l2_entry = strict_l2.get((iz, ir))
            has_l2 = l2_entry is not None
            is_direct = bool(l2_entry[2]) if l2_entry is not None else False
            for pop_name, pop_code, group_size, z_targets, requires_l2 in specs:
                pop_mask = in_panel & (data["pop"] == pop_code)
                n_points = int(pop_mask.sum())
                geom = estimate_panel_group_geometry(data, pop_mask, group_size, z_targets, rlo, rhi, rbw=rbw)
                if requires_l2 and not has_l2:
                    status = "blocked_no_L2"
                elif n_points < group_size:
                    status = "blocked_too_few_points_for_one_group"
                elif int(geom["profile_groups"]) == 0:
                    status = "blocked_no_1kpc_Z_sorted_groups"
                elif int(geom["h_sigma_target_points"]) < 6:
                    status = "blocked_hsigma_target_coverage_lt_6"
                else:
                    status = "standalone_panel_fit_possible_not_used"
                row = {
                    "population": pop_name,
                    "R_min_kpc": rlo,
                    "R_max_kpc": rhi,
                    "absZ_min_kpc": zlo,
                    "absZ_max_kpc": zhi,
                    "has_direct_l2": bool(is_direct),
                    "l2_kind": ("direct" if is_direct else "fallback_category_average") if has_l2 else "none",
                    "requires_direct_l2": bool(requires_l2),
                    "n_points": n_points,
                    "cheng_group_size": group_size,
                    "n_finite_points": geom["n_finite_points"],
                    "r_bins_with_groups": geom["r_bins_with_groups"],
                    "profile_groups_if_panel_only": geom["profile_groups"],
                    "h_sigma_target_points_if_panel_only": geom["h_sigma_target_points"],
                    "standalone_panel_status": status,
                }
                for zt in z_targets:
                    row[f"h_sigma_target_count_Z_{zt:+.1f}"] = geom["target_counts"][float(zt)]
                    row[f"h_sigma_target_R_values_Z_{zt:+.1f}"] = geom["target_r_values"][float(zt)]
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(outd / "original_panel_cheng_feasibility.csv", index=False)
    return df


def write_original_panel_surface_assignments(
    results: dict[str, dict[str, object]],
    outd: Path,
) -> pd.DataFrame:
    rows = []
    for label, result in results.items():
        surface = result.get("surface", pd.DataFrame())
        pp = result.get("pp", None)
        if surface.empty or pp is None:
            continue
        z_values = np.array(sorted(surface["Z_abs"].dropna().unique()), dtype=float)
        if z_values.size == 0:
            continue
        for iz, (zlo, zhi) in enumerate(Z_BINS):
            z_mid = 0.5 * (zlo + zhi)
            if z_mid > float(pp["zmax"]) + 1e-9:
                continue
            z_use = float(z_values[np.argmin(np.abs(z_values - z_mid))])
            for ir, (rlo, rhi) in enumerate(R_BINS):
                sub = surface[
                    (surface["R_mid"] >= rlo)
                    & (surface["R_mid"] < rhi)
                    & np.isclose(surface["Z_abs"], z_use)
                ].sort_values("R_mid")
                for _, row in sub.iterrows():
                    rows.append(
                        {
                            "population": label,
                            "panel_iz": iz,
                            "panel_ir": ir,
                            "R_min_kpc": rlo,
                            "R_max_kpc": rhi,
                            "absZ_min_kpc": zlo,
                            "absZ_max_kpc": zhi,
                            "panel_absZ_mid_kpc": z_mid,
                            "surface_Z_abs_used": z_use,
                            "surface_Z_delta_kpc": abs(z_use - z_mid),
                            "R_mid": float(row["R_mid"]),
                            "accepted_nonnegative": bool(row["accepted_nonnegative"]),
                            "reject_reason": str(row["reject_reason"]),
                            "Sigma_median_Msun_pc2": float(row["Sigma_median_Msun_pc2"])
                            if np.isfinite(row["Sigma_median_Msun_pc2"]) else np.nan,
                            "Sigma_p16_Msun_pc2": float(row["Sigma_p16_Msun_pc2"])
                            if np.isfinite(row["Sigma_p16_Msun_pc2"]) else np.nan,
                            "Sigma_p84_Msun_pc2": float(row["Sigma_p84_Msun_pc2"])
                            if np.isfinite(row["Sigma_p84_Msun_pc2"]) else np.nan,
                            "Sigma_raw_median_Msun_pc2": float(row["Sigma_raw_median_Msun_pc2"]),
                            "Sigma_raw_p16_Msun_pc2": float(row["Sigma_raw_p16_Msun_pc2"]),
                            "Sigma_raw_p84_Msun_pc2": float(row["Sigma_raw_p84_Msun_pc2"]),
                            "Kz_km2_s2_kpc": float(row["Kz_km2_s2_kpc"])
                            if np.isfinite(row["Kz_km2_s2_kpc"]) else np.nan,
                            "Kz_raw_km2_s2_kpc": float(row["Kz_raw_km2_s2_kpc"]),
                            "SHM_Msun_pc2": float(row["SHM_Msun_pc2"]),
                            "SHM_Kz_km2_s2_kpc": float(row["SHM_Kz_km2_s2_kpc"]),
                            "Sigma_over_SHM": float(row["Sigma_over_SHM"])
                            if np.isfinite(row["Sigma_over_SHM"]) else np.nan,
                            "SHM_inside_accepted_1sigma": bool(row["SHM_inside_accepted_1sigma"]),
                        }
                    )
    df = pd.DataFrame(rows)
    df.to_csv(outd / "original_panel_surface_assignments.csv", index=False)
    return df


def write_fig8_panel_assignment_table(outd: Path) -> pd.DataFrame:
    paths = [
        outd / "fig8_sigma_R_points_strict_panel_l2.csv",
        outd / "fig8_sigma_R_rejected_nonphysical_strict_panel_l2.csv",
    ]
    frames = [pd.read_csv(path) for path in paths if path.exists()]
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.to_csv(outd / "original_panel_fig8_sigma_assignments.csv", index=False)
    return df


def plot_original_panel_sigma_assignments(assignments: pd.DataFrame, outd: Path) -> None:
    if assignments.empty:
        write_status_figure(
            outd,
            "original_panel_sigma_R_assignments.png",
            "No Original-Panel Sigma Assignments",
            ["No strict surface-density assignment rows were produced."],
        )
        return
    styles = {
        "thin": dict(color="#2166ac", marker="o", ls="-"),
        "thick": dict(color="#b2182b", marker="s", ls="--"),
        "halo": dict(color="#238b45", marker="^", ls="-."),
    }
    fig, axes = plt.subplots(len(Z_BINS), 1, figsize=(10, 13), sharex=True)
    if len(Z_BINS) == 1:
        axes = [axes]
    for ax, (zlo, zhi) in zip(axes, Z_BINS):
        sub_z = assignments[
            np.isclose(assignments["absZ_min_kpc"], zlo)
            & np.isclose(assignments["absZ_max_kpc"], zhi)
        ].copy()
        if sub_z.empty:
            ax.text(0.5, 0.5, "no strict surface rows", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(0, 1)
        else:
            rej = sub_z[~sub_z["accepted_nonnegative"]].sort_values("R_mid")
            for label, style in styles.items():
                acc = sub_z[
                    (sub_z["population"] == label)
                    & sub_z["accepted_nonnegative"]
                ].sort_values("R_mid")
                if acc.empty:
                    continue
                ax.errorbar(
                    acc["R_mid"],
                    acc["Sigma_median_Msun_pc2"],
                    yerr=[
                        acc["Sigma_median_Msun_pc2"] - acc["Sigma_p16_Msun_pc2"],
                        acc["Sigma_p84_Msun_pc2"] - acc["Sigma_median_Msun_pc2"],
                    ],
                    fmt=style["marker"],
                    ls=style["ls"],
                    color=style["color"],
                    ms=4,
                    lw=1.4,
                    capsize=2,
                    label=f"accepted strict {label}",
                )
            if not rej.empty:
                for label, style in styles.items():
                    rej_l = rej[rej["population"] == label]
                    if rej_l.empty:
                        continue
                    ax.scatter(
                        rej_l["R_mid"],
                        rej_l["Sigma_raw_median_Msun_pc2"],
                        marker="x",
                        color=style["color"],
                        alpha=0.65,
                        s=25,
                        label=f"raw rejected {label}",
                    )
            z_ref = float(sub_z["surface_Z_abs_used"].iloc[0])
            rr = np.linspace(float(min(R_BINS[0][0], sub_z["R_mid"].min())), float(max(R_BINS[-1][1], sub_z["R_mid"].max())), 260)
            shm_vals = np.array([sdc.shm_sigma(float(rv), [z_ref])[0] for rv in rr], dtype=float)
            ax.plot(rr, shm_vals, color="firebrick", lw=1.4, label="SHM")
            vals = sub_z.loc[sub_z["accepted_nonnegative"], "Sigma_median_Msun_pc2"].dropna().tolist()
            vals += rej["Sigma_raw_median_Msun_pc2"].dropna().tolist()
            if vals:
                ymin = min(0.0, min(vals) * 1.1)
                ymax = max(max(vals) * 1.2, 1.0)
                ax.set_ylim(ymin, ymax)
        ax.axhline(0, color="0.4", lw=0.8, ls=":")
        ax.text(0.98, 0.88, f"{zlo:g}<|Z|<{zhi:g}", transform=ax.transAxes, ha="right", va="top")
        ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        ax.tick_params(direction="in", top=True, right=True)
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        secax.set_ylabel(r"$|K_Z|$")
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "original_panel_sigma_R_assignments.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_original_panel_sigma_z_assignments(assignments: pd.DataFrame, outd: Path) -> None:
    if assignments.empty:
        write_status_figure(
            outd,
            "original_panel_sigma_Z_assignments.png",
            "No Original-Panel Sigma-Z Assignments",
            ["No strict surface-density assignment rows were produced."],
        )
        return
    styles = {
        "thin": dict(color="#2166ac", marker="o", ls="-"),
        "thick": dict(color="#b2182b", marker="s", ls="--"),
    }
    ncols = 3
    nrows = int(np.ceil(len(R_BINS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 11), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    for pi, (rlo, rhi) in enumerate(R_BINS):
        ax = axes_flat[pi]
        sub_r = assignments[
            np.isclose(assignments["R_min_kpc"], rlo)
            & np.isclose(assignments["R_max_kpc"], rhi)
        ].copy()
        if sub_r.empty:
            ax.text(0.5, 0.5, "no strict surface rows", transform=ax.transAxes, ha="center", va="center", fontsize=9)
            ax.set_ylim(0, 1)
        else:
            for label, style in styles.items():
                acc = sub_r[
                    (sub_r["population"] == label)
                    & sub_r["accepted_nonnegative"]
                ].sort_values(["panel_absZ_mid_kpc", "R_mid"])
                if acc.empty:
                    continue
                ax.errorbar(
                    acc["surface_Z_abs_used"],
                    acc["Sigma_median_Msun_pc2"],
                    yerr=[
                        acc["Sigma_median_Msun_pc2"] - acc["Sigma_p16_Msun_pc2"],
                        acc["Sigma_p84_Msun_pc2"] - acc["Sigma_median_Msun_pc2"],
                    ],
                    fmt=style["marker"],
                    ls="",
                    color=style["color"],
                    ms=4,
                    lw=0.9,
                    capsize=2,
                    alpha=0.82,
                    label=f"{label}" if pi == 0 else None,
                )
                grouped = (
                    acc.groupby("surface_Z_abs_used", as_index=False)["Sigma_median_Msun_pc2"]
                    .median()
                    .sort_values("surface_Z_abs_used")
                )
                if len(grouped) >= 2:
                    ax.plot(
                        grouped["surface_Z_abs_used"],
                        grouped["Sigma_median_Msun_pc2"],
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.5,
                        alpha=0.95,
                    )
            rej = sub_r[~sub_r["accepted_nonnegative"]]
            if not rej.empty:
                ax.scatter(
                    rej["surface_Z_abs_used"],
                    rej["Sigma_raw_median_Msun_pc2"],
                    marker="x",
                    color="0.35",
                    s=18,
                    alpha=0.55,
                    label="raw rejected" if pi == 0 else None,
                )
            zmax = max(float(sub_r["surface_Z_abs_used"].max()), 0.1)
            zr = np.linspace(0.01, min(4.0, zmax + 0.2), 180)
            r_ref = float(sub_r["R_mid"].median())
            ax.plot(zr, sdc.shm_sigma(r_ref, zr), color="firebrick", lw=1.2, alpha=0.9, label="SHM" if pi == 0 else None)
            vals = sub_r.loc[sub_r["accepted_nonnegative"], "Sigma_median_Msun_pc2"].dropna().tolist()
            if vals:
                ax.set_ylim(0, max(max(vals) * 1.2, 1.0))
        ax.axhline(0, color="0.4", lw=0.7, ls=":")
        ax.set_title(f"{rlo:g}<R<{rhi:g} kpc", fontsize=10)
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        if pi % ncols == ncols - 1:
            secax.set_ylabel(r"$|K_Z|$", fontsize=8)
        if pi % ncols == 0:
            ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        if pi >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$|Z|$ [kpc]")
    for pi in range(len(R_BINS), len(axes_flat)):
        axes_flat[pi].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes_flat[0].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outd / "original_panel_sigma_Z_assignments.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sigma_R_original_panel_grid(assignments: pd.DataFrame, outd: Path) -> None:
    if assignments.empty:
        write_status_figure(
            outd,
            "original_panel_grid_sigma_R.png",
            "No Original-Panel Sigma(R) Grid",
            ["No strict surface-density assignment rows were produced."],
        )
        return
    styles = {
        "thin": dict(color="#2166ac", marker="o", ls="-"),
        "thick": dict(color="#b2182b", marker="s", ls="--"),
        "halo": dict(color="#238b45", marker="^", ls="-."),
    }
    fig, axes = plt.subplots(len(Z_BINS), len(R_BINS), figsize=(24, 12), sharex=False, sharey=False)
    global_vals = assignments.loc[assignments["accepted_nonnegative"].astype(bool), "Sigma_median_Msun_pc2"].dropna()
    ymax_global = max(float(global_vals.quantile(0.95)) * 1.35, 1.0) if len(global_vals) else 1.0
    dense_rows = []
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            ax = axes[iz, ir]
            sub = assignments[(assignments["panel_iz"] == iz) & (assignments["panel_ir"] == ir)].copy()
            if sub.empty:
                ax.text(0.5, 0.5, "no Sigma rows", transform=ax.transAxes, ha="center", va="center", fontsize=7)
            else:
                for label, style in styles.items():
                    acc = sub[
                        (sub["population"] == label)
                        & sub["accepted_nonnegative"].astype(bool)
                        & np.isfinite(sub["Sigma_median_Msun_pc2"])
                    ].sort_values("R_mid")
                    if acc.empty:
                        continue
                    dense_n = max(
                        PANEL_GRID_MIN_DENSE_SAMPLES,
                        int(np.ceil((rhi - rlo) * PANEL_GRID_DENSE_SAMPLES_PER_KPC)),
                    )
                    dense_r = np.linspace(rlo, rhi, dense_n)
                    if len(acc) >= 2:
                        src_r = acc["R_mid"].to_numpy(float)
                        order = np.argsort(src_r)
                        src_r = src_r[order]
                        dense_med = np.interp(dense_r, src_r, acc["Sigma_median_Msun_pc2"].to_numpy(float)[order])
                        dense_lo = np.interp(dense_r, src_r, acc["Sigma_p16_Msun_pc2"].to_numpy(float)[order])
                        dense_hi = np.interp(dense_r, src_r, acc["Sigma_p84_Msun_pc2"].to_numpy(float)[order])
                        dense_source = "dense interpolation between accepted Cheng R-bin rows"
                    else:
                        dense_med = np.full_like(dense_r, float(acc["Sigma_median_Msun_pc2"].iloc[0]), dtype=float)
                        dense_lo = np.full_like(dense_r, float(acc["Sigma_p16_Msun_pc2"].iloc[0]), dtype=float)
                        dense_hi = np.full_like(dense_r, float(acc["Sigma_p84_Msun_pc2"].iloc[0]), dtype=float)
                        dense_source = "single accepted Cheng R-bin row shown across this panel span"
                    ax.plot(
                        dense_r,
                        dense_med,
                        color=style["color"],
                        ls=style["ls"],
                        lw=1.15,
                        alpha=0.92,
                        label=label if iz == 0 and ir == 0 else None,
                    )
                    ax.scatter(
                        dense_r,
                        dense_med,
                        color=style["color"],
                        marker=".",
                        s=4,
                        alpha=0.50,
                        linewidths=0,
                        zorder=3,
                    )
                    ax.fill_between(dense_r, dense_lo, dense_hi, color=style["color"], alpha=0.08, linewidth=0)
                    for rv, sv, sl, sh in zip(dense_r, dense_med, dense_lo, dense_hi):
                        dense_rows.append(
                            {
                                "panel_iz": iz,
                                "panel_ir": ir,
                                "R_min_kpc": rlo,
                                "R_max_kpc": rhi,
                                "absZ_min_kpc": zlo,
                                "absZ_max_kpc": zhi,
                                "population": label,
                                "R_dense_kpc": float(rv),
                                "Sigma_dense_Msun_pc2": float(sv),
                                "Sigma_dense_p16_Msun_pc2": float(sl),
                                "Sigma_dense_p84_Msun_pc2": float(sh),
                                "Kz_dense_km2_s2_kpc": float(sdc.s2k(sv)),
                                "source": dense_source,
                            }
                        )
                    ax.errorbar(
                        acc["R_mid"],
                        acc["Sigma_median_Msun_pc2"],
                        yerr=[
                            acc["Sigma_median_Msun_pc2"] - acc["Sigma_p16_Msun_pc2"],
                            acc["Sigma_p84_Msun_pc2"] - acc["Sigma_median_Msun_pc2"],
                        ],
                        fmt=style["marker"],
                        ls="",
                        color=style["color"],
                        ms=3.5,
                        lw=0.8,
                        capsize=1.5,
                    )
                rej = sub[~sub["accepted_nonnegative"].astype(bool)]
                if not rej.empty:
                    ax.scatter(rej["R_mid"], rej["Sigma_raw_median_Msun_pc2"], marker="x", color="0.35", s=14, alpha=0.65)
            ax.axvspan(rlo, rhi, color="0.88", alpha=0.45)
            ax.axhline(0, color="0.45", lw=0.6, ls=":")
            ax.set_xlim(max(0, rlo - 1.0), rhi + 1.0)
            ax.set_ylim(0, ymax_global)
            ax.set_title(f"{rlo:g}<R<{rhi:g}\n{zlo:g}<|Z|<{zhi:g}", fontsize=8)
            ax.tick_params(direction="in", top=True, right=True, labelsize=7)
            secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
            secax.tick_params(labelsize=6, direction="in")
            if ir == len(R_BINS) - 1:
                secax.set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]", fontsize=7)
            if ir == 0:
                ax.set_ylabel(r"$\Sigma$")
            if iz == len(Z_BINS) - 1:
                ax.set_xlabel(r"$R$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0, 0].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.suptitle("Original combined R-|Z| panel grid: assigned accepted Sigma(R); right axes show |K_Z|", y=0.995, fontsize=14)
    fig.tight_layout()
    fig.savefig(outd / "original_panel_grid_sigma_R.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(dense_rows).to_csv(outd / "original_panel_grid_sigma_R_dense_interpolated_rows.csv", index=False)


def data_subset(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "R": data["R"][mask].astype(np.float64),
        "Z": data["Z"][mask].astype(np.float64),
        "vR": data["vR"][mask].astype(np.float64),
        "vZ": data["vZ"][mask].astype(np.float64),
        "vp": data["vp"][mask].astype(np.float64),
    }


def fit_effective_tracer_scales(
    data: dict[str, np.ndarray],
    mask: np.ndarray,
    label: str,
    outd: Path,
    r_min: float,
    r_max: float,
    z_max: float,
    r_bin_width: float = 1.0,
    z_bin_width: float = 0.5,
    min_count: int = 50,
) -> tuple[float, float]:
    """Fit double-exponential tracer gradients from selected counts.

    This is used only for the L1-left halo diagnostic, because Cheng's paper
    defines disk tracer scale lengths but not a halo tracer for Eq. (1).
    """
    rr = np.asarray(data["R"][mask], dtype=float)
    zz = np.abs(np.asarray(data["Z"][mask], dtype=float))
    ok = np.isfinite(rr) & np.isfinite(zz) & (rr >= r_min) & (rr < r_max) & (zz >= 0.0) & (zz < z_max)
    rr = rr[ok]
    zz = zz[ok]
    rows = []
    for rl in np.arange(r_min, r_max, r_bin_width):
        rh = min(float(rl + r_bin_width), float(r_max))
        for zl in np.arange(0.0, z_max, z_bin_width):
            zh = min(float(zl + z_bin_width), float(z_max))
            cell = (rr >= rl) & (rr < rh) & (zz >= zl) & (zz < zh)
            n = int(cell.sum())
            if n < min_count:
                continue
            volume = np.pi * (rh ** 2 - rl ** 2) * (zh - zl) * 2.0
            rows.append(
                {
                    "population": label,
                    "R_mid": 0.5 * (rl + rh),
                    "Z_abs_mid": 0.5 * (zl + zh),
                    "N": n,
                    "volume_kpc3": volume,
                    "ln_number_density": np.log(n / volume),
                }
            )
    bins = pd.DataFrame(rows)
    bins.to_csv(outd / f"strict_combined_{label}_effective_tracer_density_bins.csv", index=False)
    if len(bins) < 6:
        raise RuntimeError(f"{label}: too few count bins to fit effective tracer gradients")
    x = np.column_stack(
        [
            np.ones(len(bins)),
            bins["R_mid"].to_numpy(float) - PAPER_R0_KPC,
            bins["Z_abs_mid"].to_numpy(float),
        ]
    )
    y = bins["ln_number_density"].to_numpy(float)
    w = np.sqrt(bins["N"].to_numpy(float))
    coeff, _, _, _ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    _a, b_r, b_z = [float(v) for v in coeff]
    if not (np.isfinite(b_r) and b_r < 0.0 and np.isfinite(b_z) and b_z < 0.0):
        raise RuntimeError(
            f"{label}: effective tracer gradient fit is nonphysical "
            f"(dlnnu/dR={b_r}, dlnnu/dZ={b_z}); not using fallback values"
        )
    h_r = -1.0 / b_r
    h_z = -1.0 / b_z
    pd.DataFrame(
        [
            {
                "population": label,
                "n_density_bins": int(len(bins)),
                "dlnnu_dR": b_r,
                "dlnnu_dZ": b_z,
                "hR_kpc": h_r,
                "hZ_kpc": h_z,
                "min_count_per_density_bin": int(min_count),
                "R_min_kpc": float(r_min),
                "R_max_kpc": float(r_max),
                "Z_abs_max_kpc": float(z_max),
            }
        ]
    ).to_csv(outd / f"strict_combined_{label}_effective_tracer_scales.csv", index=False)
    return h_z, h_r


def save_public_profile(profile: pd.DataFrame, out_path: Path) -> None:
    if profile.empty:
        pd.DataFrame().to_csv(out_path, index=False)
        return
    pd.DataFrame(
        {
            "R_mid": profile.rm,
            "Z_median": profile.zm,
            "Z_p16": profile.z16,
            "Z_p84": profile.z84,
            "sigma_R": profile.sR,
            "sigma_R_err": profile.sRe,
            "sigma_phi": profile.sp,
            "sigma_phi_err": profile.spe,
            "sigma_Z": profile.sZ,
            "sigma_Z_err": profile.sZe,
            "sigma_RZ2": profile.crz,
            "sigma_RZ2_err": profile.crze,
        }
    ).to_csv(out_path, index=False)


def hsigma_target_coverage(profile: pd.DataFrame, z_targets: list[float], label: str) -> pd.DataFrame:
    rows = []
    total = 0
    for zt in z_targets:
        pts = sdc.crz_at_z(profile, zt)
        total += len(pts)
        if pts:
            r_values = ",".join(f"{float(p[0]):.1f}" for p in pts)
        else:
            r_values = ""
        rows.append(
            {
                "population": label,
                "z_target": float(zt),
                "n_R_points": int(len(pts)),
                "R_mid_values": r_values,
            }
        )
    rows.append(
        {
            "population": label,
            "z_target": np.nan,
            "n_R_points": int(total),
            "R_mid_values": "TOTAL",
        }
    )
    return pd.DataFrame(rows)


def hsigma_model_points_from_odd_linear_fits(
    fits: pd.DataFrame,
    z_targets: list[float],
    label: str,
) -> pd.DataFrame:
    """Evaluate fixed-Z sigma_RZ^2 points from the fitted odd-linear relation.

    Cheng's Jeans implementation fits sigma_RZ^2(Z) = m(R) Z through the
    mid-plane in every R bin. These points use that fitted relation only; the
    chemical population selection itself is unchanged.
    """
    rows = []
    if fits.empty:
        return pd.DataFrame(
            columns=[
                "population",
                "R_mid",
                "R_min_kpc",
                "R_max_kpc",
                "z_target",
                "sigma_RZ2_model",
                "sigma_RZ2_err",
                "m_sigma_RZ2_per_kpc",
                "m_sigma_RZ2_per_kpc_err",
                "source",
            ]
        )
    for _, row in fits.sort_values("rm").iterrows():
        m = float(row["m"])
        me = float(row["me"])
        if not (np.isfinite(m) and np.isfinite(me) and me > 0.0):
            continue
        for zt in z_targets:
            zt = float(zt)
            rows.append(
                {
                    "population": label,
                    "R_mid": float(row["rm"]),
                    "R_min_kpc": float(row["rl"]),
                    "R_max_kpc": float(row["rh"]),
                    "z_target": zt,
                    "sigma_RZ2_model": m * zt,
                    "sigma_RZ2_err": abs(zt) * me,
                    "m_sigma_RZ2_per_kpc": m,
                    "m_sigma_RZ2_per_kpc_err": me,
                    "source": "sigma_RZ2=m*Z from per-R odd-linear fit",
                }
            )
    return pd.DataFrame(rows)


def fit_hsigma_from_point_table(points: pd.DataFrame, min_points: int = 6) -> tuple[float, float, dict]:
    if points.empty or len(points) < min_points:
        return np.nan, np.nan, {}
    R = points["R_mid"].to_numpy(float)
    y = points["sigma_RZ2_model"].to_numpy(float)
    e = points["sigma_RZ2_err"].to_numpy(float)
    zv = points["z_target"].to_numpy(float)
    ok = np.isfinite(R) & np.isfinite(y) & np.isfinite(e) & (e > 0.0) & np.isfinite(zv)
    if int(ok.sum()) < min_points:
        return np.nan, np.nan, {}
    return sdc._fit_hs_xy(R[ok], y[ok], e[ok], zv[ok], Rref=8.0)


def surface_curves(fits: pd.DataFrame, pp: dict, z_eval: np.ndarray, nmc: int, rng) -> pd.DataFrame:
    columns = [
        "R_mid",
        "Z_abs",
        "Sigma_median_Msun_pc2",
        "Sigma_p16_Msun_pc2",
        "Sigma_p84_Msun_pc2",
        "Sigma_raw_median_Msun_pc2",
        "Sigma_raw_p16_Msun_pc2",
        "Sigma_raw_p84_Msun_pc2",
        "Sigma_central_Msun_pc2",
        "accepted_nonnegative",
        "reject_reason",
        "SHM_Msun_pc2",
        "Sigma_minus_SHM_Msun_pc2",
        "Sigma_over_SHM",
        "Sigma_raw_minus_SHM_Msun_pc2",
        "Sigma_raw_over_SHM",
        "Kz_km2_s2_kpc",
        "Kz_raw_km2_s2_kpc",
        "SHM_Kz_km2_s2_kpc",
        "SHM_inside_raw_1sigma",
        "SHM_inside_accepted_1sigma",
        "jeans_den_term_Msun_pc2",
        "jeans_grad_term_Msun_pc2",
        "jeans_tilt_term_Msun_pc2",
        "jeans_bracket",
        "sigma_Z_model_kms",
        "sigma_RZ2_model_km2s2",
        "tilt_coefficient",
        "h_sigma_kpc",
        "h_sigma_err_kpc",
        "hZ_kpc",
        "hR_kpc",
    ]
    rows = []
    for _, row in fits.sort_values("rm").iterrows():
        med, lo, hi = sdc.mc_sig(row, pp, z_eval, nmc=nmc, rng=rng)
        central, comps = sdc.sig_from_row(row, pp, z_eval)
        shm = np.array([sdc.shm_sigma(float(row.rm), [float(z)])[0] for z in z_eval], dtype=float)
        for zi, z in enumerate(z_eval):
            raw_med = float(med[zi])
            raw_lo = float(lo[zi])
            raw_hi = float(hi[zi])
            central_sig = float(central[zi])
            if not (np.isfinite(raw_med) and np.isfinite(raw_lo) and np.isfinite(raw_hi)):
                reason = SURFACE_REJECT_NONFINITE
            elif raw_med < 0.0:
                reason = SURFACE_REJECT_NEGATIVE_MEDIAN
            elif raw_lo < 0.0:
                reason = SURFACE_REJECT_NEGATIVE_BAND
            else:
                reason = SURFACE_REJECT_ACCEPTED
            accepted = reason == SURFACE_REJECT_ACCEPTED
            shm_val = float(shm[zi])
            raw_minus_shm = raw_med - shm_val if np.isfinite(raw_med) and np.isfinite(shm_val) else np.nan
            raw_over_shm = raw_med / shm_val if np.isfinite(raw_med) and np.isfinite(shm_val) and shm_val > 0 else np.nan
            accepted_med = raw_med if accepted else np.nan
            accepted_lo = raw_lo if accepted else np.nan
            accepted_hi = raw_hi if accepted else np.nan
            accepted_minus_shm = (
                accepted_med - shm_val if accepted and np.isfinite(shm_val) else np.nan
            )
            accepted_over_shm = (
                accepted_med / shm_val if accepted and np.isfinite(shm_val) and shm_val > 0 else np.nan
            )
            kz = float(sdc.s2k(accepted_med)) if accepted and np.isfinite(accepted_med) else np.nan
            kz_raw = float(sdc.s2k(raw_med)) if np.isfinite(raw_med) else np.nan
            shm_kz = float(sdc.s2k(shm_val)) if np.isfinite(shm_val) else np.nan
            shm_inside_raw = bool(
                np.isfinite(raw_lo) and np.isfinite(raw_hi) and np.isfinite(shm_val) and raw_lo <= shm_val <= raw_hi
            )
            shm_inside_accepted = bool(accepted and shm_inside_raw)
            rows.append(
                {
                    "R_mid": float(row.rm),
                    "Z_abs": float(z),
                    "Sigma_median_Msun_pc2": accepted_med,
                    "Sigma_p16_Msun_pc2": accepted_lo,
                    "Sigma_p84_Msun_pc2": accepted_hi,
                    "Sigma_raw_median_Msun_pc2": raw_med,
                    "Sigma_raw_p16_Msun_pc2": raw_lo,
                    "Sigma_raw_p84_Msun_pc2": raw_hi,
                    "Sigma_central_Msun_pc2": central_sig,
                    "accepted_nonnegative": accepted,
                    "reject_reason": reason,
                    "SHM_Msun_pc2": shm_val,
                    "Sigma_minus_SHM_Msun_pc2": accepted_minus_shm,
                    "Sigma_over_SHM": accepted_over_shm,
                    "Sigma_raw_minus_SHM_Msun_pc2": raw_minus_shm,
                    "Sigma_raw_over_SHM": raw_over_shm,
                    "Kz_km2_s2_kpc": kz,
                    "Kz_raw_km2_s2_kpc": kz_raw,
                    "SHM_Kz_km2_s2_kpc": shm_kz,
                    "SHM_inside_raw_1sigma": shm_inside_raw,
                    "SHM_inside_accepted_1sigma": shm_inside_accepted,
                    "jeans_den_term_Msun_pc2": float(comps["den_term"][zi]),
                    "jeans_grad_term_Msun_pc2": float(comps["grad_term"][zi]),
                    "jeans_tilt_term_Msun_pc2": float(comps["tilt_term"][zi]),
                    "jeans_bracket": float(comps["bracket"][zi]),
                    "sigma_Z_model_kms": float(comps["sz"][zi]),
                    "sigma_RZ2_model_km2s2": float(comps["srz2"][zi]),
                    "tilt_coefficient": float(comps["tc"]),
                    "h_sigma_kpc": float(pp["hs"]),
                    "h_sigma_err_kpc": float(pp["hs_e"]),
                    "hZ_kpc": float(pp["hZ"]),
                    "hR_kpc": float(pp["hR"]),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def run_population(
    data: dict[str, np.ndarray],
    pop_code: np.uint8,
    label: str,
    pp_template: dict,
    z_targets_hsigma: list[float],
    args,
    rng,
) -> dict[str, object]:
    mask = data["pop"] == pop_code
    n = int(mask.sum())
    pp = dict(pp_template)
    out_prefix = f"strict_combined_{label}"
    summary = {
        "population": label,
        "n_points": n,
        "group_size": int(pp["ng"]),
        "status": "not_started",
        "h_sigma_kpc": np.nan,
        "h_sigma_err_kpc": np.nan,
        "h_sigma_method": "",
        "n_profile_groups": 0,
        "n_surface_R_bins": 0,
        "h_sigma_target_points": 0,
        "h_sigma_target_points_direct": 0,
        "h_sigma_target_points_model": 0,
        "direct_h_sigma_kpc": np.nan,
        "direct_h_sigma_err_kpc": np.nan,
        "direct_h_sigma_boundary_hit": False,
        "model_h_sigma_kpc": np.nan,
        "model_h_sigma_err_kpc": np.nan,
        "n_surface_rows": 0,
        "n_accepted_nonnegative_surface_rows": 0,
        "n_rejected_nonphysical_surface_rows": 0,
        "n_rejected_negative_median_surface_rows": 0,
        "n_rejected_uncertainty_crosses_negative_surface_rows": 0,
        "n_rejected_nonfinite_surface_rows": 0,
        "n_shm_inside_accepted_1sigma_surface_rows": 0,
    }
    if n < int(pp["ng"]):
        summary["status"] = "too_few_points_for_one_group"
        return {"summary": summary, "profile": pd.DataFrame(), "fits": pd.DataFrame(), "surface": pd.DataFrame()}

    print(f"[analysis] {label}: {n:,} points, group_size={pp['ng']}", flush=True)
    dat = data_subset(data, mask)
    profile = sdc.cprof(
        dat,
        ng=int(pp["ng"]),
        rl0=args.profile_r_min,
        rl1=args.profile_r_max,
        rbw=args.r_bin_width,
        nb=args.nboot,
        rng=rng,
        label=f"strict-{label}",
    )
    profile.to_csv(args.output_dir / f"{out_prefix}_cprof_internal.csv", index=False)
    save_public_profile(profile, args.output_dir / f"{out_prefix}_velocity_dispersion_profiles.csv")
    summary["n_profile_groups"] = int(len(profile))
    if profile.empty:
        summary["status"] = "empty_cprof"
        return {"summary": summary, "profile": profile, "fits": pd.DataFrame(), "surface": pd.DataFrame()}

    coverage = hsigma_target_coverage(profile, z_targets_hsigma, label)
    coverage.to_csv(args.output_dir / f"{out_prefix}_hsigma_target_coverage.csv", index=False)
    n_direct = int(coverage.loc[coverage["R_mid_values"] == "TOTAL", "n_R_points"].iloc[0])
    summary["h_sigma_target_points_direct"] = n_direct
    hs_direct, hs_direct_e, amps_direct = sdc.fit_hs(profile, zts=z_targets_hsigma)
    summary["direct_h_sigma_kpc"] = float(hs_direct) if np.isfinite(hs_direct) else np.nan
    summary["direct_h_sigma_err_kpc"] = float(hs_direct_e) if np.isfinite(hs_direct_e) else np.nan
    summary["direct_h_sigma_boundary_hit"] = bool(np.isfinite(hs_direct) and hs_direct >= HSIGMA_DIRECT_UPPER_BOUND_GUARD)

    fits, model_grid = sdc.fit_bins(
        profile,
        zmx=float(pp["zmax"]),
        robust=not args.no_robust_fit,
        clip=args.fit_clip,
        sigz_model="linear",
        pp=pp,
        rl0=args.fit_r_min,
        rl1=args.fit_r_max,
    )
    fits.to_csv(args.output_dir / f"{out_prefix}_linear_fits.csv", index=False)
    model_grid.to_csv(args.output_dir / f"{out_prefix}_sigz_model_selection.csv", index=False)
    summary["n_surface_R_bins"] = int(len(fits))
    if fits.empty:
        summary["status"] = "no_R_bins_with_valid_sigma_fits"
        pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_summary.csv", index=False)
        pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_hsigma_fit.csv", index=False)
        return {
            "summary": summary,
            "profile": profile,
            "fits": fits,
            "model_grid": model_grid,
            "surface": pd.DataFrame(),
            "pp": pp,
            "hsigma_points": pd.DataFrame(),
            "hsigma_amps": {},
        }

    model_points = hsigma_model_points_from_odd_linear_fits(fits, z_targets_hsigma, label)
    model_points.to_csv(args.output_dir / f"{out_prefix}_hsigma_model_evaluated_points.csv", index=False)
    summary["h_sigma_target_points_model"] = int(len(model_points))
    hs_model, hs_model_e, amps_model = fit_hsigma_from_point_table(model_points)
    summary["model_h_sigma_kpc"] = float(hs_model) if np.isfinite(hs_model) else np.nan
    summary["model_h_sigma_err_kpc"] = float(hs_model_e) if np.isfinite(hs_model_e) else np.nan

    if np.isfinite(hs_direct) and hs_direct > 0.0 and not summary["direct_h_sigma_boundary_hit"]:
        hs = float(hs_direct)
        hs_e = float(hs_direct_e) if np.isfinite(hs_direct_e) and hs_direct_e > 0.0 else float(0.1 * hs)
        h_method = HSIGMA_DIRECT_PROFILE
        h_points = n_direct
        h_amps = amps_direct
    elif np.isfinite(hs_model) and hs_model > 0.0:
        hs = float(hs_model)
        hs_e = float(hs_model_e) if np.isfinite(hs_model_e) and hs_model_e > 0.0 else float(0.1 * hs)
        h_method = HSIGMA_ODD_LINEAR_MODEL
        h_points = int(len(model_points))
        h_amps = amps_model
    else:
        summary["status"] = "h_sigma_fit_failed_after_direct_and_odd_linear_model"
        summary["h_sigma_target_points"] = max(n_direct, int(len(model_points)))
        pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_hsigma_fit.csv", index=False)
        pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_summary.csv", index=False)
        return {
            "summary": summary,
            "profile": profile,
            "fits": fits,
            "model_grid": model_grid,
            "surface": pd.DataFrame(),
            "pp": pp,
            "hsigma_points": model_points,
            "hsigma_amps": {},
        }

    pp["hs"] = hs
    pp["hs_e"] = hs_e
    summary["h_sigma_kpc"] = pp["hs"]
    summary["h_sigma_err_kpc"] = pp["hs_e"]
    summary["h_sigma_method"] = h_method
    summary["h_sigma_target_points"] = h_points
    pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_hsigma_fit.csv", index=False)

    z_special = FIG8_Z[FIG8_Z <= pp["zmax"] + 1e-9]
    z_eval = np.unique(np.concatenate([np.linspace(0.01, pp["zmax"], args.z_grid), z_special]))
    surface = surface_curves(fits, pp, z_eval, args.nmc, rng)
    surface.to_csv(args.output_dir / f"{out_prefix}_surface_density_curves.csv", index=False)
    rejected = surface[~surface["accepted_nonnegative"]].copy()
    rejected.to_csv(args.output_dir / f"{out_prefix}_surface_density_rejected_nonphysical.csv", index=False)
    summary["status"] = "ok"
    summary["n_surface_rows"] = int(len(surface))
    summary["n_accepted_nonnegative_surface_rows"] = int(surface["accepted_nonnegative"].sum())
    summary["n_rejected_nonphysical_surface_rows"] = int((~surface["accepted_nonnegative"]).sum())
    summary["n_rejected_negative_median_surface_rows"] = int(
        (surface["reject_reason"] == SURFACE_REJECT_NEGATIVE_MEDIAN).sum()
    )
    summary["n_rejected_uncertainty_crosses_negative_surface_rows"] = int(
        (surface["reject_reason"] == SURFACE_REJECT_NEGATIVE_BAND).sum()
    )
    summary["n_rejected_nonfinite_surface_rows"] = int(
        (surface["reject_reason"] == SURFACE_REJECT_NONFINITE).sum()
    )
    summary["n_shm_inside_accepted_1sigma_surface_rows"] = int(
        surface["SHM_inside_accepted_1sigma"].sum()
    )
    pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / f"{out_prefix}_hsigma_fit.csv", index=False)
    return {
        "summary": summary,
        "profile": profile,
        "fits": fits,
        "model_grid": model_grid,
        "surface": surface,
        "pp": pp,
        "hsigma_points": model_points,
        "hsigma_amps": h_amps,
    }


def nearest_surface_rows(surface: pd.DataFrame, z_value: float) -> pd.DataFrame:
    if surface.empty:
        return surface
    zvals = np.array(sorted(surface["Z_abs"].unique()))
    z_use = zvals[np.argmin(np.abs(zvals - z_value))]
    return surface[np.isclose(surface["Z_abs"], z_use)].copy()


def ordered_population_labels(results: dict[str, dict[str, object]]) -> list[str]:
    return [label for label in ("thin", "thick", "halo") if label in results]


def plot_fig8(results: dict[str, dict[str, object]], outd: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    styles = {
        "thin": dict(color="#2166ac", ls="-", marker="o", label="strict panel-L2 thin"),
        "thick": dict(color="#b2182b", ls="--", marker="s", label="strict panel-L2 thick"),
        "halo": dict(color="#238b45", ls="-.", marker="^", label="strict L1-left halo"),
    }
    for ax, z_value in zip(axes, FIG8_Z):
        y_values = []
        for label in ordered_population_labels(results):
            result = results.get(label, {})
            surface = result.get("surface", pd.DataFrame())
            pp = result.get("pp", None)
            if surface.empty or pp is None or z_value > pp["zmax"] + 1e-9:
                continue
            sub = nearest_surface_rows(surface, z_value).sort_values("R_mid")
            if sub.empty:
                continue
            style = styles[label]
            x = sub["R_mid"].to_numpy(float)
            y = sub["Sigma_median_Msun_pc2"].to_numpy(float)
            lo = sub["Sigma_p16_Msun_pc2"].to_numpy(float)
            hi = sub["Sigma_p84_Msun_pc2"].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
            if not np.any(ok):
                continue
            x, y, lo, hi = x[ok], y[ok], lo[ok], hi[ok]
            y_values.extend(y[np.isfinite(y)].tolist())
            ax.plot(x, y, color=style["color"], ls=style["ls"], marker=style["marker"], lw=2.1, ms=5,
                    label=style["label"])
            ax.fill_between(x, lo, hi, color=style["color"], alpha=0.18)
        plotted_x = []
        for label in ordered_population_labels(results):
            result = results.get(label, {})
            surface = result.get("surface", pd.DataFrame())
            pp = result.get("pp", None)
            if surface.empty or pp is None or z_value > pp["zmax"] + 1e-9:
                continue
            sub = nearest_surface_rows(surface, z_value)
            finite_y = np.isfinite(sub["Sigma_median_Msun_pc2"].to_numpy(float))
            plotted_x.extend(sub.loc[finite_y, "R_mid"].to_numpy(float).tolist())
        if plotted_x:
            rr_min = max(0.3, min(plotted_x) - 0.5)
            rr_max = max(plotted_x) + 0.5
        else:
            rr_min = 0.5
            rr_max = 25.0
        rr = np.linspace(rr_min, rr_max, 220)
        ax.plot(rr, [sdc.shm_sigma(r, [z_value])[0] for r in rr], color="firebrick", lw=1.8, label="SHM")
        ax.axhline(0.0, color="0.35", lw=0.8, ls=":")
        ax.text(0.03, 0.90, f"|Z|={z_value:.1f} kpc", transform=ax.transAxes, fontsize=11)
        ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        if y_values and min(y_values) < 0:
            ax.set_ylim(min(y_values) * 1.15, None)
        else:
            ax.set_ylim(bottom=0)
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        secax.set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]")
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, fontsize=9)
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig8_sigma_vs_R.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_all_sigma_overplot(results: dict[str, dict[str, object]], outd: Path) -> None:
    labels = ordered_population_labels(results)
    if not labels:
        write_status_figure(
            outd,
            "all_strict_panel_l2_sigma_R_overplot.png",
            "No Sigma(R) Overplot",
            ["No population surface-density rows were produced."],
        )
        return
    fig, axes = plt.subplots(len(labels), 1, figsize=(10.5, max(5.0, 4.2 * len(labels))), sharex=True)
    if len(labels) == 1:
        axes = [axes]
    cmap = plt.cm.viridis
    markers = {"thin": "o", "thick": "s", "halo": "^"}
    linestyles = {"thin": "-", "thick": "--", "halo": "-."}
    for ax, label in zip(axes, labels):
        result = results.get(label, {})
        surface = result.get("surface", pd.DataFrame())
        pp = result.get("pp", None)
        if surface.empty or pp is None:
            continue
        z_values = OVERPLOT_Z[OVERPLOT_Z <= float(pp["zmax"]) + 1e-9]
        if z_values.size == 0:
            continue
        colors = cmap(np.linspace(0.08, 0.92, len(z_values)))
        for z_value, color in zip(z_values, colors):
            if z_value > pp["zmax"] + 1e-9:
                continue
            sub = nearest_surface_rows(surface, z_value).sort_values("R_mid")
            if sub.empty:
                continue
            sub = sub[
                np.isfinite(sub["Sigma_median_Msun_pc2"])
                & np.isfinite(sub["Sigma_p16_Msun_pc2"])
                & np.isfinite(sub["Sigma_p84_Msun_pc2"])
            ].copy()
            if sub.empty:
                continue
            z_used = float(sub["Z_abs"].dropna().iloc[0])
            rr = np.linspace(float(sub["R_mid"].min()), float(sub["R_mid"].max()), 240)
            shm = np.array([sdc.shm_sigma(float(r), [z_used])[0] for r in rr], dtype=float)
            ok_shm = np.isfinite(rr) & np.isfinite(shm)
            if np.any(ok_shm):
                # SHM for this |Z| slice, drawn in the SAME colour as the slice
                # (dotted, so it stays distinct from the measured Sigma line).
                ax.plot(
                    rr[ok_shm],
                    shm[ok_shm],
                    color=color,
                    ls=":",
                    lw=1.7,
                    alpha=0.95,
                    zorder=2,
                )
            ax.errorbar(
                sub["R_mid"],
                sub["Sigma_median_Msun_pc2"],
                yerr=[
                    sub["Sigma_median_Msun_pc2"] - sub["Sigma_p16_Msun_pc2"],
                    sub["Sigma_p84_Msun_pc2"] - sub["Sigma_median_Msun_pc2"],
                ],
                fmt=markers[label],
                ls=linestyles[label],
                color=color,
                capsize=3,
                lw=1.5,
                ms=5,
                label=f"|Z|={z_value:.1f}",
                zorder=3,
            )
        ax.axhline(0.0, color="0.35", lw=0.8, ls=":")
        ax.set_ylabel(r"$\Sigma(R,|Z|)$")
        ax.set_title(f"{label}: accepted non-negative Sigma(R) by |Z| slice with SHM")
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        secax.set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]")
        ax.tick_params(direction="in", top=True, right=True)
        handles, leg_labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="0.4", ls=":", lw=1.7))
        leg_labels.append("SHM (dotted, same colour as its slice)")
        ax.legend(handles, leg_labels, frameon=False, ncol=3, fontsize=8)
        ax.set_ylim(bottom=0)
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "all_strict_panel_l2_sigma_R_overplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_fig8_points(results: dict[str, dict[str, object]], outd: Path) -> None:
    columns = [
        "population",
        "fig8_Z_abs",
        "R_mid",
        "Sigma_median_Msun_pc2",
        "Sigma_p16_Msun_pc2",
        "Sigma_p84_Msun_pc2",
        "Sigma_raw_median_Msun_pc2",
        "Sigma_raw_p16_Msun_pc2",
        "Sigma_raw_p84_Msun_pc2",
        "accepted_nonnegative",
        "reject_reason",
        "SHM_Msun_pc2",
        "Sigma_minus_SHM_Msun_pc2",
        "Sigma_over_SHM",
        "Sigma_raw_minus_SHM_Msun_pc2",
        "Sigma_raw_over_SHM",
        "Kz_km2_s2_kpc",
        "Kz_raw_km2_s2_kpc",
        "SHM_Kz_km2_s2_kpc",
        "SHM_inside_raw_1sigma",
        "SHM_inside_accepted_1sigma",
        "jeans_den_term_Msun_pc2",
        "jeans_grad_term_Msun_pc2",
        "jeans_tilt_term_Msun_pc2",
        "jeans_bracket",
        "sigma_Z_model_kms",
        "sigma_RZ2_model_km2s2",
        "tilt_coefficient",
        "panel_R_min_kpc",
        "panel_R_max_kpc",
        "panel_absZ_min_kpc",
        "panel_absZ_max_kpc",
        "h_sigma_kpc",
        "hZ_kpc",
        "hR_kpc",
    ]
    rows = []
    rejected_rows = []
    for label, result in results.items():
        surface = result.get("surface", pd.DataFrame())
        pp = result.get("pp", None)
        if surface.empty or pp is None:
            continue
        for z_value in FIG8_Z:
            if z_value > pp["zmax"] + 1e-9:
                continue
            sub = nearest_surface_rows(surface, z_value)
            for _, row in sub.iterrows():
                ir = next((i for i, (lo, hi) in enumerate(R_BINS) if lo <= row["R_mid"] < hi), -1)
                iz = next((i for i, (lo, hi) in enumerate(Z_BINS) if lo <= z_value <= hi), -1)
                rlo, rhi = R_BINS[ir] if ir >= 0 else (np.nan, np.nan)
                zlo, zhi = Z_BINS[iz] if iz >= 0 else (np.nan, np.nan)
                record = {
                    "population": label,
                    "fig8_Z_abs": float(z_value),
                    "R_mid": float(row["R_mid"]),
                    "Sigma_median_Msun_pc2": float(row["Sigma_median_Msun_pc2"])
                    if np.isfinite(row["Sigma_median_Msun_pc2"]) else np.nan,
                    "Sigma_p16_Msun_pc2": float(row["Sigma_p16_Msun_pc2"])
                    if np.isfinite(row["Sigma_p16_Msun_pc2"]) else np.nan,
                    "Sigma_p84_Msun_pc2": float(row["Sigma_p84_Msun_pc2"])
                    if np.isfinite(row["Sigma_p84_Msun_pc2"]) else np.nan,
                    "Sigma_raw_median_Msun_pc2": float(row["Sigma_raw_median_Msun_pc2"]),
                    "Sigma_raw_p16_Msun_pc2": float(row["Sigma_raw_p16_Msun_pc2"]),
                    "Sigma_raw_p84_Msun_pc2": float(row["Sigma_raw_p84_Msun_pc2"]),
                    "accepted_nonnegative": bool(row["accepted_nonnegative"]),
                    "reject_reason": str(row["reject_reason"]),
                    "SHM_Msun_pc2": float(row["SHM_Msun_pc2"]),
                    "Sigma_minus_SHM_Msun_pc2": float(row["Sigma_minus_SHM_Msun_pc2"])
                    if np.isfinite(row["Sigma_minus_SHM_Msun_pc2"]) else np.nan,
                    "Sigma_over_SHM": float(row["Sigma_over_SHM"])
                    if np.isfinite(row["Sigma_over_SHM"]) else np.nan,
                    "Sigma_raw_minus_SHM_Msun_pc2": float(row["Sigma_raw_minus_SHM_Msun_pc2"]),
                    "Sigma_raw_over_SHM": float(row["Sigma_raw_over_SHM"]),
                    "Kz_km2_s2_kpc": float(row["Kz_km2_s2_kpc"])
                    if np.isfinite(row["Kz_km2_s2_kpc"]) else np.nan,
                    "Kz_raw_km2_s2_kpc": float(row["Kz_raw_km2_s2_kpc"]),
                    "SHM_Kz_km2_s2_kpc": float(row["SHM_Kz_km2_s2_kpc"]),
                    "SHM_inside_raw_1sigma": bool(row["SHM_inside_raw_1sigma"]),
                    "SHM_inside_accepted_1sigma": bool(row["SHM_inside_accepted_1sigma"]),
                    "jeans_den_term_Msun_pc2": float(row["jeans_den_term_Msun_pc2"]),
                    "jeans_grad_term_Msun_pc2": float(row["jeans_grad_term_Msun_pc2"]),
                    "jeans_tilt_term_Msun_pc2": float(row["jeans_tilt_term_Msun_pc2"]),
                    "jeans_bracket": float(row["jeans_bracket"]),
                    "sigma_Z_model_kms": float(row["sigma_Z_model_kms"]),
                    "sigma_RZ2_model_km2s2": float(row["sigma_RZ2_model_km2s2"]),
                    "tilt_coefficient": float(row["tilt_coefficient"]),
                    "panel_R_min_kpc": rlo,
                    "panel_R_max_kpc": rhi,
                    "panel_absZ_min_kpc": zlo,
                    "panel_absZ_max_kpc": zhi,
                    "h_sigma_kpc": float(row["h_sigma_kpc"]),
                    "hZ_kpc": float(row["hZ_kpc"]),
                    "hR_kpc": float(row["hR_kpc"]),
                }
                if record["accepted_nonnegative"]:
                    rows.append(record)
                else:
                    rejected_rows.append(record)
    pd.DataFrame(rows, columns=columns).to_csv(outd / "fig8_sigma_R_points_strict_panel_l2.csv", index=False)
    pd.DataFrame(rejected_rows, columns=columns).to_csv(
        outd / "fig8_sigma_R_rejected_nonphysical_strict_panel_l2.csv", index=False
    )


def write_shm_comparison_summary(outd: Path) -> None:
    accepted_path = outd / "fig8_sigma_R_points_strict_panel_l2.csv"
    rejected_path = outd / "fig8_sigma_R_rejected_nonphysical_strict_panel_l2.csv"
    accepted = pd.read_csv(accepted_path) if accepted_path.exists() else pd.DataFrame()
    rejected = pd.read_csv(rejected_path) if rejected_path.exists() else pd.DataFrame()
    rows = []
    populations = sorted(set(accepted.get("population", [])) | set(rejected.get("population", [])))
    for population in populations:
        for z_value in FIG8_Z:
            acc = accepted[
                (accepted.get("population", pd.Series(dtype=str)) == population)
                & np.isclose(accepted.get("fig8_Z_abs", pd.Series(dtype=float)), z_value)
            ] if not accepted.empty else pd.DataFrame()
            rej = rejected[
                (rejected.get("population", pd.Series(dtype=str)) == population)
                & np.isclose(rejected.get("fig8_Z_abs", pd.Series(dtype=float)), z_value)
            ] if not rejected.empty else pd.DataFrame()
            rows.append(
                {
                    "population": population,
                    "fig8_Z_abs": float(z_value),
                    "n_accepted_nonnegative": int(len(acc)),
                    "n_rejected_nonphysical": int(len(rej)),
                    "n_SHM_inside_accepted_1sigma": int(acc["SHM_inside_accepted_1sigma"].sum())
                    if not acc.empty else 0,
                    "median_Sigma_over_SHM": float(np.nanmedian(acc["Sigma_over_SHM"]))
                    if not acc.empty else np.nan,
                    "min_Sigma_over_SHM": float(np.nanmin(acc["Sigma_over_SHM"]))
                    if not acc.empty else np.nan,
                    "max_Sigma_over_SHM": float(np.nanmax(acc["Sigma_over_SHM"]))
                    if not acc.empty else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(outd / "fig8_sigma_R_shm_comparison_summary.csv", index=False)


def write_status_figure(outd: Path, fname: str, title: str, lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axis("off")
    ax.text(0.5, 0.82, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.08, 0.62, "\n".join(lines), ha="left", va="top", fontsize=11, linespacing=1.45)
    fig.tight_layout()
    fig.savefig(outd / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig1_strict_chemical_plane(data: dict[str, np.ndarray], strict_l2, outd: Path) -> None:
    lims = DISPLAY_LIMITS[COMBINED_KEY]
    xlim = lims["xlim"]
    ylim = lims["ylim"]
    x = data["xchem"]
    y = data["ychem"]
    ok = np.isfinite(x) & np.isfinite(y)
    ok &= (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])

    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    ax.hist2d(
        x[ok],
        y[ok],
        bins=(520, 420),
        range=[xlim, ylim],
        cmap="viridis",
        norm=LogNorm(vmin=1),
    )
    xx = np.linspace(xlim[0], xlim[1], 300)
    ax.plot(xx, HALO_LINE_M * xx + HALO_LINE_B, color="magenta", lw=2.2, ls="-", label="L1")
    direct_labeled = False
    fallback_labeled = False
    for (_iz, _ir), (m, b, is_direct) in sorted(strict_l2.items()):
        yy = m * xx + b
        use = (yy >= ylim[0]) & (yy <= ylim[1])
        if not np.any(use):
            continue
        if is_direct:
            label = None if direct_labeled else "direct panel L2"
            direct_labeled = True
            ax.plot(xx[use], yy[use], color="magenta", lw=0.9, ls=":", alpha=0.30, label=label)
        else:
            label = None if fallback_labeled else "fallback panel L2 (cat. avg)"
            fallback_labeled = True
            ax.plot(xx[use], yy[use], color="orange", lw=0.9, ls=":", alpha=0.30, label=label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(combined_category().x_label)
    ax.set_ylabel(combined_category().y_label)
    ax.set_title("Figure 1-style strict clean combined chemical plane")
    ax.legend(frameon=False, loc="upper left")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(outd / "fig1_strict_chemical_plane_l1_l2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_strict_surface_grid(results: dict[str, dict[str, object]], outd: Path) -> None:
    r_values = sorted(
        {
            float(r)
            for result in results.values()
            for r in result.get("surface", pd.DataFrame()).get("R_mid", pd.Series(dtype=float)).dropna().unique()
        }
    )
    if not r_values:
        write_status_figure(
            outd,
            "fig7_sigma_grid.png",
            "Figure 7 Not Computable",
            ["No strict non-fallback surface-density curves were produced."],
        )
        return

    styles = {
        "thin": dict(color="#2166ac", ls="-", label="strict panel-L2 thin"),
        "thick": dict(color="#b2182b", ls="--", label="strict panel-L2 thick"),
        "halo": dict(color="#238b45", ls="-.", label="strict L1-left halo"),
    }
    nc = 2
    nr = int(np.ceil(len(r_values) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(12, max(3.5, nr * 3.2)), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    z_plot_max = max(
        float(result.get("pp", {}).get("zmax", 4.0))
        for result in results.values()
        if result.get("pp", None) is not None
    )
    for pi, rm in enumerate(r_values):
        ax = axes_flat[pi]
        panel_y = []
        for label in ordered_population_labels(results):
            result = results.get(label, {})
            surface = result.get("surface", pd.DataFrame())
            pp = result.get("pp", None)
            if surface.empty or pp is None:
                continue
            sub = surface[np.isclose(surface["R_mid"], rm)].sort_values("Z_abs").copy()
            sub = sub[np.isfinite(sub["Sigma_median_Msun_pc2"])]
            if sub.empty:
                continue
            style = styles[label]
            ax.plot(
                sub["Z_abs"],
                sub["Sigma_median_Msun_pc2"],
                color=style["color"],
                ls=style["ls"],
                lw=2.0,
                label=style["label"] if pi == 0 else None,
            )
            band = np.isfinite(sub["Sigma_p16_Msun_pc2"]) & np.isfinite(sub["Sigma_p84_Msun_pc2"])
            if band.any():
                ax.fill_between(
                    sub.loc[band, "Z_abs"],
                    sub.loc[band, "Sigma_p16_Msun_pc2"],
                    sub.loc[band, "Sigma_p84_Msun_pc2"],
                    color=style["color"],
                    alpha=0.14,
                    linewidth=0,
                )
            panel_y.extend(sub["Sigma_median_Msun_pc2"].to_numpy(float).tolist())
        zr = np.linspace(0.01, z_plot_max, 250)
        ax.plot(zr, sdc.shm_sigma(rm, zr), color="firebrick", lw=1.5, label="SHM" if pi == 0 else None)
        ax.text(0.97, 0.91, f"R = {rm:.1f} kpc", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_xlim(0, z_plot_max)
        ax.set_ylim(bottom=0)
        if panel_y:
            ax.set_ylim(0, max(max(panel_y) * 1.15, ax.get_ylim()[1]))
        ax.tick_params(direction="in", top=True, right=True)
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        if pi % nc == nc - 1:
            secax.set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]")
        if pi % nc == 0:
            ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        if pi >= (nr - 1) * nc:
            ax.set_xlabel(r"$|Z|$ [kpc]")
    for pi in range(len(r_values), len(axes_flat)):
        axes_flat[pi].axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes_flat[0].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outd / "fig7_sigma_grid.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def nearest_fit_row(result: dict[str, object], target_R: float = PAPER_R0_KPC):
    fits = result.get("fits", pd.DataFrame())
    if fits.empty:
        return None
    idx = int(np.argmin(np.abs(fits["rm"].to_numpy(float) - target_R)))
    return fits.iloc[idx]


def nearest_profile_sig2(profile: pd.DataFrame, target_R: float = PAPER_R0_KPC):
    if profile.empty or "rm" not in profile.columns:
        return np.array([]), np.array([]), np.array([]), np.nan
    rms = np.array(sorted(profile["rm"].dropna().unique()), dtype=float)
    if rms.size == 0:
        return np.array([]), np.array([]), np.array([]), np.nan
    rm = float(rms[np.argmin(np.abs(rms - target_R))])
    sub = profile[np.isclose(profile["rm"], rm)].copy()
    absz = np.abs(sub["zm"].to_numpy(float))
    sig2 = sub["sZ"].to_numpy(float) ** 2
    esig2 = 2.0 * sub["sZ"].to_numpy(float) * sub["sZe"].to_numpy(float)
    ok = np.isfinite(absz) & np.isfinite(sig2) & np.isfinite(esig2) & (sig2 > 0)
    order = np.argsort(absz[ok])
    return absz[ok][order], sig2[ok][order], esig2[ok][order], rm


def plot_fig5_hsigma_results(results: dict[str, dict[str, object]], outd: Path) -> None:
    specs = [
        ("thin", [-1.0, 1.0]),
        ("thick", [-2.0, -1.0, 1.0, 2.0]),
        ("halo", [-6.0, -4.0, -2.0, 2.0, 4.0, 6.0]),
    ]
    specs = [(label, zts) for label, zts in specs if label in results]
    fig, axes = plt.subplots(len(specs), 1, figsize=(9, max(5.0, 4.8 * len(specs))), sharex=True)
    if len(specs) == 1:
        axes = [axes]
    for ax, (label, z_targets) in zip(axes, specs):
        result = results.get(label, {})
        profile = result.get("profile", pd.DataFrame())
        summary = result.get("summary", {})
        amps = result.get("hsigma_amps", {})
        hs = float(summary.get("h_sigma_kpc", np.nan))
        he = float(summary.get("h_sigma_err_kpc", np.nan))
        method = str(summary.get("h_sigma_method", ""))
        cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(z_targets)))
        all_x = []
        for zt, col in zip(z_targets, cols):
            direct_pts = sdc.crz_at_z(profile, zt) if not profile.empty else []
            if direct_pts:
                Rp, yp, ep = zip(*direct_pts)
                ax.errorbar(
                    Rp,
                    yp,
                    yerr=ep,
                    fmt="o",
                    color=col,
                    mfc="white",
                    ms=5,
                    capsize=2.5,
                    lw=0.9,
                    elinewidth=0.9,
                    label=f"direct Z={zt:+g}",
                )
                all_x.extend([float(x) for x in Rp])
            model_points = result.get("hsigma_points", pd.DataFrame())
            model_z = model_points[np.isclose(model_points.get("z_target", pd.Series(dtype=float)), zt)] if not model_points.empty else pd.DataFrame()
            if not model_z.empty:
                ax.errorbar(
                    model_z["R_mid"],
                    model_z["sigma_RZ2_model"],
                    yerr=model_z["sigma_RZ2_err"],
                    fmt="x",
                    color=col,
                    ms=5,
                    capsize=2.0,
                    lw=0.8,
                    alpha=0.72,
                    label=f"odd-linear model Z={zt:+g}",
                )
                all_x.extend(model_z["R_mid"].to_numpy(float).tolist())
            if np.isfinite(hs) and float(zt) in amps and all_x:
                rr = np.linspace(min(all_x) - 0.5, max(all_x) + 0.5, 260)
                ax.plot(rr, amps[float(zt)] * np.exp(-(rr - 8.0) / hs), lw=2.0, color=col)
        ax.axhline(0, color="0.4", lw=1, ls=":")
        title = f"{label.capitalize()} disc"
        if np.isfinite(hs):
            title += f": h_sigma={hs:.2f}+/-{he:.2f} kpc"
        if method:
            title += f" ({method})"
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
        ax.tick_params(direction="in", top=True, right=True)
        ax.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig5_hsigma_fit.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# --- Effect of the three Cheng sigma_z(|Z|) assumptions on Sigma(|Z|) --------
# The vertical Jeans surface density depends on d(sigma_Z^2)/dz, so the assumed
# sigma_Z(|Z|) functional form changes the inferred Sigma(|Z|). The Cheng paper
# (App. A & B) uses: linear sigma_Z^2 = a|Z|+b (primary), quadratic
# sigma_Z^2 = kZ^2 + sigma0^2, and tanh sigma_Z = kz*tanh(|Z|/L) + s0.
SIGMAZ_MODELS = ("linear", "quadratic", "tanh")
SIGMAZ_FX_STYLE = {"linear": ("-", "#1f77b4"), "quadratic": ("--", "#ff7f0e"), "tanh": (":", "#2ca02c")}
SIGMAZ_POP_COLOR = {"thin": "#2166ac", "thick": "#b2182b", "halo": "#238b45"}


def sigmaz_model_fits(profile: pd.DataFrame, pp: dict) -> dict[str, pd.DataFrame]:
    """Per-R-bin Jeans fit tables under each sigma_Z(|Z|) model: {model: fits_df}."""
    out: dict[str, pd.DataFrame] = {}
    if profile is None or profile.empty or pp is None:
        return out
    for model in SIGMAZ_MODELS:
        try:
            fits, _ = sdc.fit_bins(
                profile, zmx=float(pp["zmax"]), robust=True, clip=3.0,
                sigz_model=model, pp=pp, rl0=0, rl1=25,
            )
            if fits is not None and not fits.empty:
                out[model] = fits
        except Exception:
            continue
    return out


def _sigma_band(fits: pd.DataFrame, pp: dict, r_target: float, z: np.ndarray,
                nmc: int = 200, rng=None) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Sigma(z) median + 16-84% MC band at the fitted R bin nearest r_target.

    sdc.mc_sig propagates the fit-parameter covariance of whichever sigma_Z model
    the row carries (linear / quadratic / tanh) plus the tracer-scale and h_sigma
    uncertainties, so every model gets its own error band.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    row = fits.iloc[int((fits["rm"] - r_target).abs().values.argmin())]
    med, lo, hi = sdc.mc_sig(row, pp, z, nmc=nmc, rng=rng)
    return (float(row["rm"]), np.asarray(med, dtype=float),
            np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))


def _surface_band(surface: pd.DataFrame, r_target: float):
    """Measured (linear MC) 16-84% Sigma(|Z|) band at the R bin nearest r_target."""
    if surface is None or surface.empty:
        return None
    rv = np.array(sorted(surface["R_mid"].dropna().unique()), dtype=float)
    if rv.size == 0:
        return None
    ru = float(rv[np.argmin(np.abs(rv - r_target))])
    sb = surface[
        np.isclose(surface["R_mid"], ru)
        & np.isfinite(surface["Sigma_p16_Msun_pc2"])
        & np.isfinite(surface["Sigma_p84_Msun_pc2"])
    ].sort_values("Z_abs")
    return sb if not sb.empty else None


def plot_population_sigmaz_model_effect(label: str, result: dict, pop_dir: Path) -> None:
    """Per-population: how Sigma(|Z|) changes under linear/quadratic/tanh sigma_Z,
    shown across several R bins. Saved as sigma_Z_model_comparison.png."""
    prof = result.get("profile", pd.DataFrame())
    pp = result.get("pp", None)
    if prof is None or prof.empty or pp is None:
        return
    mfits = sigmaz_model_fits(prof, pp)
    if not mfits:
        return
    zmax = float(pp["zmax"])
    z = np.linspace(0.02, zmax, 220)
    rng = np.random.default_rng(11)
    r_targets = [4.0, 6.0, 8.0, 10.0, 12.0]
    fig, axes = plt.subplots(1, len(r_targets), figsize=(3.5 * len(r_targets), 4.3), squeeze=False, sharex=True)
    for ax, rt in zip(axes[0], r_targets):
        r_used = rt
        for model in SIGMAZ_MODELS:
            if model not in mfits:
                continue
            ls, c = SIGMAZ_FX_STYLE[model]
            r_used, med, lo, hi = _sigma_band(mfits[model], pp, rt, z, rng=rng)
            ax.fill_between(z, lo, hi, color=c, alpha=0.18)
            ax.plot(z, med, ls=ls, color=c, lw=2.0, label=model)
        ax.plot(z, sdc.shm_sigma(r_used, z), color="firebrick", lw=1.4, label="SHM")
        ax.set_title(rf"$R\simeq{r_used:.1f}$ kpc", fontsize=9)
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_xlim(0, zmax)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
    axes[0][0].set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    axes[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle(rf"{label.capitalize()}: $\Sigma(|Z|)$ under linear / quadratic / tanh $\sigma_Z$ assumptions", y=1.0, fontsize=11)
    fig.tight_layout()
    fig.savefig(pop_dir / "sigma_Z_model_comparison.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_fig6_strict_nearest(results: dict[str, dict[str, object]], outd: Path) -> None:
    """Fig. 6: how Sigma(|Z|) at the solar radius changes under the three Cheng
    sigma_Z(|Z|) assumptions (linear / quadratic / tanh), one panel per population."""
    labels = ordered_population_labels(results)
    if not labels:
        write_status_figure(outd, "fig6_solar_sigma.png", "Figure 6 Not Computable",
                            ["No population surface-density results were produced."])
        return
    fig, axes = plt.subplots(1, len(labels), figsize=(5.8 * len(labels), 5.2), squeeze=False)
    for ax, label in zip(axes[0], labels):
        result = results.get(label, {})
        prof = result.get("profile", pd.DataFrame())
        pp = result.get("pp", None)
        surface = result.get("surface", pd.DataFrame())
        if prof is None or prof.empty or pp is None:
            ax.text(0.5, 0.5, "no profile/pp", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(label.capitalize())
            continue
        zmax = float(pp["zmax"])
        z = np.linspace(0.02, zmax, 240)
        rng = np.random.default_rng(13)
        mfits = sigmaz_model_fits(prof, pp)
        r_used = PAPER_R0_KPC
        for model in SIGMAZ_MODELS:
            if model not in mfits:
                continue
            ls, c = SIGMAZ_FX_STYLE[model]
            r_used, med, lo, hi = _sigma_band(mfits[model], pp, PAPER_R0_KPC, z, rng=rng)
            ax.fill_between(z, lo, hi, color=c, alpha=0.18)
            ax.plot(z, med, ls=ls, color=c, lw=2.3, label=f"{model} (16-84% band)")
        ax.plot(z, sdc.shm_sigma(r_used, z), color="firebrick", lw=1.7, label="SHM")
        ax.set_title(rf"{label.capitalize()} disc, $R={r_used:.1f}$ kpc")
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_xlim(0, zmax)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s)).set_ylabel(r"$|K_Z|$")
        ax.legend(frameon=False, fontsize=8)
    axes[0][0].set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    fig.suptitle(r"Effect of the three $\sigma_Z(|Z|)$ assumptions on $\Sigma(|Z|)$ at the solar radius (with MC error bands)", y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(outd / "fig6_solar_sigma.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig6_combined(results: dict[str, dict[str, object]], outd: Path) -> None:
    """Fig. 6.1: all populations x all three sigma_Z models overplotted on a
    single Sigma(|Z|) axis at the solar radius, each with its MC error band.
    Colour = population, line style = sigma_Z model."""
    labels = ordered_population_labels(results)
    if not labels:
        write_status_figure(outd, "fig6p1_solar_sigma_combined.png", "Figure 6.1 Not Computable",
                            ["No population surface-density results were produced."])
        return
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    rng = np.random.default_rng(17)
    z_top = 0.0
    r_ref = PAPER_R0_KPC
    for label in labels:
        result = results.get(label, {})
        prof = result.get("profile", pd.DataFrame())
        pp = result.get("pp", None)
        if prof is None or prof.empty or pp is None:
            continue
        pop_color = SIGMAZ_POP_COLOR.get(label, "0.3")
        zmax = float(pp["zmax"])
        z = np.linspace(0.02, zmax, 240)
        z_top = max(z_top, zmax)
        mfits = sigmaz_model_fits(prof, pp)
        for model in SIGMAZ_MODELS:
            if model not in mfits:
                continue
            ls, _c = SIGMAZ_FX_STYLE[model]
            r_ref, med, lo, hi = _sigma_band(mfits[model], pp, PAPER_R0_KPC, z, rng=rng)
            ax.fill_between(z, lo, hi, color=pop_color, alpha=0.08)
            ax.plot(z, med, ls=ls, color=pop_color, lw=2.0)
    zr = np.linspace(0.02, max(z_top, 1.0), 300)
    ax.plot(zr, sdc.shm_sigma(r_ref, zr), color="black", lw=2.2, label="SHM")
    pop_handles = [Line2D([0], [0], color=SIGMAZ_POP_COLOR[l], lw=2.4, label=l) for l in labels]
    model_handles = [Line2D([0], [0], color="0.25", ls=SIGMAZ_FX_STYLE[m][0], lw=2.0, label=m) for m in SIGMAZ_MODELS]
    leg1 = ax.legend(handles=pop_handles, frameon=False, fontsize=10, loc="upper left", title="population (colour)")
    ax.add_artist(leg1)
    ax.legend(handles=model_handles + [Line2D([0], [0], color="black", lw=2.2, label="SHM")],
              frameon=False, fontsize=10, loc="lower right", title=r"$\sigma_Z$ model (line style)")
    ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=12)
    ax.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=12)
    ax.set_title(r"Fig. 6.1: $\Sigma(|Z|)$ at the Sun, all populations $\times$ three $\sigma_Z$ models (MC bands)")
    ax.set_xlim(0, z_top)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s)).set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(outd / "fig6p1_solar_sigma_combined.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig9_strict_exp_sech2(results: dict[str, dict[str, object]], outd: Path, rng) -> None:
    labels = ordered_population_labels(results)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.4 * len(labels), 5.5), sharey=False)
    if len(labels) == 1:
        axes = [axes]
    for ax, label in zip(axes, labels):
        result = results.get(label, {})
        row = nearest_fit_row(result)
        pp = result.get("pp", None)
        if row is None or pp is None:
            ax.text(0.5, 0.5, "no valid strict R-bin fit", transform=ax.transAxes, ha="center", va="center")
            ax.set_ylim(0, 1)
            continue
        z = np.linspace(0.02, float(pp["zmax"]), 300)
        color = {"thin": "#2166ac", "thick": "#b2182b", "halo": "#238b45"}.get(label, "0.25")
        for law, ls, law_label in [("exp", "-", "exponential nu"), ("sech2", "--", r"sech$^2$ nu")]:
            med, lo, hi = sdc.mc_sig(row, pp, z, nmc=400, law=law, rng=rng)
            ok = np.isfinite(med) & np.isfinite(lo) & np.isfinite(hi) & (med >= 0.0) & (lo >= 0.0)
            y = np.where(ok, med, np.nan)
            ylo = np.where(ok, lo, np.nan)
            yhi = np.where(ok, hi, np.nan)
            ax.plot(z, y, ls=ls, color=color, lw=2.2, label=law_label)
            ax.fill_between(z, ylo, yhi, color=color, alpha=0.12)
        ax.plot(z, sdc.shm_sigma(float(row["rm"]), z), color="firebrick", lw=1.8, label="SHM")
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        ax.set_title(f"{label.capitalize()} disc R={float(row['rm']):.1f}: exp vs sech2")
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(0, float(pp["zmax"]))
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
    fig.tight_layout()
    fig.savefig(outd / "fig9_exp_vs_sech2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fig10_strict_kg(results: dict[str, dict[str, object]], outd: Path) -> None:
    labels = ordered_population_labels(results)
    fig, axes = plt.subplots(1, len(labels), figsize=(6.4 * len(labels), 5.5))
    if len(labels) == 1:
        axes = [axes]
    rows = []
    for ax, label in zip(axes, labels):
        result = results.get(label, {})
        profile = result.get("profile", pd.DataFrame())
        pp = result.get("pp", sdc.TN if label == "thin" else sdc.TK)
        az, sig2, esig2, rm = nearest_profile_sig2(profile)
        color = {"thin": "#2166ac", "thick": "#b2182b", "halo": "#238b45"}.get(label, "0.25")
        ax.errorbar(az, sig2, yerr=esig2, fmt="o", color=color, ms=4, capsize=2, lw=0.9, elinewidth=0.9, label="strict profile")
        res = sdc.fit_kg(az, sig2, esig2, float(pp["hZ"]))
        if res and len(az) > 0:
            zz = np.linspace(0, max(float(az.max()) * 1.1, 0.05), 300)
            ax.plot(zz, sdc.kg_sig2(zz, res["Sd"], res["rdm"], float(pp["hZ"])), "k-", lw=2.0, label=f"KG fit: Sigma={res['Sd']:.2f}, rhoDM={res['rdm']:.4f}")
            rows.append(
                {
                    "population": label,
                    "R_mid_used_kpc": rm,
                    "n_sigmaZ_points": int(len(az)),
                    "Sigma_disk_Msun_pc2": float(res["Sd"]),
                    "Sigma_disk_err_Msun_pc2": float(res["Sd_e"]),
                    "rho_dm_Msun_pc3": float(res["rdm"]),
                    "rho_dm_err_Msun_pc3": float(res["rdm_e"]),
                    "status": "ok",
                }
            )
        else:
            ax.text(0.5, 0.5, "insufficient strict sigma_Z profile points", transform=ax.transAxes, ha="center", va="center")
            rows.append(
                {
                    "population": label,
                    "R_mid_used_kpc": rm,
                    "n_sigmaZ_points": int(len(az)),
                    "Sigma_disk_Msun_pc2": np.nan,
                    "Sigma_disk_err_Msun_pc2": np.nan,
                    "rho_dm_Msun_pc3": np.nan,
                    "rho_dm_err_Msun_pc3": np.nan,
                    "status": "insufficient_profile_points",
                }
            )
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_ylabel(r"$\sigma^2_Z$ [km$^2$ s$^{-2}$]")
        ax.set_title(f"Figure 10-style {label} disc R={rm:.1f} kpc")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(outd / "fig10_KG_integral.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outd / "fig10_KG_integral_fit_results.csv", index=False)


def population_panel_count_table(data: dict[str, np.ndarray], pop_code: np.uint8, label: str) -> pd.DataFrame:
    rows = []
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            mask = (data["panel_iz"] == iz) & (data["panel_ir"] == ir) & (data["pop"] == pop_code)
            rows.append(
                {
                    "population": label,
                    "panel_iz": iz,
                    "panel_ir": ir,
                    "R_min_kpc": rlo,
                    "R_max_kpc": rhi,
                    "absZ_min_kpc": zlo,
                    "absZ_max_kpc": zhi,
                    "n_points": int(mask.sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_population_sigma_r(label: str, result: dict[str, object], outd: Path) -> None:
    surface = result.get("surface", pd.DataFrame())
    pp = result.get("pp", None)
    if surface.empty or pp is None:
        write_status_figure(outd, "sigma_R_profiles.png", f"No {label} Sigma(R)", ["No surface-density rows were produced for this population."])
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    colors = {0.3: "#1b9e77", 1.0: "#7570b3", 3.0: "#d95f02"}
    marker = {"thin": "o", "thick": "s", "halo": "^"}.get(label, "o")
    for z_value in FIG8_Z:
        if z_value > float(pp["zmax"]) + 1e-9:
            continue
        sub = nearest_surface_rows(surface, z_value).sort_values("R_mid")
        sub = sub[
            np.isfinite(sub["Sigma_median_Msun_pc2"])
            & np.isfinite(sub["Sigma_p16_Msun_pc2"])
            & np.isfinite(sub["Sigma_p84_Msun_pc2"])
        ]
        if sub.empty:
            continue
        ax.errorbar(
            sub["R_mid"],
            sub["Sigma_median_Msun_pc2"],
            yerr=[
                sub["Sigma_median_Msun_pc2"] - sub["Sigma_p16_Msun_pc2"],
                sub["Sigma_p84_Msun_pc2"] - sub["Sigma_median_Msun_pc2"],
            ],
            fmt=marker,
            ls="-",
            color=colors[float(z_value)],
            lw=1.5,
            ms=5,
            capsize=3,
            label=f"|Z|={z_value:.1f}",
        )
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    ax.set_title(f"{label.capitalize()} accepted non-negative Sigma(R)")
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outd / "sigma_R_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_population_sigma_z(label: str, result: dict[str, object], outd: Path) -> None:
    surface = result.get("surface", pd.DataFrame())
    if surface.empty:
        write_status_figure(outd, "sigma_Z_profiles.png", f"No {label} Sigma(Z)", ["No surface-density rows were produced for this population."])
        return
    r_values = sorted(surface["R_mid"].dropna().unique())
    if not r_values:
        write_status_figure(outd, "sigma_Z_profiles.png", f"No {label} Sigma(Z)", ["No valid R bins were produced for this population."])
        return
    ncols = 3
    nrows = int(np.ceil(len(r_values) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, max(4.0, 3.4 * nrows)), sharex=True, squeeze=False)
    axes_flat = axes.ravel()
    color = {"thin": "#2166ac", "thick": "#b2182b", "halo": "#238b45"}.get(label, "0.25")
    for pi, rm in enumerate(r_values):
        ax = axes_flat[pi]
        sub = surface[
            np.isclose(surface["R_mid"], rm)
            & np.isfinite(surface["Sigma_median_Msun_pc2"])
            & np.isfinite(surface["Sigma_p16_Msun_pc2"])
            & np.isfinite(surface["Sigma_p84_Msun_pc2"])
        ].sort_values("Z_abs")
        if sub.empty:
            ax.text(0.5, 0.5, "no accepted rows", transform=ax.transAxes, ha="center", va="center", fontsize=8)
            ax.set_ylim(0, 1)
        else:
            ax.plot(sub["Z_abs"], sub["Sigma_median_Msun_pc2"], color=color, lw=1.8)
            ax.fill_between(sub["Z_abs"], sub["Sigma_p16_Msun_pc2"], sub["Sigma_p84_Msun_pc2"], color=color, alpha=0.16)
            ax.plot(sub["Z_abs"], sdc.shm_sigma(float(rm), sub["Z_abs"]), color="firebrick", lw=1.0)
            ax.set_ylim(0, max(float(sub["Sigma_median_Msun_pc2"].max()) * 1.2, 1.0))
        ax.text(0.97, 0.90, f"R={float(rm):.1f}", transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        if pi % ncols == 0:
            ax.set_ylabel(r"$\Sigma$")
        if pi >= (nrows - 1) * ncols:
            ax.set_xlabel(r"$|Z|$ [kpc]")
    for pi in range(len(r_values), len(axes_flat)):
        axes_flat[pi].axis("off")
    fig.suptitle(f"{label.capitalize()} accepted non-negative Sigma(Z)", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(outd / "sigma_Z_profiles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_population_outputs(data: dict[str, np.ndarray], results: dict[str, dict[str, object]], outd: Path) -> None:
    specs = [
        ("thin", POP_THIN),
        ("thick", POP_THICK),
        ("halo", POP_HALO),
    ]
    for label, pop_code in specs:
        pop_dir = outd / f"population_{label}"
        pop_dir.mkdir(parents=True, exist_ok=True)
        population_panel_count_table(data, pop_code, label).to_csv(pop_dir / "panel_counts.csv", index=False)
        if label == "halo" and label not in results:
            pd.DataFrame(
                [
                    {
                        "population": "halo",
                        "n_points": int((data["pop"] == POP_HALO).sum()),
                        "status": "classified_left_of_L1_no_Cheng_disk_surface_density",
                    }
                ]
            ).to_csv(pop_dir / "summary.csv", index=False)
            continue
        result = results.get(label, {})
        pd.DataFrame([result.get("summary", {"population": label, "status": "missing"})]).to_csv(pop_dir / "summary.csv", index=False)
        result.get("profile", pd.DataFrame()).to_csv(pop_dir / "cprof_internal.csv", index=False)
        save_public_profile(result.get("profile", pd.DataFrame()), pop_dir / "velocity_dispersion_profiles.csv")
        result.get("fits", pd.DataFrame()).to_csv(pop_dir / "linear_fits.csv", index=False)
        result.get("model_grid", pd.DataFrame()).to_csv(pop_dir / "sigz_model_selection.csv", index=False)
        result.get("hsigma_points", pd.DataFrame()).to_csv(pop_dir / "hsigma_model_evaluated_points.csv", index=False)
        surface = result.get("surface", pd.DataFrame())
        surface.to_csv(pop_dir / "surface_density_curves.csv", index=False)
        if not surface.empty:
            surface[~surface["accepted_nonnegative"]].to_csv(pop_dir / "surface_density_rejected_nonphysical.csv", index=False)
            fig8_rows = []
            pp = result.get("pp", None)
            if pp is not None:
                for z_value in FIG8_Z:
                    if z_value <= float(pp["zmax"]) + 1e-9:
                        sub = nearest_surface_rows(surface, z_value).copy()
                        sub.insert(0, "fig8_Z_abs", float(z_value))
                        fig8_rows.append(sub)
            (pd.concat(fig8_rows, ignore_index=True) if fig8_rows else pd.DataFrame()).to_csv(pop_dir / "sigma_R_fig8_rows.csv", index=False)
        plot_population_sigma_r(label, result, pop_dir)
        plot_population_sigma_z(label, result, pop_dir)
        plot_population_sigmaz_model_effect(label, result, pop_dir)


def range_token(lo: float, hi: float) -> str:
    def tok(v: float) -> str:
        return f"{v:g}".replace("-", "m").replace(".", "p")

    return f"{tok(lo)}_{tok(hi)}"


def original_panel_dir(outd: Path, iz: int, ir: int) -> Path:
    zlo, zhi = Z_BINS[iz]
    rlo, rhi = R_BINS[ir]
    return outd / "original_panels" / f"panel_z{iz:02d}_{range_token(zlo, zhi)}_R{ir:02d}_{range_token(rlo, rhi)}"


def panel_surface_rows_for_R(results: dict[str, dict[str, object]], rlo: float, rhi: float, zlo: float, zhi: float) -> pd.DataFrame:
    frames = []
    for label, result in results.items():
        surface = result.get("surface", pd.DataFrame())
        if surface.empty:
            continue
        sub = surface[(surface["R_mid"] >= rlo) & (surface["R_mid"] < rhi)].copy()
        if sub.empty:
            continue
        sub.insert(0, "population", label)
        sub["within_panel_absZ"] = (sub["Z_abs"] >= zlo) & (sub["Z_abs"] < zhi)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def panel_surface_rows_for_Z(results: dict[str, dict[str, object]], z_mid: float, rlo: float, rhi: float) -> pd.DataFrame:
    frames = []
    for label, result in results.items():
        surface = result.get("surface", pd.DataFrame())
        if surface.empty:
            continue
        zvals = np.array(sorted(surface["Z_abs"].dropna().unique()), dtype=float)
        if zvals.size == 0:
            continue
        z_use = float(zvals[np.argmin(np.abs(zvals - z_mid))])
        sub = surface[np.isclose(surface["Z_abs"], z_use)].copy()
        if sub.empty:
            continue
        sub.insert(0, "population", label)
        sub["panel_Z_mid_kpc"] = float(z_mid)
        sub["surface_Z_delta_kpc"] = abs(z_use - z_mid)
        sub["within_panel_R"] = (sub["R_mid"] >= rlo) & (sub["R_mid"] < rhi)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_original_panel_chemical_selection(
    data: dict[str, np.ndarray],
    strict_l2,
    iz: int,
    ir: int,
    pdir: Path,
) -> None:
    lims = DISPLAY_LIMITS[COMBINED_KEY]
    xlim = lims["xlim"]
    ylim = lims["ylim"]
    mask = (data["panel_iz"] == iz) & (data["panel_ir"] == ir)
    x = data["xchem"][mask]
    y = data["ychem"][mask]
    pop = data["pop"][mask]
    ok = np.isfinite(x) & np.isfinite(y) & (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    x = x[ok]
    y = y[ok]
    pop = pop[ok]
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    ax.set_facecolor(plt.cm.viridis(0.0))
    if len(x):
        # Same rendering as the grid-figure panels: dithered viridis density to
        # avoid abundance-grid moire, plus the shared overdensity contours/crest.
        ax.hist2d(dither(x), dither(y), bins=(260, 220), range=[list(xlim), list(ylim)],
                  cmap="viridis", norm=LogNorm(vmin=1))
        draw_chemical_contours_and_crest(ax, x, y, xlim, ylim, show_contours=True, crest=True)
    else:
        ax.text(0.5, 0.5, "no clean viridis points in this panel", transform=ax.transAxes, ha="center", va="center")
    xx = np.linspace(xlim[0], xlim[1], 300)
    ax.plot(xx, HALO_LINE_M * xx + HALO_LINE_B, color="magenta", lw=2.0, ls="-", label="L1 halo boundary")
    l2_entry = strict_l2.get((iz, ir))
    if l2_entry is not None:
        m, b, is_direct = l2_entry
        yy = m * xx + b
        use = (yy >= ylim[0]) & (yy <= ylim[1])
        ax.plot(xx[use], yy[use], color="magenta", lw=1.6, ls=":",
                label="direct L2" if is_direct else "fallback L2 (cat. avg)")
    counts = {
        "thin": int((pop == POP_THIN).sum()),
        "thick": int((pop == POP_THICK).sum()),
        "halo": int((pop == POP_HALO).sum()),
        "unclass": int((pop == POP_UNCLASSIFIED).sum()),
    }
    rlo, rhi = R_BINS[ir]
    zlo, zhi = Z_BINS[iz]
    l2_tag = "direct L2" if (l2_entry is not None and l2_entry[2]) else (
        "fallback L2" if l2_entry is not None else "no L2")
    ax.set_title(
        f"{rlo:g}<R<{rhi:g}, {zlo:g}<|Z|<{zhi:g}  [{l2_tag}]\n"
        f"thin={counts['thin']:,}, thick={counts['thick']:,}, halo={counts['halo']:,}"
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel(combined_category().x_label)
    ax.set_ylabel(combined_category().y_label)
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(pdir / "chemical_selection_l1_l2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_panel_sigma_R_context(rows: pd.DataFrame, iz: int, ir: int, pdir: Path) -> None:
    if rows.empty:
        write_status_figure(
            pdir,
            "sigma_R_context.png",
            "No Sigma(R) Rows For This Panel",
            ["No strict surface-density rows were assigned at this panel's |Z| midpoint."],
        )
        return
    rlo, rhi = R_BINS[ir]
    zlo, zhi = Z_BINS[iz]
    styles = {
        "thin": dict(color="#2166ac", marker="o", ls="-"),
        "thick": dict(color="#b2182b", marker="s", ls="--"),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for label, style in styles.items():
        sub = rows[
            (rows["population"] == label)
            & rows["accepted_nonnegative"].astype(bool)
            & np.isfinite(rows["Sigma_median_Msun_pc2"])
        ].sort_values("R_mid")
        if sub.empty:
            continue
        ax.errorbar(
            sub["R_mid"],
            sub["Sigma_median_Msun_pc2"],
            yerr=[
                sub["Sigma_median_Msun_pc2"] - sub["Sigma_p16_Msun_pc2"],
                sub["Sigma_p84_Msun_pc2"] - sub["Sigma_median_Msun_pc2"],
            ],
            fmt=style["marker"],
            ls=style["ls"],
            color=style["color"],
            ms=4,
            lw=1.4,
            capsize=2,
            label=f"{label} accepted",
        )
    rejected = rows[~rows["accepted_nonnegative"].astype(bool)]
    if not rejected.empty:
        ax.scatter(rejected["R_mid"], rejected["Sigma_raw_median_Msun_pc2"], marker="x", color="0.35", s=24, label="raw rejected")
    ax.axvspan(rlo, rhi, color="0.85", alpha=0.45, label="this R panel")
    z_ref = float(rows["surface_Z_abs_used"].dropna().iloc[0]) if "surface_Z_abs_used" in rows and rows["surface_Z_abs_used"].notna().any() else 0.5 * (zlo + zhi)
    rr = np.linspace(max(0.1, min(R_BINS[0][0], rows["R_mid"].min())), max(R_BINS[-1][1], rows["R_mid"].max()), 260)
    ax.plot(rr, [sdc.shm_sigma(float(rv), [z_ref])[0] for rv in rr], color="firebrick", lw=1.3, label="SHM")
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title(f"Sigma(R) at panel |Z| midpoint, {zlo:g}<|Z|<{zhi:g}")
    ax.set_xlabel(r"$R$ [kpc]")
    ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "sigma_R_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_panel_sigma_Z_context(rows: pd.DataFrame, iz: int, ir: int, pdir: Path) -> None:
    if rows.empty:
        write_status_figure(
            pdir,
            "sigma_Z_context.png",
            "No Sigma(Z) Rows For This Panel",
            ["No strict surface-density rows were available in this panel's R range."],
        )
        return
    rlo, rhi = R_BINS[ir]
    zlo, zhi = Z_BINS[iz]
    styles = {
        "thin": dict(color="#2166ac", ls="-"),
        "thick": dict(color="#b2182b", ls="--"),
        "halo": dict(color="#238b45", ls="-."),
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    for label, style in styles.items():
        sub = rows[
            (rows["population"] == label)
            & rows["accepted_nonnegative"].astype(bool)
            & np.isfinite(rows["Sigma_median_Msun_pc2"])
        ].copy()
        if sub.empty:
            continue
        grouped = (
            sub.groupby("Z_abs", as_index=False)["Sigma_median_Msun_pc2"]
            .median()
            .sort_values("Z_abs")
        )
        ax.plot(grouped["Z_abs"], grouped["Sigma_median_Msun_pc2"], color=style["color"], ls=style["ls"], lw=2.0, label=f"{label} accepted median")
        ax.scatter(sub["Z_abs"], sub["Sigma_median_Msun_pc2"], color=style["color"], s=10, alpha=0.35)
    rejected = rows[~rows["accepted_nonnegative"].astype(bool)]
    if not rejected.empty:
        ax.scatter(rejected["Z_abs"], rejected["Sigma_raw_median_Msun_pc2"], marker="x", color="0.35", s=18, label="raw rejected")
    ax.axvspan(zlo, zhi, color="0.85", alpha=0.45, label="this |Z| panel")
    r_ref = float(rows["R_mid"].median()) if rows["R_mid"].notna().any() else 0.5 * (rlo + rhi)
    zz = np.linspace(0.01, max(4.0, float(rows["Z_abs"].max()) if rows["Z_abs"].notna().any() else 4.0), 260)
    ax.plot(zz, sdc.shm_sigma(r_ref, zz), color="firebrick", lw=1.3, label=f"SHM R~{r_ref:.1f}")
    ax.axhline(0, color="0.4", lw=0.8, ls=":")
    ax.set_title(f"Sigma(|Z|) in panel R range, {rlo:g}<R<{rhi:g}")
    ax.set_xlabel(r"$|Z|$ [kpc]")
    ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(pdir / "sigma_Z_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_panel_fig8_sigma_R(fig8_rows: pd.DataFrame, fallback_rows: pd.DataFrame, iz: int, ir: int, pdir: Path) -> None:
    rlo, rhi = R_BINS[ir]
    zlo, zhi = Z_BINS[iz]
    styles = {
        "thin": dict(color="#2166ac", marker="o", ls="-"),
        "thick": dict(color="#b2182b", marker="s", ls="--"),
        "halo": dict(color="#238b45", marker="^", ls="-."),
    }
    if fig8_rows.empty and fallback_rows.empty:
        write_status_figure(
            pdir,
            "fig8_sigma_R_panel.png",
            "No Panel Fig8 Sigma(R)",
            ["No Fig8-height or nearest-panel-Z Sigma(R) rows were assigned to this panel."],
        )
        return

    if not fig8_rows.empty:
        z_values = sorted(fig8_rows["fig8_Z_abs"].dropna().unique())
        fig, axes = plt.subplots(len(z_values), 1, figsize=(8.3, max(4.2, 3.2 * len(z_values))), sharex=True)
        if len(z_values) == 1:
            axes = [axes]
        for ax, z_value in zip(axes, z_values):
            sub_z = fig8_rows[np.isclose(fig8_rows["fig8_Z_abs"], z_value)].copy()
            for label, style in styles.items():
                sub = sub_z[
                    (sub_z["population"] == label)
                    & sub_z["accepted_nonnegative"].astype(bool)
                    & np.isfinite(sub_z["Sigma_median_Msun_pc2"])
                ].sort_values("R_mid")
                if sub.empty:
                    continue
                ax.errorbar(
                    sub["R_mid"],
                    sub["Sigma_median_Msun_pc2"],
                    yerr=[
                        sub["Sigma_median_Msun_pc2"] - sub["Sigma_p16_Msun_pc2"],
                        sub["Sigma_p84_Msun_pc2"] - sub["Sigma_median_Msun_pc2"],
                    ],
                    fmt=style["marker"],
                    ls=style["ls"],
                    color=style["color"],
                    ms=4,
                    lw=1.3,
                    capsize=2,
                    label=label,
                )
            rejected = sub_z[~sub_z["accepted_nonnegative"].astype(bool)]
            if not rejected.empty:
                ax.scatter(rejected["R_mid"], rejected["Sigma_raw_median_Msun_pc2"], marker="x", color="0.35", s=22, label="raw rejected")
            rr = np.linspace(max(0.1, rlo), rhi, 200)
            ax.plot(rr, [sdc.shm_sigma(float(rv), [float(z_value)])[0] for rv in rr], color="firebrick", lw=1.2, label="SHM")
            ax.axvspan(rlo, rhi, color="0.88", alpha=0.45)
            ax.axhline(0, color="0.45", lw=0.7, ls=":")
            ax.set_title(f"Fig8 |Z|={float(z_value):.1f} kpc")
            ax.set_ylabel(r"$\Sigma$")
            ax.set_ylim(bottom=0)
            ax.tick_params(direction="in", top=True, right=True)
            secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
            secax.set_ylabel(r"$|K_Z|$")
            ax.legend(frameon=False, fontsize=8, ncol=2)
        axes[-1].set_xlabel(r"$R$ [kpc]")
        fig.suptitle(f"Panel {rlo:g}<R<{rhi:g}, {zlo:g}<|Z|<{zhi:g}: Fig8 Sigma(R)", y=0.995)
    else:
        fig, ax = plt.subplots(figsize=(8.3, 5.2))
        for label, style in styles.items():
            sub = fallback_rows[
                (fallback_rows["population"] == label)
                & fallback_rows["accepted_nonnegative"].astype(bool)
                & np.isfinite(fallback_rows["Sigma_median_Msun_pc2"])
            ].sort_values("R_mid")
            if sub.empty:
                continue
            ax.errorbar(
                sub["R_mid"],
                sub["Sigma_median_Msun_pc2"],
                yerr=[
                    sub["Sigma_median_Msun_pc2"] - sub["Sigma_p16_Msun_pc2"],
                    sub["Sigma_p84_Msun_pc2"] - sub["Sigma_median_Msun_pc2"],
                ],
                fmt=style["marker"],
                ls=style["ls"],
                color=style["color"],
                ms=4,
                lw=1.3,
                capsize=2,
                label=label,
            )
        rejected = fallback_rows[~fallback_rows["accepted_nonnegative"].astype(bool)]
        if not rejected.empty:
            ax.scatter(rejected["R_mid"], rejected["Sigma_raw_median_Msun_pc2"], marker="x", color="0.35", s=22, label="raw rejected")
        z_ref = float(fallback_rows["surface_Z_abs_used"].dropna().iloc[0]) if "surface_Z_abs_used" in fallback_rows and fallback_rows["surface_Z_abs_used"].notna().any() else 0.5 * (zlo + zhi)
        rr = np.linspace(max(0.1, rlo), rhi, 200)
        ax.plot(rr, [sdc.shm_sigma(float(rv), [z_ref])[0] for rv in rr], color="firebrick", lw=1.2, label="SHM")
        ax.axvspan(rlo, rhi, color="0.88", alpha=0.45)
        ax.axhline(0, color="0.45", lw=0.7, ls=":")
        ax.set_title(f"Panel nearest-Z Sigma(R), |Z|~{z_ref:.2f} kpc")
        ax.set_xlabel(r"$R$ [kpc]")
        ax.set_ylabel(r"$\Sigma$")
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        secax = ax.secondary_yaxis("right", functions=(sdc.s2k, sdc.k2s))
        secax.set_ylabel(r"$|K_Z|$")
        ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(pdir / "fig8_sigma_R_panel.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_original_panel_outputs(
    data: dict[str, np.ndarray],
    strict_l2,
    counts: pd.DataFrame,
    feasibility: pd.DataFrame,
    assignments: pd.DataFrame,
    results: dict[str, dict[str, object]],
    outd: Path,
) -> pd.DataFrame:
    root = outd / "original_panels"
    root.mkdir(parents=True, exist_ok=True)
    fig8_path = outd / "original_panel_fig8_sigma_assignments.csv"
    fig8 = pd.read_csv(fig8_path) if fig8_path.exists() else pd.DataFrame()
    manifest = []
    for iz, (zlo, zhi) in enumerate(Z_BINS):
        for ir, (rlo, rhi) in enumerate(R_BINS):
            pdir = original_panel_dir(outd, iz, ir)
            pdir.mkdir(parents=True, exist_ok=True)
            count_row = counts[
                np.isclose(counts["R_min_kpc"], rlo)
                & np.isclose(counts["R_max_kpc"], rhi)
                & np.isclose(counts["absZ_min_kpc"], zlo)
                & np.isclose(counts["absZ_max_kpc"], zhi)
            ].copy()
            feas_rows = feasibility[
                np.isclose(feasibility["R_min_kpc"], rlo)
                & np.isclose(feasibility["R_max_kpc"], rhi)
                & np.isclose(feasibility["absZ_min_kpc"], zlo)
                & np.isclose(feasibility["absZ_max_kpc"], zhi)
            ].copy()
            assign_rows = assignments[
                (assignments["panel_iz"] == iz)
                & (assignments["panel_ir"] == ir)
            ].copy() if not assignments.empty else pd.DataFrame()
            fig8_rows = fig8[
                np.isclose(fig8.get("panel_R_min_kpc", pd.Series(dtype=float)), rlo)
                & np.isclose(fig8.get("panel_R_max_kpc", pd.Series(dtype=float)), rhi)
                & np.isclose(fig8.get("panel_absZ_min_kpc", pd.Series(dtype=float)), zlo)
                & np.isclose(fig8.get("panel_absZ_max_kpc", pd.Series(dtype=float)), zhi)
            ].copy() if not fig8.empty else pd.DataFrame()
            rows_R = panel_surface_rows_for_Z(results, 0.5 * (zlo + zhi), rlo, rhi)
            rows_Z = panel_surface_rows_for_R(results, rlo, rhi, zlo, zhi)

            count_row.to_csv(pdir / "population_counts.csv", index=False)
            feas_rows.to_csv(pdir / "standalone_cheng_feasibility.csv", index=False)
            assign_rows.to_csv(pdir / "surface_assignments_nearest_panel_z.csv", index=False)
            fig8_rows.to_csv(pdir / "fig8_sigma_R_rows.csv", index=False)
            rows_R.to_csv(pdir / "sigma_R_context_rows.csv", index=False)
            rows_Z.to_csv(pdir / "sigma_Z_context_rows.csv", index=False)
            plot_original_panel_chemical_selection(data, strict_l2, iz, ir, pdir)
            plot_panel_sigma_R_context(assign_rows if not assign_rows.empty else rows_R, iz, ir, pdir)
            plot_panel_sigma_Z_context(rows_Z, iz, ir, pdir)
            plot_panel_fig8_sigma_R(fig8_rows, assign_rows if not assign_rows.empty else rows_R, iz, ir, pdir)

            is_direct = bool(count_row["has_direct_l2"].iloc[0]) if not count_row.empty else False
            l2_kind = "direct" if is_direct else "fallback_category_average"
            status = "has_assigned_surface_rows" if not assign_rows.empty else "no_assigned_surface_rows"
            l2_sentence = (
                "uses a directly fitted per-panel L2 thin/thick boundary."
                if is_direct
                else "uses the category-average fallback L2 thin/thick boundary "
                "(the same line drawn on the grid figure), because the per-panel two-ridge/valley "
                "fit did not converge for this bin."
            )
            readme_lines = [
                f"# Original Panel z{iz:02d} R{ir:02d}",
                "",
                f"Panel: {rlo:g}<R<{rhi:g} kpc, {zlo:g}<|Z|<{zhi:g} kpc.",
                f"This panel {l2_sentence}",
                "Files in this folder use the strict clean combined viridis catalogue and the L1/L2 thin/thick/halo classification.",
                "Standalone Cheng feasibility is recorded separately because a single original R-|Z| panel does not by itself provide the full fixed-Z cross-R h_sigma geometry.",
                "Accepted Sigma/Kz columns are non-negative; rejected raw rows are retained only for diagnostics.",
                "",
                "Key files:",
                "- `chemical_selection_l1_l2.png`",
                "- `population_counts.csv`",
                "- `standalone_cheng_feasibility.csv`",
                "- `surface_assignments_nearest_panel_z.csv`",
                "- `sigma_R_context.png` and `sigma_R_context_rows.csv`",
                "- `sigma_Z_context.png` and `sigma_Z_context_rows.csv`",
                "- `fig8_sigma_R_panel.png` and `fig8_sigma_R_rows.csv`",
            ]
            (pdir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="ascii")
            manifest.append(
                {
                    "panel_iz": iz,
                    "panel_ir": ir,
                    "R_min_kpc": rlo,
                    "R_max_kpc": rhi,
                    "absZ_min_kpc": zlo,
                    "absZ_max_kpc": zhi,
                    "has_direct_l2": is_direct,
                    "l2_kind": l2_kind,
                    "n_clean_combined_points": int(count_row["clean_combined_points"].iloc[0]) if not count_row.empty else 0,
                    "n_thin": int(count_row["n_thin"].iloc[0]) if not count_row.empty else 0,
                    "n_thick": int(count_row["n_thick"].iloc[0]) if not count_row.empty else 0,
                    "n_halo_l1_left": int(count_row["n_halo_l1_left"].iloc[0]) if not count_row.empty else 0,
                    "n_surface_assignment_rows": int(len(assign_rows)),
                    "n_fig8_rows": int(len(fig8_rows)),
                    "status": status,
                    "folder": str(pdir),
                }
            )
    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(root / "original_panel_output_manifest.csv", index=False)
    return manifest_df


# --- Velocity-dispersion figures (Cheng Fig. 2-4 style) ---------------------
# Three populations and the three Cheng sigma_Z(|Z|) models (paper App. A & B):
#   linear     : sigma_Z^2 = a|Z| + b           (Sec 4.1 / App A, primary)
#   quadratic  : sigma_Z^2 = k Z^2 + sigma0^2    (App B, continuous at Z=0)
#   tanh       : sigma_Z   = kz*tanh(|Z|/L) + s0 (App B, continuous + linear at large Z)
VDISP_RANGES = [
    (3, 6, "fig2_velocity_dispersion_R3_6.png"),
    (6, 9, "fig3_velocity_dispersion_R6_9.png"),
    (9, 12, "fig4_velocity_dispersion_R9_12.png"),
]
VDISP_POPS = [("thin", "#2166ac"), ("thick", "#b2182b"), ("halo", "#238b45")]
SIGMAZ_MODEL_LS = {"linear": "-", "quadratic": "--", "tanh": ":"}


def fit_sigmaz_models(z_abs: np.ndarray, sZ: np.ndarray, sZe: np.ndarray) -> dict[str, tuple]:
    """Fit the three Cheng sigma_Z(|Z|) models. Returns {name: (predict_fn, rms_kms)}."""
    z = np.abs(np.asarray(z_abs, dtype=float))
    s = np.asarray(sZ, dtype=float)
    e = np.asarray(sZe, dtype=float)
    good = np.isfinite(z) & np.isfinite(s) & (s > 0)
    z, s = z[good], s[good]
    e = e[good]
    if z.size < 5:
        return {}
    s2 = s ** 2
    # weight on sigma^2: d(sigma^2) ~ 2 sigma d(sigma)
    w = np.where(np.isfinite(e) & (e > 0), 1.0 / np.maximum(2.0 * s * e, 1e-6), 1.0)
    out: dict[str, tuple] = {}

    def rms(fn):
        return float(np.sqrt(np.mean((fn(z) - s) ** 2)))

    try:  # linear: sigma^2 = a|Z| + b
        a, b = np.polyfit(z, s2, 1, w=w)
        fn = lambda Z, a=a, b=b: np.sqrt(np.clip(a * np.abs(Z) + b, 0.0, None))
        out["linear"] = (fn, rms(fn))
    except Exception:
        pass
    try:  # quadratic: sigma^2 = k Z^2 + sigma0^2
        k, c = np.polyfit(z ** 2, s2, 1, w=w)
        fn = lambda Z, k=k, c=c: np.sqrt(np.clip(k * np.asarray(Z) ** 2 + c, 0.0, None))
        out["quadratic"] = (fn, rms(fn))
    except Exception:
        pass
    try:  # tanh: sigma = kz tanh(|Z|/L) + s0  (L bounded so it can't collapse to a step)
        sigma = e if np.all(np.isfinite(e) & (e > 0)) else None
        p0 = [max(float(s.max() - s.min()), 1.0), 1.0, max(float(s.min()), 1.0)]
        popt, _ = curve_fit(
            lambda Z, kz, L, s0: kz * np.tanh(np.abs(Z) / L) + s0,
            z, s, p0=p0, sigma=sigma, absolute_sigma=False, maxfev=20000,
            bounds=([0.0, 0.3, 0.0], [400.0, 25.0, 300.0]),
        )
        fn = lambda Z, p=popt: p[0] * np.tanh(np.abs(np.asarray(Z)) / p[1]) + p[2]
        out["tanh"] = (fn, rms(fn))
    except Exception:
        pass
    return out


def plot_velocity_dispersion_models(results: dict[str, dict[str, object]], outd: Path) -> None:
    """Fig. 2-4: thin/thick/halo sigma_R, sigma_phi, sigma_Z vs Z in 1-kpc R columns.

    Fixes the empty-column bug (1-kpc columns are built from the actual fine R
    bins, not by matching integer bin edges), adds the halo series, and overlays
    the three Cheng sigma_Z(|Z|) models on the sigma_Z row with their RMS.
    """
    profs = {lab: results.get(lab, {}).get("profile", pd.DataFrame()) for lab, _ in VDISP_POPS}
    rows = [("sR", "sRe", r"$\sigma_R$"), ("sp", "spe", r"$\sigma_\phi$"), ("sZ", "sZe", r"$\sigma_Z$")]
    for r0, r1, fname in VDISP_RANGES:
        colbins = list(range(r0, r1))
        fig, axes = plt.subplots(3, len(colbins), figsize=(4.3 * len(colbins), 10.5), squeeze=False, sharex=False)
        for jc, c0 in enumerate(colbins):
            for i, (ycol, ecol, ylabel) in enumerate(rows):
                ax = axes[i][jc]
                rms_lines = []
                for lab, color in VDISP_POPS:
                    pf = profs[lab]
                    if pf.empty:
                        continue
                    sub = pf[(pf["rm"] >= c0) & (pf["rm"] < c0 + 1)]
                    if sub.empty:
                        continue
                    ax.errorbar(
                        sub["zm"], sub[ycol], yerr=sub[ecol], fmt="o", ms=2.6, lw=0.6,
                        capsize=1.2, color=color, alpha=0.7,
                        label=lab if (i == 0 and jc == 0) else None,
                    )
                    if ycol == "sZ":
                        models = fit_sigmaz_models(sub["zm"].to_numpy(), sub["sZ"].to_numpy(), sub["sZe"].to_numpy())
                        if models:
                            zg = np.linspace(float(sub["zm"].min()), float(sub["zm"].max()), 240)
                            for mname, (fn, _rms) in models.items():
                                ax.plot(zg, fn(zg), color=color, ls=SIGMAZ_MODEL_LS[mname], lw=1.4, alpha=0.95, zorder=4)
                            rms_lines.append(
                                f"{lab}: " + " ".join(f"{m[:3]}={v:.0f}" for m, (_f, v) in models.items())
                            )
                if ycol == "sZ" and rms_lines:
                    ax.text(
                        0.03, 0.97, "RMS [km/s]\n" + "\n".join(rms_lines), transform=ax.transAxes,
                        fontsize=6.5, va="top", ha="left",
                        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.75),
                    )
                ax.axvline(0, color="0.75", lw=0.8)
                ax.set_title(f"R={c0}-{c0 + 1} kpc", fontsize=9)
                ax.tick_params(direction="in", top=True, right=True, labelsize=8)
                if jc == 0:
                    ax.set_ylabel(ylabel)
                if i == 2:
                    ax.set_xlabel(r"$Z$ [kpc]")
        # legends: populations (top-left), sigma_Z models (sigma_Z first column)
        pop_handles = [Line2D([0], [0], color=c, marker="o", ls="", label=lab) for lab, c in VDISP_POPS]
        axes[0][0].legend(handles=pop_handles, frameon=False, fontsize=8, loc="upper left")
        model_handles = [Line2D([0], [0], color="0.3", ls=ls, lw=1.4, label=m)
                         for m, ls in SIGMAZ_MODEL_LS.items()]
        axes[2][0].legend(handles=model_handles, frameon=False, fontsize=7.5, loc="lower right", title=r"$\sigma_Z$ models")
        fig.suptitle(f"Velocity dispersions with three $\\sigma_Z(|Z|)$ models, R={r0}-{r1} kpc", y=0.995, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        fig.savefig(outd / fname, dpi=160, bbox_inches="tight")
        plt.close(fig)


def generate_paper_figure_set(data: dict[str, np.ndarray], strict_l2, results, outd: Path, rng) -> None:
    manifest = []

    def add(fig: str, path: str, status: str, note: str) -> None:
        manifest.append({"figure": fig, "path": path, "status": status, "note": note})

    plot_fig1_strict_chemical_plane(data, strict_l2, outd)
    add("Figure 1", "fig1_strict_chemical_plane_l1_l2.png", "ok", "Clean combined viridis chemical plane with fixed L1 and direct non-fallback L2 lines.")

    pn = results["thin"].get("profile", pd.DataFrame())
    pk = results["thick"].get("profile", pd.DataFrame())
    fn = results["thin"].get("fits", pd.DataFrame())
    fk = results["thick"].get("fits", pd.DataFrame())

    ph = results.get("halo", {}).get("profile", pd.DataFrame())
    if not pn.empty or not pk.empty or not ph.empty:
        plot_velocity_dispersion_models(results, outd)
        add("Figure 2", "fig2_velocity_dispersion_R3_6.png", "ok", "Thin/thick/halo sigma_R, sigma_phi, sigma_Z vs Z in 1-kpc R columns; sigma_Z overlaid with linear, quadratic and tanh models (RMS annotated).")
        add("Figure 3", "fig3_velocity_dispersion_R6_9.png", "ok", "As Figure 2 for R=6-9 kpc.")
        add("Figure 4", "fig4_velocity_dispersion_R9_12.png", "ok", "As Figure 2 for R=9-12 kpc.")
        plot_fig5_hsigma_results(results, outd)
        add("Figure 5", "fig5_hsigma_fit.png", "ok", "Adopted h_sigma fits; direct fixed-Z points are used when sufficient, otherwise paper odd-linear sigma_RZ fixed-Z points are used.")
    else:
        for fig, fname in [
            ("Figure 2", "fig2_velocity_dispersion_R3_6.png"),
            ("Figure 3", "fig3_velocity_dispersion_R6_9.png"),
            ("Figure 4", "fig4_velocity_dispersion_R9_12.png"),
            ("Figure 5", "fig5_hsigma_fit.png"),
        ]:
            write_status_figure(outd, fname, f"{fig} Not Computable", ["No strict velocity-dispersion profile rows were produced."])
            add(fig, fname, "blocked", "No strict velocity-dispersion profile rows were produced.")

    plot_fig6_strict_nearest(results, outd)
    add("Figure 6", "fig6_solar_sigma.png", "ok", "Sigma(|Z|) at the solar radius under the linear/quadratic/tanh sigma_Z assumptions (one panel per population, MC error bands).")
    plot_fig6_combined(results, outd)
    add("Figure 6.1", "fig6p1_solar_sigma_combined.png", "ok", "All populations x three sigma_Z models overplotted on a single Sigma(|Z|) axis at the solar radius, with MC error bands.")

    plot_fig7_strict_surface_grid(results, outd)
    add("Figure 7", "fig7_sigma_grid.png", "ok", "Accepted non-negative strict surface-density curves with SHM and right-axis |K_Z|.")

    add("Figure 8", "fig8_sigma_vs_R.png", "ok", "Accepted non-negative strict Sigma(R) points; right axis is |K_Z|=2*pi*G*Sigma.")

    plot_fig9_strict_exp_sech2(results, outd, rng)
    add("Figure 9", "fig9_exp_vs_sech2.png", "ok", "Strict nearest-R exponential-vs-sech2 comparison; negative raw intervals are omitted from accepted curves.")

    plot_fig10_strict_kg(results, outd)
    add("Figure 10", "fig10_KG_integral.png", "ok", "Strict nearest-R KG integral fit to sigma_Z^2 profiles; fit table is saved separately.")

    pd.DataFrame(manifest).to_csv(outd / "paper_figures_1_to_10_manifest.csv", index=False)


def write_readme(args, strict_l2, counts, summaries) -> None:
    with (args.output_dir / "README_strict_panel_l2_surface_density.md").open("w", encoding="ascii") as f:
        f.write("# Strict Combined-Panel L1/L2 Cheng Surface Density\n\n")
        f.write("This run starts from the clean combined abundance points used in the viridis R-|Z| panels.\n")
        f.write("L1 is the fixed steep chemical boundary. L2 is the combined-panel thin/thick boundary: the directly fitted per-panel L2 where the two-ridge/valley fit converged, and the category-average fallback L2 (the same line drawn on the grid figure) elsewhere, so every panel is classified and used. Each panel records its `l2_kind` (direct or fallback_category_average).\n")
        f.write("The halo partition is the left side of L1. Because the paper does not define halo tracer scale parameters for Eq. (1), the halo/L1-left run is an explicitly labeled diagnostic using effective tracer scale lengths fitted from the halo counts.\n\n")
        f.write(f"Paper-method choices mirrored here: fine R bins (default {args.r_bin_width:g} kpc, giving >=5-6 Sigma(R) points across each combined R panel), Z-sorted velocity-dispersion groups, bootstrap uncertainties, linear sigma_Z(|Z|), odd-through-zero sigma_RZ^2(Z), exponential tracer density, fitted h_sigma, thin-disc zmax=1 kpc, thick-disc zmax=4 kpc, and halo diagnostic zmax=8 kpc.\n\n")
        f.write("When direct interpolation of sigma_RZ^2 at fixed Z is too sparse, h_sigma is fitted from fixed-Z points evaluated from the same per-R odd-through-zero sigma_RZ^2(Z)=mZ model used in the Cheng Jeans calculation. This keeps the chemical populations separated and does not mix or borrow stars between populations.\n\n")
        f.write(f"The profile range is R={args.profile_r_min:g}-{args.profile_r_max:g} kpc and the surface-fit range is R={args.fit_r_min:g}-{args.fit_r_max:g} kpc, chosen to cover the full combined R-|Z| panel grid.\n\n")
        n_direct = sum(1 for v in strict_l2.values() if v[2])
        n_fallback = len(strict_l2) - n_direct
        f.write(f"Combined L2 panels: {n_direct} direct fits + {n_fallback} category-average fallback = {len(strict_l2)} / {len(R_BINS) * len(Z_BINS)} classified and used.\n")
        f.write(f"Total clean combined points in strict cache: {int(counts['clean_combined_points'].sum()):,}.\n")
        for summary in summaries:
            f.write(
                f"- {summary['population']}: status={summary['status']}, "
                f"N={int(summary['n_points']):,}, h_sigma={summary['h_sigma_kpc']}, "
                f"h_sigma_method={summary.get('h_sigma_method', '')}, "
                f"R bins={int(summary['n_surface_R_bins'])}, "
                f"accepted non-negative surface rows={int(summary.get('n_accepted_nonnegative_surface_rows', 0))}, "
                f"rejected nonphysical surface rows={int(summary.get('n_rejected_nonphysical_surface_rows', 0))}.\n"
            )
        f.write("\nNegative surface densities are not clipped. They are treated as failed/unphysical Jeans solutions: raw values and Jeans terms are retained in the rejection tables, while the accepted Sigma columns and plotted Fig. 8 points contain only rows whose median and 16th percentile are both non-negative.\n")
        f.write("\nKey outputs:\n")
        f.write("- `panel_l1_l2_population_counts.csv`: point counts per original combined R-|Z| panel.\n")
        f.write("- `original_panel_cheng_feasibility.csv`: per-original-panel thin/thick/halo feasibility audit for standalone Cheng fits; this records why a single R-|Z| panel cannot by itself satisfy the paper h_sigma target-coverage requirement without fallback.\n")
        f.write("- `original_panel_surface_assignments.csv`: accepted/rejected strict Sigma and K_Z rows mapped back onto the original combined R-|Z| panels from the strict global Cheng fit.\n")
        f.write("- `original_panel_fig8_sigma_assignments.csv`: accepted and rejected Fig. 8 Sigma(R) rows with their original panel assignments.\n")
        f.write("- `original_panel_sigma_R_assignments.png`: panel-row Sigma(R) assignment plot with accepted points, rejected raw points, SHM, and K_Z axes.\n")
        f.write("- `original_panel_sigma_Z_assignments.png`: panel-column Sigma(|Z|) assignment plot, separated by thin/thick population.\n")
        f.write("- `original_panel_grid_sigma_R.png`: 5x9 R-|Z| panel-layout Sigma(R) grid matching the combined abundance panel layout, with dense within-panel interpolation of accepted Cheng R-bin rows and a right-side K_Z axis.\n")
        f.write("- `original_panel_grid_sigma_R_dense_interpolated_rows.csv`: dense non-negative Sigma/K_Z rows used for the panel-grid curves.\n")
        f.write("- `original_panels/`: one folder per original combined R-|Z| panel with counts, feasibility, Sigma(R), Sigma(Z), and chemical-selection plots.\n")
        f.write("- `original_panels/panel_*/fig8_sigma_R_panel.png`: per-original-panel Fig. 8-style Sigma(R) plot with K_Z right axis.\n")
        f.write("- `population_thin/`, `population_thick/`, `population_halo/`: per-population folders with counts, fits, Sigma(R), Sigma(Z), and rejected raw diagnostics where applicable.\n")
        f.write("- `strict_combined_*_hsigma_model_evaluated_points.csv`: fixed-Z sigma_RZ^2 points evaluated from the odd-linear per-R model for h_sigma auditing.\n")
        f.write("- `strict_combined_*_surface_density_curves.csv`: surface curves for populations whose strict h_sigma fit succeeds; accepted Sigma columns are non-negative and raw Sigma columns preserve rejected diagnostics.\n")
        f.write("- `strict_combined_*_surface_density_rejected_nonphysical.csv`: rejected raw negative/nonfinite Sigma rows with Jeans-term breakdown.\n")
        f.write("- `fig8_sigma_R_points_strict_panel_l2.csv`: accepted non-negative Fig. 8 Sigma(R) points at |Z|=0.3, 1.0, 3.0 kpc.\n")
        f.write("- `fig8_sigma_R_rejected_nonphysical_strict_panel_l2.csv`: Fig. 8 rows rejected before plotting because the raw Jeans solution was negative or its uncertainty band crossed below zero.\n")
        f.write("- `fig8_sigma_R_shm_comparison_summary.csv`: accepted-point comparison to the Standard Halo Model.\n")
        f.write("- `fig8_sigma_vs_R.png`: Cheng Fig. 8-style three-panel Sigma(R) plot using accepted non-negative points only.\n")
        f.write("- `all_strict_panel_l2_sigma_R_overplot.png`: population-separated Sigma(R) overplot with multiple |Z| slices, including thin slices up to 1 kpc, halo slices up to 8 kpc, SHM reference curves, and a right-side K_Z axis.\n")
        f.write("- `fig1_strict_chemical_plane_l1_l2.png` through `fig10_KG_integral.png`: paper-style Figure 1-10 sequence where strict data coverage permits it; blocked figures are explicit status panels rather than fallback plots.\n")
        f.write("- `paper_figures_1_to_10_manifest.csv`: status of each paper-style figure and the reason for any blocked plot.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--l2-lines", type=Path, default=DEFAULT_RZ_OUTPUT_DIR / "four_category_quality_motivated_rz_l2_lines.csv")
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--transform-batch-size", type=int, default=200_000)
    parser.add_argument(
        "--r-bin-width",
        type=float,
        default=0.3,
        help="Radial bin width (kpc) for the Sigma(R) profile. 0.3 gives >=6 points "
        "across the narrowest 2-kpc combined R panel; adjustable.",
    )
    parser.add_argument("--profile-r-min", type=float, default=R_BINS[0][0])
    parser.add_argument("--profile-r-max", type=float, default=R_BINS[-1][1])
    parser.add_argument("--fit-r-min", type=float, default=R_BINS[0][0])
    parser.add_argument("--fit-r-max", type=float, default=R_BINS[-1][1])
    parser.add_argument("--nboot", type=int, default=150)
    parser.add_argument("--nmc", type=int, default=250)
    parser.add_argument("--z-grid", type=int, default=220)
    parser.add_argument("--fit-clip", type=float, default=3.0)
    parser.add_argument("--no-robust-fit", action="store_true")
    parser.add_argument("--thin-group-size", type=int, default=100)
    parser.add_argument("--halo-group-size", type=int, default=100)
    parser.add_argument("--halo-zmax", type=float, default=8.0)
    parser.add_argument("--halo-density-min-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.cache is None:
        args.cache = args.output_dir / "strict_clean_combined_panel_l1l2_points.npz"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    strict_l2 = load_strict_l2(args.l2_lines)
    if args.rebuild_cache or not args.cache.exists():
        build_strict_cache(args, strict_l2)
    else:
        print(f"[cache] using existing {args.cache}", flush=True)

    data = load_cache(args.cache)
    # Re-apply the L1/L2 classification from the full (direct + fallback) combined-panel
    # L2 table. This makes the thin/thick/halo split independent of how the cache was
    # built, so an existing cache can be reused after the L2 policy changes.
    pop, l2_direct, l2_m, l2_b = classify_points(
        data["xchem"], data["ychem"], data["panel_iz"], data["panel_ir"], strict_l2
    )
    data["pop"], data["l2_direct"], data["l2_m"], data["l2_b"] = pop, l2_direct, l2_m, l2_b
    counts = write_classification_counts(data, strict_l2, args.output_dir)
    plot_population_counts(counts, args.output_dir)
    panel_feasibility = write_panel_cheng_feasibility(
        data,
        strict_l2,
        args.output_dir,
        rbw=args.r_bin_width,
        thin_group_size=args.thin_group_size,
        halo_group_size=args.halo_group_size,
    )

    rng = np.random.default_rng(args.seed)
    thin_pp = dict(sdc.TN)
    thin_pp["ng"] = int(args.thin_group_size)
    halo_mask = data["pop"] == POP_HALO
    halo_hz, halo_hr = fit_effective_tracer_scales(
        data,
        halo_mask,
        "halo",
        args.output_dir,
        r_min=args.profile_r_min,
        r_max=args.profile_r_max,
        z_max=args.halo_zmax,
        r_bin_width=args.r_bin_width,
        z_bin_width=0.5,
        min_count=args.halo_density_min_count,
    )
    halo_pp = dict(
        lbl="Halo L1-left",
        hZ=float(halo_hz),
        hR=float(halo_hr),
        hs=max(float(halo_hr), 1.0),
        hs_e=max(0.2 * float(halo_hr), 0.5),
        zmax=float(args.halo_zmax),
        c="#238b45",
        ls="-.",
        ng=int(args.halo_group_size),
        alpha=0.18,
    )
    results = {
        "thin": run_population(data, POP_THIN, "thin", thin_pp, [-1, 1], args, rng),
        "thick": run_population(data, POP_THICK, "thick", dict(sdc.TK), [-2, -1, 1, 2], args, rng),
        "halo": run_population(data, POP_HALO, "halo", halo_pp, [-6, -4, -2, 2, 4, 6], args, rng),
    }
    summaries = [results[label]["summary"] for label in ordered_population_labels(results)]
    pd.DataFrame(summaries).to_csv(args.output_dir / "strict_panel_l2_surface_density_summary.csv", index=False)

    plot_fig8(results, args.output_dir)
    plot_all_sigma_overplot(results, args.output_dir)
    write_fig8_points(results, args.output_dir)
    write_shm_comparison_summary(args.output_dir)
    panel_assignments = write_original_panel_surface_assignments(results, args.output_dir)
    write_fig8_panel_assignment_table(args.output_dir)
    plot_original_panel_sigma_assignments(panel_assignments, args.output_dir)
    plot_original_panel_sigma_z_assignments(panel_assignments, args.output_dir)
    plot_sigma_R_original_panel_grid(panel_assignments, args.output_dir)
    write_population_outputs(data, results, args.output_dir)
    write_original_panel_outputs(data, strict_l2, counts, panel_feasibility, panel_assignments, results, args.output_dir)
    generate_paper_figure_set(data, strict_l2, results, args.output_dir, rng)
    write_readme(args, strict_l2, counts, summaries)
    print(f"[done] outputs saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
