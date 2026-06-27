#!/usr/bin/env python3
"""Cheng-style chemical-plane contours + ridge lines.

Standard nested density contours on the smoothed 2D histogram: the outer black
levels wrap the whole overdense region (shape), the inner blue levels isolate
the dense cores. Red ridge lines trace each chemical sequence (the thin and
thick disc ridges), computed per Cheng population region. This matches the
Cheng+2024 [Mg/Fe]-[Fe/H] panel style.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None

# Cheng+2024 guide lines (metallicity on x, alpha-ratio on y).
THIN_THICK_M = -0.05000836526147814
THIN_THICK_B = 0.2000005330440794
HALO_M = -0.8002225074868987
HALO_B = -0.6006222012635331

CONTOUR_SIGMA = 1.5          # KDE-like Gaussian smoothing of the 2D histogram
SHARPEN_AMOUNT = 1.0         # unsharp-mask strength (enhance overdensity features)
SHARPEN_SIGMA_FACTOR = 4.0   # large-scale background blur (astrophysics unsharp mask)
CONTOUR_FLOOR_COUNTS = 4.0   # smoothed amplitude of a few isolated counts
BLACK_FRACS = (0.975, 0.90, 0.80, 0.68, 0.55, 0.42)  # nested shape contours
BLUE_FRACS = (0.27, 0.14)                            # dense-core contours
RIDGE_FRAC = 0.60            # ridge follows each sequence's dense spine
RIDGE_LW = 1.0               # slim crest line
MIN_PTS_CONTOUR = 400
MIN_PTS_RIDGE = 2000
SHAPE_COLOR = "black"
CORE_COLOR = "blue"
RIDGE_COLOR = "red"
DITHER = 0.0066             # jitter half-width (dex) to de-quantize 0.01-dex abundance grids
_DITHER_RNG = np.random.default_rng(20260626)


def dither(a):
    """De-quantize an abundance array by adding small uniform jitter.

    Many surveys (notably Gaia GSP-Spec) report abundances rounded to a 0.01-dex
    grid, which shows up as periodic horizontal/vertical interference stripes in
    a density plot. Spreading each value uniformly across one grid cell removes
    the stripes without shifting the smooth distribution.
    """
    a = np.asarray(a, dtype=np.float64)
    return a + _DITHER_RNG.uniform(-DITHER, DITHER, size=a.shape)


def density_level_for_fraction(density, fraction):
    """Density level above which the pixels contain `fraction` of the mass."""
    vals = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(vals)
    if cumulative[-1] <= 0:
        return np.nan
    cumulative = cumulative / cumulative[-1]
    return float(vals[min(int(np.searchsorted(cumulative, fraction)), vals.size - 1)])


def smoothed_density(x, y, xlim, ylim):
    n = len(x)
    bins = int(np.clip(np.sqrt(n) / 2.0, 50, 170))
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=[list(xlim), list(ylim)])
    if gaussian_filter is not None:
        smooth = gaussian_filter(hist, sigma=CONTOUR_SIGMA, mode="constant")
        broad = gaussian_filter(smooth, sigma=CONTOUR_SIGMA * SHARPEN_SIGMA_FACTOR, mode="constant")
        hist = np.clip(smooth + SHARPEN_AMOUNT * (smooth - broad), 0.0, None)
    floor = CONTOUR_FLOOR_COUNTS / (2.0 * np.pi * CONTOUR_SIGMA * CONTOUR_SIGMA)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return hist, x_centers, y_centers, floor


def cheng_region_masks(x: np.ndarray, y: np.ndarray):
    """thin (below solid line), thick (above solid line), halo (left of dashed)."""
    thin_thick = THIN_THICK_M * x + THIN_THICK_B
    halo = HALO_M * x + HALO_B
    is_halo = (x < -0.5) & (y >= halo)
    is_thick = (~is_halo) & (y > thin_thick)
    is_thin = (~is_halo) & (y <= thin_thick)
    return (("thin", is_thin), ("thick", is_thick), ("halo", is_halo))


def _levels(hist, fracs, floor):
    out = []
    for f in fracs:
        lv = density_level_for_fraction(hist, f)
        if np.isfinite(lv) and floor < lv < hist.max():
            out.append(round(lv, 5))
    return sorted(set(out))


def _ridge_segments(hist, x_centers, y_centers, level, max_jump):
    """Per-column density-maximum points above `level`.

    The ridge is split into separate pieces wherever it is interrupted (a column
    gap) OR jumps by more than `max_jump` in y between adjacent columns. The
    jump split prevents the spurious near-vertical connector that appears when
    the column-wise maximum hops between two sequences (thin <-> thick).
    """
    col_max = hist.max(axis=1)
    dy = y_centers[1] - y_centers[0]
    points = []
    for i in range(hist.shape[0]):
        if col_max[i] < level:
            continue
        j = int(np.argmax(hist[i]))
        if 0 < j < hist.shape[1] - 1:
            a, b, c = hist[i, j - 1], hist[i, j], hist[i, j + 1]
            denom = a - 2.0 * b + c
            shift = float(np.clip(0.5 * (a - c) / denom if denom != 0 else 0.0, -1.0, 1.0))
        else:
            shift = 0.0
        points.append((i, x_centers[i], y_centers[j] + shift * dy))
    segments, current = [], []
    for (i, xx, yy) in points:
        if current and (i != current[-1][0] + 1 or abs(yy - current[-1][2]) > max_jump):
            segments.append(current)
            current = []
        current.append((i, xx, yy))
    if current:
        segments.append(current)
    return segments


def _region_grid_masks(x_centers, y_centers):
    """Cheng thin/thick/halo region masks evaluated on the density grid."""
    XX = x_centers[:, None]
    YY = y_centers[None, :]
    thin_thick = THIN_THICK_M * XX + THIN_THICK_B
    halo = HALO_M * XX + HALO_B
    is_halo = (XX < -0.5) & (YY >= halo)
    is_thick = (~is_halo) & (YY > thin_thick)
    is_thin = (~is_halo) & (YY <= thin_thick)
    return (("thin", is_thin), ("thick", is_thick), ("halo", is_halo))


def combined_density(groups, xlim, ylim):
    """Balanced combination of per-category density maps (NOT raw pooling).

    Each abundance category's cleaned points become a smoothed + unsharp-sharpened
    density map on a common grid, normalized to unit integral, then averaged with
    equal weight. This keeps the largest survey (e.g. Gaia-XP) from dominating the
    contour shape and stops one category's features from being smeared away.
    """
    total = sum(int(np.size(g[0])) for g in groups)
    if total < MIN_PTS_CONTOUR:
        return None, None, None, 0.0
    bins = int(np.clip(np.sqrt(total) / 2.0, 50, 170))
    x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    acc = np.zeros((bins, bins), dtype=np.float64)
    used = 0
    for (gx, gy) in groups:
        gx = np.asarray(gx, dtype=np.float64)
        gy = np.asarray(gy, dtype=np.float64)
        m = np.isfinite(gx) & np.isfinite(gy)
        if int(m.sum()) < MIN_PTS_CONTOUR:
            continue
        h, _, _ = np.histogram2d(gx[m], gy[m], bins=[x_edges, y_edges])
        if gaussian_filter is not None:
            smooth = gaussian_filter(h, sigma=CONTOUR_SIGMA, mode="constant")
            broad = gaussian_filter(smooth, sigma=CONTOUR_SIGMA * SHARPEN_SIGMA_FACTOR, mode="constant")
            h = np.clip(smooth + SHARPEN_AMOUNT * (smooth - broad), 0.0, None)
        s = h.sum()
        if s > 0:
            acc += h / s
            used += 1
    if used == 0:
        return None, x_centers, y_centers, 0.0
    floor = 0.0012 * acc.max()
    return acc, x_centers, y_centers, floor


def draw_chemical_contours_and_crest(
    ax, x, y, xlim, ylim,
    crest: bool = True,
    crest_black_lw: float = 1.0,   # kept for call compatibility
    crest_gold_lw: float = 0.8,    # kept for call compatibility
    contour_color: str = SHAPE_COLOR,
    show_contours: bool = True,
    groups=None,
) -> None:
    """Nested shape contours (black) + dense-core contours (blue) + red sequence
    ridges. If `groups` (list of (x_i, y_i) per abundance category) is given, the
    density is the balanced combination of per-category normalized maps instead
    of a raw pooled histogram."""
    if groups is not None:
        hist, xc, yc, floor = combined_density(groups, xlim, ylim)
        if hist is None or hist.max() <= 0:
            return
    else:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        good = np.isfinite(x) & np.isfinite(y)
        x = x[good]
        y = y[good]
        if len(x) < MIN_PTS_CONTOUR:
            return
        hist, xc, yc, floor = smoothed_density(x, y, xlim, ylim)
        if hist.max() < 2.5 * floor:
            return

    if show_contours:
        black = _levels(hist, BLACK_FRACS, floor)
        blue = _levels(hist, BLUE_FRACS, floor)
        if black:
            ax.contour(xc, yc, hist.T, levels=black, colors=contour_color,
                       linewidths=0.45, alpha=0.9, zorder=4)
        if blue:
            ax.contour(xc, yc, hist.T, levels=blue, colors=CORE_COLOR,
                       linewidths=0.6, alpha=0.95, zorder=5)

    if not crest:
        return
    max_jump = 6.0 * (yc[1] - yc[0])
    for _name, gmask in _region_grid_masks(xc, yc):
        hr = np.where(gmask, hist, 0.0)
        if hr.max() < 1.6 * floor:
            continue
        level = max(density_level_for_fraction(hr, RIDGE_FRAC), 1.6 * floor)
        for seg in _ridge_segments(hr, xc, yc, level, max_jump):
            if len(seg) < 7:
                continue
            sx = np.array([p[1] for p in seg])
            sy = np.array([p[2] for p in seg])
            k = max(3, (len(sy) // 5) | 1)
            half = k // 2
            sy = np.array([np.median(sy[max(0, i - half):i + half + 1]) for i in range(len(sy))])
            if gaussian_filter is not None:
                sy = gaussian_filter(sy, sigma=1.5, mode="nearest")
            ax.plot(sx, sy, color=RIDGE_COLOR, lw=RIDGE_LW, solid_capstyle="round", zorder=7)
