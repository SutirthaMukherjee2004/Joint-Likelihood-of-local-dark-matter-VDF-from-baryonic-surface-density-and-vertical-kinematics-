#!/usr/bin/env python3
"""
Notebook-faithful Cheng+2024 Kz / surface-density reproduction.

This is a Python-script version of /user/sutirtha/cheng2024_kz_repro.ipynb.
It intentionally follows the notebook method:

  * required real NPZ cache load
  * grouped Z-sorted bootstrap velocity profiles
  * sigma_Z(|Z|) linear fits for Cheng+2024 main-text reproduction
    (quadratic/tanh/BIC are available only as diagnostics)
  * sigma_RZ^2(Z) signed covariance fit constrained to be odd through Z=0
  * h_sigma fit from sigma_RZ^2(R) at fixed Z targets
  * Jeans Eq. (1) surface-density calculation
  * notebook SHM reference with r_s=10.8 and integral over zg

Expected real-cache arrays:
  R, Z, vR, vZ, and either vp or vphi.
Optional chemistry labels:
  chem_mgfe with 1=thin, 2=thick.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, curve_fit
from scipy.interpolate import interp1d

import kg1989_complete as kg  # KG89 equations 22-56 (see kg1989_complete.py)

warnings.filterwarnings("ignore")


G = 4.30091727e-6
TG = 2 * np.pi * G
R0 = 8.122
TINY = 1e-12
CHEM_THIN  = np.uint8(1)
CHEM_THICK = np.uint8(2)
trapz = getattr(np, "trapezoid", np.trapz)

# KG89 internal unit normalisations (eq 41–43 context)
KG_SIGMA_UNIT = 36.7    # M_sun/pc²  (1 KG disc unit)
KG_RHO_UNIT   = 0.367   # M_sun/pc³  (1 KG density unit)

TN = dict(lbl="Thin disc", hZ=0.30, hR=2.6, hs=4.53, hs_e=1.61,
          zmax=1.0, c="#2166ac", ls="-", ng=500, alpha=0.25)
TK = dict(lbl="Thick disc", hZ=0.90, hR=2.0, hs=5.03, hs_e=1.36,
          zmax=4.0, c="#6baed6", ls="--", ng=200, alpha=0.20)
TA = dict(lbl="All stars", hZ=0.70, hR=2.3, hs=4.8, hs_e=1.5,
          zmax=4.0, c="#2ca25f", ls="-", ng=2000, alpha=0.22)

OVERLAY_STYLES = {
    "mgfe_feh_thin_chem": dict(label="MgFe thin", color="#2166ac", ls="-", marker="o"),
    "mgfe_feh_thick_chem": dict(label="MgFe thick", color="#053061", ls="--", marker="s"),
    "alphafe_feh_thin_chem": dict(label="alpha thin", color="#f4a582", ls="-", marker="^"),
    "alphafe_feh_thick_chem": dict(label="alpha thick", color="#b2182b", ls="--", marker="D"),
}

s2k = lambda s: TG * np.asarray(s) * 1e6
k2s = lambda k: np.asarray(k) / (TG * 1e6)


def load_data(cache: Path):
    if not cache.exists():
        raise FileNotFoundError(f"Required real catalogue cache not found: {cache}")
    z = np.load(cache)
    vel_phi_key = "vp" if "vp" in z.files else "vphi"
    need = ["R", "Z", "vR", "vZ", vel_phi_key]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise RuntimeError(f"Cache {cache} missing arrays: {missing}")

    def subset(mask):
        return dict(R=z["R"][mask], Z=z["Z"][mask], vR=z["vR"][mask],
                    vZ=z["vZ"][mask], vp=z[vel_phi_key][mask])

    if "chem_mgfe" in z.files:
        cls = z["chem_mgfe"]
        dn = subset(cls == 1)
        dk = subset(cls == 2)
        print(f"Real data loaded from {cache}: thin={len(dn['R']):,}, thick={len(dk['R']):,}")
    else:
        raise RuntimeError(
            f"Cache {cache} has no chem_mgfe labels. This script is catalogue-only "
            "and requires a chemically mapped cache with chem_mgfe: 1=thin, 2=thick."
        )
    if len(dn["R"]) == 0 or len(dk["R"]) == 0:
        raise RuntimeError("Chemical cache produced an empty thin or thick sample.")
    return dn, dk


def load_combined_data(cache: Path):
    if not cache.exists():
        raise FileNotFoundError(f"Required real catalogue cache not found: {cache}")
    z = np.load(cache)
    vel_phi_key = "vp" if "vp" in z.files else "vphi"
    need = ["R", "Z", "vR", "vZ", vel_phi_key]
    missing = [k for k in need if k not in z.files]
    if missing:
        raise RuntimeError(f"Cache {cache} missing arrays: {missing}")
    dat = dict(R=z["R"], Z=z["Z"], vR=z["vR"], vZ=z["vZ"], vp=z[vel_phi_key])
    print(f"Combined real data loaded from {cache}: N={len(dat['R']):,}")
    if "src" in z.files:
        vals, cnt = np.unique(z["src"], return_counts=True)
        print("Source counts:", dict(zip(vals.tolist(), cnt.tolist())), "(0=FITS chunks, 1=table CSV)")
    return dat


def bdisp(vR, vp, vZ, nb=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(vR)
    if n < 10:
        nan3 = np.full(3, np.nan)
        return nan3, nan3, np.nan, np.nan
    bs = np.empty((nb, 3))
    bc = np.empty(nb)
    for b in range(nb):
        ii = rng.integers(0, n, n)
        bs[b, 0] = vR[ii].std(ddof=1)
        bs[b, 1] = vp[ii].std(ddof=1)
        bs[b, 2] = vZ[ii].std(ddof=1)
        bc[b] = np.cov(vR[ii], vZ[ii], ddof=1)[0, 1]
    return (np.nanmedian(bs, 0), np.nanstd(bs, 0, ddof=1),
            float(np.nanmedian(bc)), float(np.nanstd(bc, ddof=1)))


def cprof(dat, ng=500, rl0=3, rl1=12, rbw=1.0, nb=150, rng=None, label=""):
    if rng is None:
        rng = np.random.default_rng(42)
    R, Z = dat["R"], dat["Z"]
    vR, vp, vZ = dat["vR"], dat["vp"], dat["vZ"]
    rows = []
    r_edges = list(np.arange(rl0, rl1, rbw))
    n_bins = len(r_edges)
    tag = f"[{label}] " if label else ""
    for bi, rl in enumerate(r_edges):
        rh = rl + rbw
        ii = np.where((R >= rl) & (R < rh))[0]
        n_grp = len(ii) // ng if len(ii) >= ng else 0
        print(f"  {tag}R={rl:.1f}-{rh:.1f} kpc  [{bi+1}/{n_bins}]  stars={len(ii):,}  groups={n_grp}", flush=True)
        if len(ii) < ng:
            continue
        ii = ii[np.argsort(Z[ii])]
        for g in range(n_grp):
            jj = ii[g * ng:(g + 1) * ng]
            sig, se, crz, ce = bdisp(vR[jj], vp[jj], vZ[jj], nb, rng)
            rows.append(dict(rl=float(rl), rh=float(rh), rm=float(0.5 * (rl + rh)),
                             zm=float(np.median(Z[jj])),
                             z16=float(np.percentile(Z[jj], 16)),
                             z84=float(np.percentile(Z[jj], 84)),
                             sR=sig[0], sRe=se[0],
                             sp=sig[1], spe=se[1],
                             sZ=sig[2], sZe=se[2],
                             crz=crz, crze=ce))
    return pd.DataFrame(rows)


def crz_at_z(pf, zt, ztol=0.45):
    pts = []
    for rl in sorted(pf.rl.unique()):
        sub = pf[pf.rl == rl].sort_values("zm")
        z, y, e = sub.zm.values, sub.crz.values, sub.crze.values
        ok = np.isfinite(z) & np.isfinite(y) & (e > 0) & np.isfinite(e)
        z, y, e = z[ok], y[ok], e[ok]
        if len(z) < 2:
            continue
        bl = z[z <= zt]
        ab = z[z >= zt]
        if not len(bl) or not len(ab):
            continue
        if abs(bl.max() - zt) > ztol or abs(ab.min() - zt) > ztol:
            continue
        pts.append((rl + 0.5, float(np.interp(zt, z, y)), float(np.interp(zt, z, e))))
    return pts


def _fit_hs_xy(R, y, e, zv, Rref=8.0):
    """Fit h_sigma from pre-extracted (R, y, e, z_target) arrays with error weighting."""
    ok = np.isfinite(R) & np.isfinite(y) & (e > 0) & np.isfinite(e)
    R, y, e, zv = R[ok], y[ok], e[ok], zv[ok]
    if len(R) < 4:
        return np.nan, np.nan, {}
    # Error-weighted chi2; cap at 95th percentile to avoid huge weight on tiny-error outliers
    raw_w = 1.0 / np.maximum(e ** 2, (0.05 * np.abs(y) + 1.0) ** 2)
    wmax = np.percentile(raw_w, 95) if len(raw_w) > 5 else raw_w.max()
    w = np.clip(raw_w, 0.0, wmax)
    zvs = np.unique(zv)

    def chi2(hs):
        if hs <= 0:
            return 1e18, {}
        f = np.exp(-(R - Rref) / hs)
        amps = {}
        model = np.zeros_like(y)
        for z0 in zvs:
            m = zv == z0
            if m.sum() < 2:
                continue
            d = np.sum(w[m] * f[m] ** 2)
            a = np.sum(w[m] * f[m] * y[m]) / d if d > 0 else np.nan
            amps[float(z0)] = float(a)
            model[m] = a * f[m]
        return float(np.sum(w * (y - model) ** 2)), amps

    grd = np.linspace(0.5, 30, 2000)
    h0 = grd[np.nanargmin([chi2(h)[0] for h in grd])]
    res = minimize_scalar(lambda h: chi2(h)[0], bounds=(0.3, 50), method="bounded")
    hb = float(res.x) if res.success else h0
    c0, amps = chi2(hb)
    dn2 = np.linspace(max(0.3, hb - 8), hb + 8, 3000)
    ins = dn2[np.array([chi2(h)[0] for h in dn2]) <= c0 + 1.0]
    he = 0.5 * (ins.max() - ins.min()) if len(ins) >= 2 else 0.2 * hb
    return hb, he, amps


def fit_hs(pf, zts, Rref=8.0):
    all_pts = []
    for zt in zts:
        for Rv, yv, ev in crz_at_z(pf, zt):
            all_pts.append((Rv, yv, ev, zt))
    if len(all_pts) < 6:
        return np.nan, np.nan, {}
    R, y, e, zv = [np.array([p[i] for p in all_pts]) for i in range(4)]
    return _fit_hs_xy(R, y, e, zv, Rref=Rref)


# ---------------------------------------------------------------------------
# Kinematic thin/thick disc separation (Bensby+2003 probabilistic method)
# ---------------------------------------------------------------------------

def bensby_classify(vR, vZ, vphi, Vc=238.0):
    """
    Assign each star a posterior P(thick disc | kinematics) using Bensby+2003
    Gaussian velocity ellipsoids.  U=-vR (toward GC), V=vphi-Vc, W=vZ.

    Thin:  sigma(U,V,W)=(35,20,16) km/s, Va=-15 km/s, local fraction 0.94
    Thick: sigma(U,V,W)=(67,38,35) km/s, Va=-51 km/s, local fraction 0.10
    """
    U = -np.asarray(vR,   dtype=float)
    V =  np.asarray(vphi, dtype=float) - Vc
    W =  np.asarray(vZ,   dtype=float)

    _thin  = dict(sU=35.0, sV=20.0, sW=16.0, Va=-15.0, X=0.94)
    _thick = dict(sU=67.0, sV=38.0, sW=35.0, Va=-51.0, X=0.10)

    def _like(p):
        norm = (2.0 * np.pi) ** 1.5 * p["sU"] * p["sV"] * p["sW"]
        ex   = (U / p["sU"]) ** 2 + ((V - p["Va"]) / p["sV"]) ** 2 + (W / p["sW"]) ** 2
        return p["X"] / norm * np.exp(-0.5 * ex)

    lt = _like(_thin)
    lk = _like(_thick)
    denom = np.where(lt + lk < 1e-300, 1e-300, lt + lk)
    return lk / denom   # P(thick)


def compute_kinematic_crz_profiles(R, Z, vR, vZ, p_thick,
                                    z_targets, R_edges,
                                    p_thin_cut=0.25, p_thick_cut=0.75,
                                    Nmin=80, dZ=0.35):
    """
    Compute sigma_RZ^2 per R bin at fixed Z slices for Bensby-selected
    kinematic thin (P_thick < p_thin_cut) and thick (P_thick > p_thick_cut).
    Returns (df_thin, df_thick) DataFrames with columns
      z_target, R_mid, sigma_RZ2, sigma_RZ2_err, N.
    """
    R   = np.asarray(R,       dtype=float)
    Z   = np.asarray(Z,       dtype=float)
    vR_ = np.asarray(vR,      dtype=float)
    vZ_ = np.asarray(vZ,      dtype=float)

    thin_mask  = p_thick < p_thin_cut
    thick_mask = p_thick > p_thick_cut

    rows_thin, rows_thick = [], []
    for z0 in z_targets:
        z_sel = np.abs(Z - z0) <= dZ
        for i in range(len(R_edges) - 1):
            rl, rh = float(R_edges[i]), float(R_edges[i + 1])
            r_sel  = (R >= rl) & (R < rh)
            base   = z_sel & r_sel
            Rmid   = 0.5 * (rl + rh)

            for mask, rows in [(thin_mask & base, rows_thin),
                                (thick_mask & base, rows_thick)]:
                if mask.sum() < Nmin:
                    continue
                vr_s = vR_[mask];  vz_s = vZ_[mask]
                n    = int(mask.sum())
                crz  = float(np.mean(vr_s * vz_s) - np.mean(vr_s) * np.mean(vz_s))
                # analytical error: sqrt(sigma_R^2 * sigma_Z^2 / n)
                err  = float(np.sqrt(np.var(vr_s, ddof=1) * np.var(vz_s, ddof=1) / n))
                rows.append({"z_target": z0, "R_mid": Rmid,
                              "sigma_RZ2": crz, "sigma_RZ2_err": err, "N": n})

    return pd.DataFrame(rows_thin), pd.DataFrame(rows_thick)


def compute_vphi_crz_profiles(R, Z, vR, vZ, vp, z_targets, R_edges,
                               vp_fast_min=218.0,
                               vp_slow_max=200.0, vp_slow_min=130.0,
                               Nmin=80, dZ=0.35):
    """
    Select by azimuthal velocity to get disc-population-like subsamples
    WITHOUT biasing sigma_RZ (vphi is orthogonal to vR and vZ).

    fast  (vphi >= vp_fast_min):           ~97% pure thin disc
    slow  (vp_slow_min <= vphi < vp_slow_max): thick disc-enhanced

    Returns (df_fast, df_slow).
    """
    R   = np.asarray(R,  dtype=float)
    Z   = np.asarray(Z,  dtype=float)
    vR_ = np.asarray(vR, dtype=float)
    vZ_ = np.asarray(vZ, dtype=float)
    vp_ = np.asarray(vp, dtype=float)

    fast_mask = vp_ >= vp_fast_min
    slow_mask = (vp_ >= vp_slow_min) & (vp_ < vp_slow_max)

    rows_fast, rows_slow = [], []
    for z0 in z_targets:
        z_sel = np.abs(Z - z0) <= dZ
        for i in range(len(R_edges) - 1):
            rl, rh = float(R_edges[i]), float(R_edges[i + 1])
            r_sel  = (R >= rl) & (R < rh)
            base   = z_sel & r_sel
            Rmid   = 0.5 * (rl + rh)
            for mask, rows in [(fast_mask & base, rows_fast),
                               (slow_mask & base, rows_slow)]:
                if mask.sum() < Nmin:
                    continue
                vr_s = vR_[mask]; vz_s = vZ_[mask]
                n    = int(mask.sum())
                crz  = float(np.mean(vr_s * vz_s) - np.mean(vr_s) * np.mean(vz_s))
                err  = float(np.sqrt(np.var(vr_s, ddof=1) * np.var(vz_s, ddof=1) / n))
                rows.append({"z_target": z0, "R_mid": Rmid,
                             "sigma_RZ2": crz, "sigma_RZ2_err": err, "N": n})

    return pd.DataFrame(rows_fast), pd.DataFrame(rows_slow)


# ---------------------------------------------------------------------------
# Chemical raw-data computation (replaces precomputed overlay CSVs).
# Default Cheng+2024 reproduction uses MgFe-only chemistry; MgFe+alpha pooling is
# retained as an explicitly requested diagnostic mode.
# ---------------------------------------------------------------------------

def load_chem_data(chem_cache: Path) -> dict:
    """Load chemistry-tagged NPZ produced by surface_density.py."""
    z = np.load(chem_cache)
    vphi_key = "vphi" if "vphi" in z.files else "vp"
    d = dict(R=z["R"].astype(float), Z=z["Z"].astype(float),
             vR=z["vR"].astype(float), vZ=z["vZ"].astype(float),
             vp=z[vphi_key].astype(float))
    for k in ("chem_mgfe", "chem_alpha", "src"):
        if k in z.files:
            d[k] = z[k]
    return d


def get_mgfe_chem_masks(da: dict):
    """Paper-faithful Cheng+2024 masks from the [Mg/Fe]-[Fe/H] selection only."""
    if "chem_mgfe" not in da:
        raise RuntimeError("chem_mgfe is required for paper-faithful MgFe selection.")
    cm = da["chem_mgfe"]
    return cm == CHEM_THIN, cm == CHEM_THICK


def get_combined_chem_masks(da: dict):
    """
    Combined thin/thick masks: MgFe and alpha-Fe treated as the same
    classification (FeH as x-axis discriminator, both indicators pooled).
      thin  = thin by either indicator, NOT thick by either
      thick = thick by either indicator, NOT thin by either
    """
    n = len(da["R"])
    cm = da.get("chem_mgfe", np.zeros(n, dtype=np.uint8))
    ca = da.get("chem_alpha", np.zeros(n, dtype=np.uint8))
    is_thin  = (cm == CHEM_THIN)  | (ca == CHEM_THIN)
    is_thick = (cm == CHEM_THICK) | (ca == CHEM_THICK)
    return is_thin & ~is_thick, is_thick & ~is_thin  # (thin_mask, thick_mask)


def get_chem_masks(da: dict, selection: str = "mgfe"):
    """Return thin/thick masks for either paper MgFe-only or exploratory pooled chemistry."""
    if selection == "mgfe":
        return get_mgfe_chem_masks(da)
    if selection == "combined":
        return get_combined_chem_masks(da)
    raise ValueError(f"Unknown chemistry selection: {selection}")


def _cprof_subset(da: dict, mask: np.ndarray, ng: int, rbw: float,
                  nboot: int, rng, label="") -> pd.DataFrame:
    """Run cprof on a masked sub-sample of da."""
    if int(mask.sum()) < ng:
        return pd.DataFrame()
    sub = {k: v[mask] for k, v in da.items()
           if k in ("R", "Z", "vR", "vZ", "vp")}
    return cprof(sub, ng=ng, rbw=rbw, nb=nboot, rng=rng, label=label)


def compute_chem_overlays_from_raw(da, thin_mask, thick_mask,
                                    nboot, rbw, rng,
                                    sigz_model, clip, nmc, outd,
                                    selection_label="Combined"):
    """
    Full pipeline for combined thin/thick chemical sub-samples computed
    directly from the raw NPZ (no precomputed overlay CSVs needed).

    Returns a list of overlay dicts in the same format as load_overlay_sets:
      [{key, label, color, ls, marker, profile (DataFrame), surface (DataFrame)}, ...]

    Also saves velocity profiles and surface density curves to outd.
    """
    populations = [
        ("combined_thin",  thin_mask,  TN,
         "#2166ac", "-",  "o", f"{selection_label} thin"),
        ("combined_thick", thick_mask, TK,
         "#b2182b", "--", "s", f"{selection_label} thick"),
    ]

    overlays = []
    raw_profiles = {}   # name → cprof DataFrame (internal column names)
    for name, mask, pp, color, ls, marker, label in populations:
        n = int(mask.sum())
        if n < pp["ng"]:
            print(f"  {name}: only {n:,} stars (need {pp['ng']}), skipping")
            continue

        print(f"  {name}: {n:,} stars → cprof (ng={pp['ng']})...", flush=True)
        prof = _cprof_subset(da, mask, pp["ng"], rbw, nboot, rng, label=name)
        if prof.empty:
            print(f"  {name}: empty profile"); continue

        # Save velocity dispersion profile in the format _profile_overlay_df expects
        prof_save = pd.DataFrame({
            "R_mid": prof.rm,       "Z_median": prof.zm,
            "sigma_R": prof.sR,     "sigma_R_err": prof.sRe,
            "sigma_phi": prof.sp,   "sigma_phi_err": prof.spe,
            "sigma_Z": prof.sZ,     "sigma_Z_err": prof.sZe,
            "sigma_RZ2": prof.crz,  "sigma_RZ2_err": prof.crze,
        })
        prof_path = outd / f"chem_{name}_velocity_dispersion_profiles.csv"
        prof_save.to_csv(prof_path, index=False)

        # Fit sigma_Z and sigma_RZ
        fa, _ = fit_bins(prof, zmx=pp["zmax"], robust=True, clip=clip,
                         sigz_model=sigz_model, pp=pp)
        fit_suffix = "linear_fits" if sigz_model == "linear" else f"{sigz_model}_fits"
        fa.to_csv(outd / f"chem_{name}_{fit_suffix}.csv", index=False)

        # Compute surface density curves (Σ vs Z at each R bin)
        # Include exact Z values used by fig8/fig7 (0.3, 1.0, 2.0, 3.0 kpc)
        Z_special = [z for z in [0.3, 1.0, 2.0, 3.0] if z <= pp["zmax"]]
        Z_eval = np.unique(np.concatenate([
            np.linspace(0.01, pp["zmax"], 200), Z_special]))
        surf_rows = []
        for _, row in fa.iterrows():
            med, p16, p84 = mc_sig(row, pp, Z_eval, nmc=nmc, rng=rng)
            for zi, z in enumerate(Z_eval):
                surf_rows.append({
                    "R_mid": row.rm, "Z_abs": z,
                    "Sigma_median_Msun_pc2": float(med[zi]),
                    "Sigma_p16_Msun_pc2":    float(p16[zi]),
                    "Sigma_p84_Msun_pc2":    float(p84[zi]),
                })
        surf_df = pd.DataFrame(surf_rows)
        surf_df.to_csv(outd / f"chem_{name}_surface_density_curves.csv", index=False)

        item = dict(
            key=name, label=label, color=color, ls=ls, marker=marker,
            profile=_profile_overlay_df(prof_path),
        )
        if not surf_df.empty:
            item["surface"] = surf_df
        overlays.append(item)
        raw_profiles[name] = prof   # store cprof DataFrame for caller
        print(f"  {name}: done ({len(prof):,} profile points, "
              f"{len(fa)} R-bins with Jeans fits)")

    # Return overlays AND the raw cprof profiles (caller can run linear fits etc.)
    return overlays, raw_profiles


def compute_chem_crz_from_raw(da, thin_mask, thick_mask,
                               nboot, rbw, rng):
    """
    Compute sigma_RZ^2 vs R at fixed Z slices for combined thin and thick.
    Returns (df_thin, df_thick) with columns z_target, R_mid, sigma_RZ2, sigma_RZ2_err.
    """
    def _crz_df(prof, z_targets):
        rows = []
        for zt in z_targets:
            for Rv, yv, ev in crz_at_z(prof, zt):
                rows.append({"z_target": zt, "R_mid": Rv,
                             "sigma_RZ2": yv, "sigma_RZ2_err": ev})
        return pd.DataFrame(rows)

    df_thin = pd.DataFrame()
    n_thin = int(thin_mask.sum())
    if n_thin >= TN["ng"]:
        print(f"  CRZ thin: {n_thin:,} stars...", flush=True)
        prof = _cprof_subset(da, thin_mask, TN["ng"], rbw, nboot, rng, label="CRZ-thin")
        if not prof.empty:
            df_thin = _crz_df(prof, [-1, 1])

    df_thick = pd.DataFrame()
    n_thick = int(thick_mask.sum())
    if n_thick >= TK["ng"]:
        print(f"  CRZ thick: {n_thick:,} stars...", flush=True)
        prof = _cprof_subset(da, thick_mask, TK["ng"], rbw, nboot, rng, label="CRZ-thick")
        if not prof.empty:
            df_thick = _crz_df(prof, [-2, -1, 1, 2])

    return df_thin, df_thick


# ---------------------------------------------------------------------------
# All-star effective tracer density fitting
# Estimates hZ_eff and hR_eff directly from star counts so the Jeans equation
# uses data-driven density gradients rather than arbitrary fixed scale heights.
#
# WARNING: This is an effective tracer-density model, NOT a physical thin/thick
#          decomposition. Raw counts are affected by survey selection function.
#          Use chemical thin/thick mode as validation; use all-star mode as
#          a high-statistics effective-tracer analysis.
# ---------------------------------------------------------------------------

def fit_allstar_density_gradients(R_data, Z_data,
                                   rl0=4.0, rl1=12.0, rbw=0.5,
                                   zmax=4.0, dz=0.3,
                                   R0=8.122, Nmin=50,
                                   n_boot=200, rng=None, outd=None):
    """Fit effective double-exponential tracer density from all-star counts.

    Bins stars in (R, |Z|) cells, estimates ln(N/V), and fits:
        ln nu = A + bR*(R - R0) + bZ*|Z|
    giving hZ_eff = -1/bZ and hR_eff = -1/bR.

    Returns a dict with:
        hZ_eff, hR_eff  – global fitted scale heights [kpc]
        hZ_lo/hi, hR_lo/hi  – 16th/84th bootstrap percentiles
        hZ_boot, hR_boot  – full bootstrap arrays (for mc_sig sampling)
        bins_df  – DataFrame of bins used in fit
        bZ, bR, A  – raw linear-fit coefficients
        local_grad  – callable (R,Z) -> (dlnnu_dZ, dlnnu_dR) for local mode
    """
    if rng is None:
        rng = np.random.default_rng(42)

    R_arr   = np.asarray(R_data, dtype=float)
    Zabs    = np.abs(np.asarray(Z_data, dtype=float))

    # Spatial selection matching kinematics analysis
    sel = (R_arr >= rl0) & (R_arr < rl1) & (Zabs <= zmax)
    R_arr, Zabs = R_arr[sel], Zabs[sel]

    R_edges = np.arange(rl0, rl1 + rbw * 0.5, rbw)
    Z_edges = np.arange(0.0, zmax + dz * 0.5, dz)

    rows = []
    for ri in range(len(R_edges) - 1):
        rl, rh = R_edges[ri], R_edges[ri + 1]
        Rmid = 0.5 * (rl + rh)
        for zi in range(len(Z_edges) - 1):
            zl, zh = Z_edges[zi], Z_edges[zi + 1]
            Zmid = 0.5 * (zl + zh)
            mask = (R_arr >= rl) & (R_arr < rh) & (Zabs >= zl) & (Zabs < zh)
            N = int(mask.sum())
            if N < Nmin:
                continue
            # Cylindrical annular volume; ×2 because |Z| bins fold ±Z together
            V = np.pi * (rh**2 - rl**2) * (zh - zl) * 2.0
            rows.append(dict(R_mid=Rmid, Z_mid=Zmid, N=N, V=V,
                             lnnu=np.log(N / V)))
    bins_df = pd.DataFrame(rows)

    if len(bins_df) < 6:
        warnings.warn("Too few density bins for all-star density fit; using default hZ=0.7, hR=2.3")
        return dict(hZ_eff=0.7, hR_eff=2.3, hZ_lo=0.56, hZ_hi=0.84,
                    hR_lo=1.84, hR_hi=2.76, hZ_boot=np.array([]),
                    hR_boot=np.array([]), bins_df=bins_df, bZ=np.nan,
                    bR=np.nan, A=np.nan, local_grad=None, n_bins=0)

    x = np.column_stack([np.ones(len(bins_df)),
                         bins_df.R_mid.values - R0,
                         bins_df.Z_mid.values])
    y = bins_df.lnnu.values

    try:
        coeffs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        A, bR, bZ = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except Exception as exc:
        warnings.warn(f"All-star density fit failed ({exc}); using default hZ=0.7, hR=2.3")
        return dict(hZ_eff=0.7, hR_eff=2.3, hZ_lo=0.56, hZ_hi=0.84,
                    hR_lo=1.84, hR_hi=2.76, hZ_boot=np.array([]),
                    hR_boot=np.array([]), bins_df=bins_df, bZ=np.nan,
                    bR=np.nan, A=np.nan, local_grad=None, n_bins=0)

    if bR >= 0:
        warnings.warn(f"All-star density fit gives bR={bR:.4f} >= 0; using hR=2.3")
        bR = -1.0 / 2.3
    if bZ >= 0:
        warnings.warn(f"All-star density fit gives bZ={bZ:.4f} >= 0; using hZ=0.7")
        bZ = -1.0 / 0.7

    hZ_eff = -1.0 / bZ
    hR_eff = -1.0 / bR

    # Bootstrap on density bins
    n = len(bins_df)
    hZ_boot_list, hR_boot_list = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        try:
            cb, _, _, _ = np.linalg.lstsq(xb, yb, rcond=None)
            _, bRb, bZb = float(cb[0]), float(cb[1]), float(cb[2])
            if bRb < 0 and bZb < 0:
                hZ_boot_list.append(-1.0 / bZb)
                hR_boot_list.append(-1.0 / bRb)
        except Exception:
            pass

    hZ_b = np.array(hZ_boot_list)
    hR_b = np.array(hR_boot_list)
    hZ_lo = float(np.nanpercentile(hZ_b, 16)) if len(hZ_b) > 5 else hZ_eff * 0.8
    hZ_hi = float(np.nanpercentile(hZ_b, 84)) if len(hZ_b) > 5 else hZ_eff * 1.2
    hR_lo = float(np.nanpercentile(hR_b, 16)) if len(hR_b) > 5 else hR_eff * 0.8
    hR_hi = float(np.nanpercentile(hR_b, 84)) if len(hR_b) > 5 else hR_eff * 1.2

    # Local gradient function (fitted_global: uniform; fitted_local: smoothed grid)
    def local_grad_global(R_val, Z_val):
        return float(bZ), float(bR)

    result = dict(hZ_eff=hZ_eff, hR_eff=hR_eff,
                  hZ_lo=hZ_lo, hZ_hi=hZ_hi,
                  hR_lo=hR_lo, hR_hi=hR_hi,
                  hZ_boot=hZ_b, hR_boot=hR_b,
                  bZ=bZ, bR=bR, A=A,
                  bins_df=bins_df, local_grad=local_grad_global,
                  n_bins=len(bins_df))

    if outd is not None:
        bins_df.to_csv(outd / "allstar_density_fit_bins.csv", index=False, float_format="%.6f")
        summary = [
            "ALL-STAR EFFECTIVE TRACER DENSITY FIT",
            "  WARNING: Effective tracer-density model, NOT physical thin/thick decomposition.",
            "  WARNING: Raw counts affected by survey selection function.",
            "  WARNING: Use chem thin/thick mode as validation.",
            "",
            "Global fit:  ln nu = A + bR*(R-R0) + bZ*|Z|",
            f"  A        = {A:.4f}",
            f"  bR       = {bR:.4f}  ->  hR_eff = {hR_eff:.3f} kpc "
            f"  [16th/84th: {hR_lo:.3f} / {hR_hi:.3f}]",
            f"  bZ       = {bZ:.4f}  ->  hZ_eff = {hZ_eff:.3f} kpc "
            f"  [16th/84th: {hZ_lo:.3f} / {hZ_hi:.3f}]",
            f"  n_bins   = {len(bins_df)} (N >= {Nmin} per bin)",
            f"  n_boot   = {n_boot}  valid_boot = {len(hZ_b)}",
        ]
        (outd / "allstar_density_fit_summary.txt").write_text("\n".join(summary) + "\n")

    return result


def plot_allstar_density_diagnostics(density_result, outd):
    """Diagnostic plots for the all-star effective tracer density fit."""
    bins_df = density_result.get("bins_df")
    if bins_df is None or bins_df.empty:
        return

    bZ  = density_result["bZ"]
    bR  = density_result["bR"]
    A   = density_result["A"]
    hZ  = density_result["hZ_eff"]
    hR  = density_result["hR_eff"]

    # ---- Fig 1: ln(N/V) vs |Z| per R bin with global fit overlay ----
    R_bins = sorted(bins_df.R_mid.unique())
    nc = min(4, len(R_bins))
    nr = int(np.ceil(len(R_bins) / nc))
    fig, ax2d = plt.subplots(nr, nc, figsize=(nc * 3.5, nr * 3.0), squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(R_bins)))
    for pi, Rm in enumerate(R_bins):
        ax = ax2d[divmod(pi, nc)[0], divmod(pi, nc)[1]]
        sub = bins_df[np.isclose(bins_df.R_mid, Rm)].sort_values("Z_mid")
        ax.scatter(sub.Z_mid, sub.lnnu, color=cmap[pi], s=20, label="data")
        zg = np.linspace(sub.Z_mid.min() - 0.1, sub.Z_mid.max() + 0.1, 100)
        ax.plot(zg, A + bR * (Rm - R0) + bZ * zg, "k--", lw=1.5, label="global fit")
        ax.set_title(f"R={Rm:.1f} kpc", fontsize=8)
        ax.tick_params(labelsize=7)
        if divmod(pi, nc)[1] == 0:
            ax.set_ylabel(r"$\ln(N/V)$", fontsize=8)
        if divmod(pi, nc)[0] == nr - 1:
            ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=8)
    for pi in range(len(R_bins), nr * nc):
        ax2d[divmod(pi, nc)[0], divmod(pi, nc)[1]].axis("off")
    fig.suptitle(f"All-star density: $h_Z^{{\\rm eff}}={hZ:.2f}$ kpc, "
                 f"$h_R^{{\\rm eff}}={hR:.2f}$ kpc", fontsize=10)
    fig.tight_layout()
    fig.savefig(outd / "fig_allstar_density_fit.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: ln(N/V) vs R at fixed |Z| slices ----
    Z_bins = sorted(bins_df.Z_mid.unique())
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    cols2 = plt.cm.plasma(np.linspace(0.1, 0.9, len(Z_bins)))
    Rg = np.linspace(bins_df.R_mid.min() - 0.2, bins_df.R_mid.max() + 0.2, 200)
    for Zm, col in zip(Z_bins, cols2):
        sub = bins_df[np.isclose(bins_df.Z_mid, Zm)].sort_values("R_mid")
        if len(sub) < 2:
            continue
        ax2.scatter(sub.R_mid, sub.lnnu, color=col, s=18, label=f"|Z|={Zm:.1f}")
        ax2.plot(Rg, A + bR * (Rg - R0) + bZ * Zm, color=col, lw=1.4)
    ax2.axhline(0, color="0.4", lw=0.8, ls=":")
    ax2.set_xlabel(r"$R$ [kpc]")
    ax2.set_ylabel(r"$\ln(N/V)$")
    ax2.set_title(f"All-star density: $h_R^{{\\rm eff}}={hR:.2f}$ kpc", fontsize=10)
    ax2.legend(frameon=False, fontsize=7, ncol=3)
    ax2.tick_params(direction="in", top=True, right=True)
    fig2.tight_layout()
    fig2.savefig(outd / "fig_allstar_density_fit_R.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    # ---- Fig 3: bootstrap distributions ----
    hZ_b = density_result.get("hZ_boot", np.array([]))
    hR_b = density_result.get("hR_boot", np.array([]))
    if len(hZ_b) > 5:
        fig3, axes3 = plt.subplots(1, 2, figsize=(10, 4))
        for ax, vals, lbl, ref in [
            (axes3[0], hZ_b, r"$h_Z^{\rm eff}$ [kpc]", hZ),
            (axes3[1], hR_b, r"$h_R^{\rm eff}$ [kpc]", hR),
        ]:
            ax.hist(vals, bins=30, color="#2166ac", alpha=0.7, density=True)
            ax.axvline(ref, color="k", lw=2)
            ax.axvline(np.nanpercentile(vals, 16), color="k", lw=1, ls="--")
            ax.axvline(np.nanpercentile(vals, 84), color="k", lw=1, ls="--")
            ax.set_xlabel(lbl)
            ax.tick_params(direction="in", top=True, right=True)
        fig3.tight_layout()
        fig3.savefig(outd / "fig_allstar_density_bootstrap.png", dpi=130, bbox_inches="tight")
        plt.close(fig3)


def jeans_comps_general_density(z, R, a, b, m, hs,
                                 dlnnu_dz_fn, dlnnu_dR_val,
                                 rz_b=0.0, sigz_model="linear",
                                 sigz_q=np.nan, sigz_s0sq=np.nan,
                                 sigz_k=np.nan, sigz_L=np.nan, sigz_s0=np.nan):
    """Jeans equation with general (data-fitted) density gradients.

    dlnnu_dz_fn : callable (R, z) -> scalar, or scalar constant
    dlnnu_dR_val: scalar d ln nu / dR at this R
    """
    z = np.asarray(z, dtype=float)
    params = dict(a=a, b=b, q=sigz_q, s0sq=sigz_s0sq, k=sigz_k, L=sigz_L, s0=sigz_s0)
    sz, sz2, dsz2 = sigz_model_eval(z, sigz_model, params)
    srz2 = rz_b + m * z

    if callable(dlnnu_dz_fn):
        _dlnn_z = np.array([float(dlnnu_dz_fn(R, float(zi))) for zi in z])
    else:
        _dlnn_z = float(dlnnu_dz_fn) * np.ones_like(z)

    _dlnn_R = float(dlnnu_dR_val)
    tc = 1.0 / R + _dlnn_R - 1.0 / hs
    bracket = dsz2 + sz2 * _dlnn_z + srz2 * tc
    S = -bracket / TG / 1e6

    hZ_loc = np.where(np.abs(_dlnn_z) > 1e-9, -1.0 / _dlnn_z, np.inf)
    hR_loc = (-1.0 / _dlnn_R) if abs(_dlnn_R) > 1e-9 else np.inf

    return dict(z=z, bracket=bracket, Sig=S,
                den_term=-sz2 * _dlnn_z / TG / 1e6,
                grad_term=-dsz2 / TG / 1e6,
                tilt_term=-srz2 * tc / TG / 1e6,
                sz=sz, sz2=sz2, srz2=srz2, tc=tc,
                dlnnu_dz=_dlnn_z, dlnnu_dR=_dlnn_R,
                hZ_eff_local=hZ_loc, hR_eff_local=float(hR_loc))


# ---------------------------------------------------------------------------
# End density-fitting module
# ---------------------------------------------------------------------------


def wls(x, y, ye=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.full((2, 2), np.nan)
    try:
        c, cov = np.polyfit(x, y, 1, cov=True)
    except Exception:
        c = np.polyfit(x, y, 1)
        cov = np.full((2, 2), np.nan)
    return c[0], c[1], cov


def wls0(x, y, ye=None):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan
    w = np.ones_like(x)
    sl = np.sum(w * x * y) / np.sum(w * x * x)
    r = y - sl * x
    c2r = np.sum(w * r ** 2) / max(len(x) - 1, 1)
    return float(sl), float(np.sqrt(c2r / np.sum(w * x * x)))


def robust_wls(x, y, max_iter=8, clip=3.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = np.isfinite(x) & np.isfinite(y)
    use = base.copy()
    coeff = np.array([np.nan, np.nan])
    cov = np.full((2, 2), np.nan)
    for _ in range(max_iter):
        if np.sum(use) < 3:
            break
        a, b, cov = wls(x[use], y[use])
        coeff = np.array([a, b])
        resid = y - (coeff[0] * x + coeff[1])
        ruse = resid[use]
        med = np.nanmedian(ruse)
        sig = 1.4826 * np.nanmedian(np.abs(ruse - med))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = base & (np.abs(resid - med) <= clip * sig)
        if np.array_equal(new_use, use):
            break
        use = new_use
    return coeff[0], coeff[1], cov, int(np.sum(use))


def robust_wls0(x, y, max_iter=8, clip=3.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    base = np.isfinite(x) & np.isfinite(y)
    use = base.copy()
    slope = err = np.nan
    for _ in range(max_iter):
        if np.sum(use) < 3:
            break
        slope, err = wls0(x[use], y[use])
        resid = y - slope * x
        ruse = resid[use]
        med = np.nanmedian(ruse)
        sig = 1.4826 * np.nanmedian(np.abs(ruse - med))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = base & (np.abs(resid - med) <= clip * sig)
        if np.array_equal(new_use, use):
            break
        use = new_use
    return slope, err, int(np.sum(use))


def _safe_sech2(x):
    x = np.clip(np.asarray(x, dtype=float), -50.0, 50.0)
    return 1.0 / np.cosh(x) ** 2


def _bic_from_resid(resid, npar):
    resid = np.asarray(resid, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size <= npar:
        return np.nan, np.nan, np.nan
    rss = float(np.sum(resid ** 2))
    rss = max(rss, TINY)
    rmse = float(np.sqrt(rss / resid.size))
    bic = float(resid.size * np.log(rss / resid.size) + npar * np.log(resid.size))
    return rss, rmse, bic


def sigz_model_eval(zabs, model, params):
    zabs = np.abs(np.asarray(zabs, dtype=float))
    if model == "quadratic":
        q = float(params.get("q", np.nan))
        s0sq = float(params.get("s0sq", np.nan))
        sz2 = q * zabs ** 2 + s0sq
        sz = np.sqrt(np.clip(sz2, 0.0, np.inf))
        dsz2 = 2.0 * q * zabs
    elif model == "tanh":
        k = float(params.get("k", np.nan))
        L = max(float(params.get("L", np.nan)), 0.03)
        s0 = float(params.get("s0", np.nan))
        x = zabs / L
        sz = k * zabs * np.tanh(x) + s0
        dsig = k * (np.tanh(x) + x * _safe_sech2(x))
        sz2 = sz ** 2
        dsz2 = 2.0 * sz * dsig
    else:
        a = float(params.get("a", np.nan))
        b = float(params.get("b", np.nan))
        sz = a * zabs + b
        sz2 = sz ** 2
        dsz2 = 2.0 * a * sz
    return sz, sz2, dsz2


def _fit_quadratic_sigz(zabs, sigz, robust=True, clip=3.0, max_iter=8):
    zabs = np.asarray(zabs, dtype=float)
    sigz = np.asarray(sigz, dtype=float)
    base = np.isfinite(zabs) & np.isfinite(sigz) & (sigz > 0)
    use = base.copy()
    popt = None
    pcov = np.full((2, 2), np.nan)

    def mod(zv, q, s0sq):
        return q * zv ** 2 + s0sq

    for _ in range(max_iter if robust else 1):
        if np.sum(use) < 4:
            return None
        zz = zabs[use]
        yy = sigz[use] ** 2
        p0 = [
            max((np.nanpercentile(yy, 90) - np.nanpercentile(yy, 10)) / max(np.nanmax(zz) ** 2, 0.25), 1.0),
            max(np.nanpercentile(sigz[use], 5) ** 2, 1.0),
        ]
        try:
            popt, pcov = curve_fit(mod, zz, yy, p0=p0, bounds=([0.0, 1.0], [2.0e5, 2.0e5]),
                                   maxfev=50000)
        except Exception:
            return None
        if not robust:
            break
        pred = np.sqrt(np.clip(mod(zabs, *popt), 0.0, np.inf))
        resid = sigz - pred
        ruse = resid[use]
        med = np.nanmedian(ruse)
        sig = 1.4826 * np.nanmedian(np.abs(ruse - med))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = base & (np.abs(resid - med) <= clip * sig)
        if np.array_equal(new_use, use):
            break
        use = new_use
    params = {"q": float(popt[0]), "s0sq": float(popt[1])}
    pred = sigz_model_eval(zabs[base], "quadratic", params)[0]
    rss, rmse, bic = _bic_from_resid(sigz[base] - pred, 2)
    return dict(model="quadratic", params=params, cov=pcov, n_used=int(np.sum(use)),
                rss=rss, rmse=rmse, bic=bic)


def _fit_tanh_sigz(zabs, sigz, robust=True, clip=3.0, max_iter=8):
    zabs = np.asarray(zabs, dtype=float)
    sigz = np.asarray(sigz, dtype=float)
    base = np.isfinite(zabs) & np.isfinite(sigz) & (sigz > 0)
    use = base.copy()
    popt = None
    pcov = np.full((3, 3), np.nan)

    def mod(zv, k, L, s0):
        return k * zv * np.tanh(zv / L) + s0

    for _ in range(max_iter if robust else 1):
        if np.sum(use) < 5:
            return None
        zz = zabs[use]
        yy = sigz[use]
        p0 = [
            max((np.nanpercentile(yy, 90) - np.nanpercentile(yy, 10)) / max(np.nanmax(zz), 0.5), 1.0),
            0.5,
            max(np.nanpercentile(yy, 5), 1.0),
        ]
        try:
            popt, pcov = curve_fit(mod, zz, yy, p0=p0,
                                   bounds=([0.0, 0.03, 0.0], [300.0, 20.0, 300.0]),
                                   maxfev=50000)
        except Exception:
            return None
        if not robust:
            break
        resid = sigz - mod(zabs, *popt)
        ruse = resid[use]
        med = np.nanmedian(ruse)
        sig = 1.4826 * np.nanmedian(np.abs(ruse - med))
        if not np.isfinite(sig) or sig <= 0:
            break
        new_use = base & (np.abs(resid - med) <= clip * sig)
        if np.array_equal(new_use, use):
            break
        use = new_use
    params = {"k": float(popt[0]), "L": float(popt[1]), "s0": float(popt[2])}
    pred = sigz_model_eval(zabs[base], "tanh", params)[0]
    rss, rmse, bic = _bic_from_resid(sigz[base] - pred, 3)
    return dict(model="tanh", params=params, cov=pcov, n_used=int(np.sum(use)),
                rss=rss, rmse=rmse, bic=bic)


def fit_sigz_models(zabs, sigz, robust=True, clip=3.0, choice="best"):
    zabs = np.asarray(zabs, dtype=float)
    sigz = np.asarray(sigz, dtype=float)
    base = np.isfinite(zabs) & np.isfinite(sigz) & (sigz > 0)
    fits = []
    if np.sum(base) < 4:
        return None, []

    if robust:
        a, b, cov, n_used = robust_wls(zabs[base], sigz[base], clip=clip)
    else:
        a, b, cov = wls(zabs[base], sigz[base])
        n_used = int(np.sum(base))
    lin_params = {"a": float(a), "b": float(b)}
    lin_pred = sigz_model_eval(zabs[base], "linear", lin_params)[0]
    rss, rmse, bic = _bic_from_resid(sigz[base] - lin_pred, 2)
    fits.append(dict(model="linear", params=lin_params, cov=cov, n_used=n_used,
                     rss=rss, rmse=rmse, bic=bic))

    quad = _fit_quadratic_sigz(zabs, sigz, robust=robust, clip=clip)
    if quad is not None:
        fits.append(quad)
    tanh = _fit_tanh_sigz(zabs, sigz, robust=robust, clip=clip)
    if tanh is not None:
        fits.append(tanh)

    valid = [f for f in fits if np.isfinite(f["bic"])]
    if not valid:
        return fits[0], fits
    if choice != "best":
        forced = [f for f in valid if f["model"] == choice]
        return (forced[0] if forced else valid[0]), fits
    return min(valid, key=lambda f: f["bic"]), fits


def _surface_safe_best_sigz(all_models, R, a, b, rz_b, rz_m, pp, zmx):
    valid = [f for f in all_models if np.isfinite(f["bic"])]
    if not valid or pp is None:
        return min(valid, key=lambda f: f["bic"]) if valid else None
    # Include small z so that models with steep dσ²_Z/dZ near the mid-plane
    # (the linear model can have bracket > 0 at z < 0.3 kpc) are correctly
    # rejected in favour of quadratic/tanh whose gradient vanishes at z=0.
    zgrid = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 1.5, 2.0, 3.0], dtype=float)
    zgrid = zgrid[zgrid <= zmx + 1e-9]
    safe = []
    for f in valid:
        params = f["params"]
        kwargs = dict(sigz_model=f["model"])
        if f["model"] == "quadratic":
            kwargs.update(sigz_q=params["q"], sigz_s0sq=params["s0sq"])
        elif f["model"] == "tanh":
            kwargs.update(sigz_k=params["k"], sigz_L=params["L"], sigz_s0=params["s0"])
        try:
            c = jeans_comps(zgrid, R, a, b, rz_m, pp["hZ"], pp["hR"], pp["hs"],
                            rz_b=rz_b, **kwargs)
        except Exception:
            continue
        sig = c["Sig"]
        if np.all(np.isfinite(sig)) and np.nanmin(sig) >= -1e-8:
            safe.append(f)
    if safe:
        return min(safe, key=lambda f: f["bic"])
    # No σ_Z model produces non-negative Σ(R,|Z|) — physics failure (not a fitting artefact).
    # B(R,Z) = dσ²_Z/dZ + σ²_Z ∂lnν/∂Z + σ²_RZ·T_c  must be ≤ 0 everywhere.
    # With σ_RZ²(0)=0 enforced (odd-linear), a positive bracket means σ_Z slope is too steep,
    # hZ is inconsistent with the kinematics, or T_c sign is wrong.  Investigate.
    diag = []
    for f in valid:
        kw = dict(sigz_model=f["model"])
        if f["model"] == "quadratic":
            kw.update(sigz_q=f["params"].get("q", np.nan), sigz_s0sq=f["params"].get("s0sq", np.nan))
        elif f["model"] == "tanh":
            kw.update(sigz_k=f["params"].get("k", np.nan), sigz_L=f["params"].get("L", np.nan),
                      sigz_s0=f["params"].get("s0", np.nan))
        try:
            sig_vals = jeans_comps(zgrid, R, a, b, rz_m, pp["hZ"], pp["hR"], pp["hs"],
                                   rz_b=rz_b, **kw)["Sig"]
            diag.append(f"{f['model']}:Σmin={np.nanmin(sig_vals):.1f}")
        except Exception as exc:
            diag.append(f"{f['model']}:error({exc})")
    raise RuntimeError(
        f"No σ_Z model yields Σ≥0 at R={R:.2f} kpc "
        f"(hZ={pp['hZ']:.3f}, hR={pp['hR']:.2f}, hs={pp['hs']:.2f}, rz_m={rz_m:.2f}). "
        + "; ".join(diag)
    )


def fit_bins(pf, zmx=4.0, rl0=4, rl1=12, robust=True, clip=3.0, sigz_model="linear", pp=None):
    rows = []
    model_rows = []
    for rl in sorted(pf.rl.unique()):
        rh = float(pf.loc[pf.rl == rl, "rh"].iloc[0]) if "rh" in pf.columns else rl + 1
        if rl < rl0 or rl >= rl1:
            continue
        sub = pf[(pf.rl == rl) & np.isfinite(pf.sZ) & np.isfinite(pf.crz)].copy()
        sub = sub[np.abs(sub.zm) <= zmx]
        if len(sub) < 4:
            continue
        az = np.abs(sub.zm.values)
        sz = sub.sZ.values.astype(float)
        best, all_models = fit_sigz_models(az, sz, robust=robust, clip=clip, choice=sigz_model)
        if best is None:
            continue
        # σ_RZ²(Z) must be an odd function of Z (mid-plane symmetry: σ_RZ²(0)=0).
        # Cheng (2024) eq. 1 requires the tilt term srz2 * tc to vanish at Z=0.
        # Using wls0 (through-origin WLS) enforces this constraint; a non-zero
        # intercept from plain OLS can flip the sign of the tilt correction and
        # produce unphysical negative surface densities at low |Z|.
        rz_m, rz_e = wls0(sub.zm.values, sub.crz.values)
        rz_b = 0.0          # odd-linear by construction
        n_rz_used = len(sub)
        lin = next((m for m in all_models if m["model"] == "linear"), best)
        a = lin["params"].get("a", np.nan)
        b = lin["params"].get("b", np.nan)
        cov = lin["cov"]
        if sigz_model == "best":
            try:
                best = _surface_safe_best_sigz(all_models, 0.5 * (rl + rh), a, b, rz_b, rz_m, pp, zmx)
            except RuntimeError as _e:
                print(f"  [fit_bins] R=[{rl},{rh}] kpc skipped — {_e}")
                continue
            if best is None:
                continue
        n_sz_used = best["n_used"]
        pos = sub[sub.zm > 0.2].crz
        fn = (pos < 0).mean() if len(pos) > 0 else np.nan
        row = dict(rl=rl, rh=rh, rm=0.5 * (rl + rh),
                   sigz_model=best["model"],
                   sigz_bic=best["bic"], sigz_rmse=best["rmse"], sigz_rss=best["rss"],
                   a=a, b=b, caa=cov[0, 0], cab=cov[0, 1], cbb=cov[1, 1],
                   sigz_q=np.nan, sigz_s0sq=np.nan,
                   sigz_q_var=np.nan, sigz_q_s0sq_cov=np.nan, sigz_s0sq_var=np.nan,
                   sigz_k=np.nan, sigz_L=np.nan, sigz_s0=np.nan,
                   sigz_k_var=np.nan, sigz_k_L_cov=np.nan, sigz_k_s0_cov=np.nan,
                   sigz_L_var=np.nan, sigz_L_s0_cov=np.nan, sigz_s0_var=np.nan,
                   rz_b=0.0, rz_be=0.0, m=rz_m, me=rz_e,
                   rz_b_var=0.0, rz_b_m_cov=0.0, rz_m_var=rz_e ** 2 if np.isfinite(rz_e) else np.nan,
                   n=len(sub), n_sigz_fit=n_sz_used, n_rz_fit=n_rz_used, fn=fn)
        quad_fit = next((mfit for mfit in all_models if mfit["model"] == "quadratic"), None)
        tanh_fit = next((mfit for mfit in all_models if mfit["model"] == "tanh"), None)
        if quad_fit is not None:
            bcov = quad_fit["cov"]
            row.update(sigz_q=quad_fit["params"]["q"], sigz_s0sq=quad_fit["params"]["s0sq"],
                       sigz_q_var=bcov[0, 0], sigz_q_s0sq_cov=bcov[0, 1], sigz_s0sq_var=bcov[1, 1])
        if tanh_fit is not None:
            bcov = tanh_fit["cov"]
            row.update(sigz_k=tanh_fit["params"]["k"], sigz_L=tanh_fit["params"]["L"], sigz_s0=tanh_fit["params"]["s0"],
                       sigz_k_var=bcov[0, 0], sigz_k_L_cov=bcov[0, 1], sigz_k_s0_cov=bcov[0, 2],
                       sigz_L_var=bcov[1, 1], sigz_L_s0_cov=bcov[1, 2], sigz_s0_var=bcov[2, 2])
        rows.append(row)
        for mfit in all_models:
            model_rows.append(dict(rl=rl, rh=rh, rm=0.5 * (rl + rh),
                                   model=mfit["model"], selected=(mfit["model"] == best["model"]),
                                   rss=mfit["rss"], rmse=mfit["rmse"], bic=mfit["bic"],
                                   n_sigz_fit=mfit["n_used"]))
    return pd.DataFrame(rows), pd.DataFrame(model_rows)


def jeans_comps(z, R, a, b, m, hZ, hR, hs, law="exp", rz_b=0.0,
                sigz_model="linear", sigz_q=np.nan, sigz_s0sq=np.nan,
                sigz_k=np.nan, sigz_L=np.nan, sigz_s0=np.nan):
    z = np.asarray(z, dtype=float)
    params = dict(a=a, b=b, q=sigz_q, s0sq=sigz_s0sq, k=sigz_k, L=sigz_L, s0=sigz_s0)
    sz, sz2, dsz2 = sigz_model_eval(z, sigz_model, params)
    srz2 = rz_b + m * z
    if law == "exp":
        dlnn = -1.0 / hZ
    elif law == "sech2":
        dlnn = -np.tanh(z / (2 * hZ)) / hZ
    else:
        raise ValueError(f"Unknown density law: {law}")
    tc = 1 / R - 1 / hR - 1 / hs
    bracket = dsz2 + sz2 * dlnn + srz2 * tc
    S = -bracket / TG / 1e6
    return dict(z=z, bracket=bracket, Sig=S,
                den_term=-sz2 * dlnn / TG / 1e6,
                grad_term=-dsz2 / TG / 1e6,
                tilt_term=-srz2 * tc / TG / 1e6,
                sz=sz, sz2=sz2, srz2=srz2, tc=tc)


def _row_get(row, key, default=np.nan):
    try:
        val = row[key]
    except Exception:
        val = getattr(row, key, default)
    if isinstance(val, str):
        return val
    try:
        if pd.isna(val):
            return default
    except Exception:
        pass
    return val


def _row_sigz_params(row):
    return dict(
        sigz_model=_row_get(row, "sigz_model", "linear"),
        sigz_q=_row_get(row, "sigz_q", np.nan),
        sigz_s0sq=_row_get(row, "sigz_s0sq", np.nan),
        sigz_k=_row_get(row, "sigz_k", np.nan),
        sigz_L=_row_get(row, "sigz_L", np.nan),
        sigz_s0=_row_get(row, "sigz_s0", np.nan),
    )


def sig_from_row(row, pp, z, law="exp"):
    c = jeans_comps(z, row.rm, row.a, row.b, row.m, pp["hZ"], pp["hR"], pp["hs"], law,
                    rz_b=_row_get(row, "rz_b", 0.0), **_row_sigz_params(row))
    return c["Sig"], c


def debug_negative_sigma(fa, pp, zvals=None, name=""):
    """Print a component breakdown for every R-bin that has Σ < 0.

    For each such bin, prints:
      - The three Jeans bracket terms at each z:
          grad_term  = -dσ²_Z/dZ / (TG·1e6)          [gradient of velocity dispersion]
          den_term   = -σ²_Z · ∂lnν/∂Z / (TG·1e6)   [density fall-off]
          tilt_term  = -σ²_RZ · T_c / (TG·1e6)       [cross-tilt correction]
      - srz2 = σ_RZ²(Z) = m·Z  (rz_b must be 0)
      - tc = 1/R − 1/hR − 1/hs  (tilt coefficient)
    A negative Σ means bracket > 0, i.e. tilt or gradient term dominates.
    """
    if zvals is None:
        zvals = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    header = f"=== debug_negative_sigma [{name}] ==="
    printed_header = False
    for _, row in fa.iterrows():
        sig, c = sig_from_row(row, pp, np.asarray(zvals, dtype=float))
        if np.any(sig < 0):
            if not printed_header:
                print(header)
                printed_header = True
            print(f"\n  R=[{row.rl:.1f},{row.rh:.1f}] kpc  rm={row.rm:.2f}  "
                  f"sigz_model={_row_get(row,'sigz_model','?')}  "
                  f"rz_b={_row_get(row,'rz_b',0.0):.3f}  m={row.m:.3f}  "
                  f"hZ={pp['hZ']:.3f} hR={pp['hR']:.2f} hs={pp['hs']:.2f}")
            print(f"  tc = 1/R-1/hR-1/hs = {c['tc']:.4f}")
            print(f"  {'|Z|':>6}  {'Sig':>8}  {'grad':>8}  {'den':>8}  {'tilt':>8}  "
                  f"{'srz2':>8}  {'bracket':>10}")
            for i, zv in enumerate(zvals):
                print(f"  {zv:6.2f}  {sig[i]:8.2f}  {c['grad_term'][i]:8.2f}  "
                      f"{c['den_term'][i]:8.2f}  {c['tilt_term'][i]:8.2f}  "
                      f"{c['srz2'][i]:8.2f}  {c['bracket'][i]:10.4f}")
    if not printed_header:
        print(f"{header}\n  All bins have Σ≥0 — no negative surface densities found.")


def _valid_cov(cov):
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return False
    if not np.all(np.isfinite(cov)):
        return False
    return np.linalg.eigvalsh(cov).min() >= -1e-8


def mc_sig(row, pp, z, nmc=400, law="exp", rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    z = np.asarray(z, dtype=float)
    med, _ = sig_from_row(row, pp, z, law)
    if nmc <= 1:
        return med, med, med
    model = str(_row_get(row, "sigz_model", "linear"))
    sigz_draws = None
    if model == "quadratic":
        cov_q = np.array([[_row_get(row, "sigz_q_var", np.nan), _row_get(row, "sigz_q_s0sq_cov", np.nan)],
                          [_row_get(row, "sigz_q_s0sq_cov", np.nan), _row_get(row, "sigz_s0sq_var", np.nan)]], dtype=float)
        mean_q = np.array([_row_get(row, "sigz_q", np.nan), _row_get(row, "sigz_s0sq", np.nan)], dtype=float)
        if _valid_cov(cov_q) and np.all(np.isfinite(mean_q)):
            sigz_draws = rng.multivariate_normal(mean_q, cov_q, nmc)
            sigz_draws[:, 0] = np.clip(sigz_draws[:, 0], 0.0, np.inf)
            sigz_draws[:, 1] = np.clip(sigz_draws[:, 1], 1.0, np.inf)
    elif model == "tanh":
        cov_t = np.array([
            [_row_get(row, "sigz_k_var", np.nan), _row_get(row, "sigz_k_L_cov", np.nan), _row_get(row, "sigz_k_s0_cov", np.nan)],
            [_row_get(row, "sigz_k_L_cov", np.nan), _row_get(row, "sigz_L_var", np.nan), _row_get(row, "sigz_L_s0_cov", np.nan)],
            [_row_get(row, "sigz_k_s0_cov", np.nan), _row_get(row, "sigz_L_s0_cov", np.nan), _row_get(row, "sigz_s0_var", np.nan)],
        ], dtype=float)
        mean_t = np.array([_row_get(row, "sigz_k", np.nan), _row_get(row, "sigz_L", np.nan), _row_get(row, "sigz_s0", np.nan)], dtype=float)
        if _valid_cov(cov_t) and np.all(np.isfinite(mean_t)):
            sigz_draws = rng.multivariate_normal(mean_t, cov_t, nmc)
            sigz_draws[:, 0] = np.clip(sigz_draws[:, 0], 0.0, np.inf)
            sigz_draws[:, 1] = np.clip(sigz_draws[:, 1], 0.03, 20.0)
            sigz_draws[:, 2] = np.clip(sigz_draws[:, 2], 0.0, np.inf)
    else:
        cov_ab = np.array([[row.caa, row.cab], [row.cab, row.cbb]], dtype=float)
        if _valid_cov(cov_ab):
            sigz_draws = rng.multivariate_normal([row.a, row.b], cov_ab, nmc)
    if sigz_draws is None:
        return med, med, med
    cov_rz = np.array([[_row_get(row, "rz_b_var", np.nan), _row_get(row, "rz_b_m_cov", np.nan)],
                       [_row_get(row, "rz_b_m_cov", np.nan), _row_get(row, "rz_m_var", np.nan)]], dtype=float)
    mean_rz = np.array([_row_get(row, "rz_b", 0.0), row.m], dtype=float)
    if _valid_cov(cov_rz):
        rz_draws = rng.multivariate_normal(mean_rz, cov_rz, nmc)
    else:
        rz_draws = np.column_stack([
            rng.normal(mean_rz[0], max(float(_row_get(row, "rz_be", 0.0)), 1e-4), nmc),
            rng.normal(mean_rz[1], max(float(row.me), 1e-4), nmc),
        ])
    fr = 0.10
    # If pp carries bootstrap distributions from all-star density fit, sample from them
    if "hZ_boot" in pp and len(pp["hZ_boot"]) > 5:
        idx = rng.integers(0, len(pp["hZ_boot"]), nmc)
        hZ = pp["hZ_boot"][idx].clip(0.05, 10)
    else:
        hZ = rng.normal(pp["hZ"], fr * pp["hZ"], nmc).clip(0.05, 10)
    if "hR_boot" in pp and len(pp["hR_boot"]) > 5:
        idx = rng.integers(0, len(pp["hR_boot"]), nmc)
        hR = pp["hR_boot"][idx].clip(0.5, 20)
    else:
        hR = rng.normal(pp["hR"], fr * pp["hR"], nmc).clip(0.5, 20)
    hs = rng.normal(pp["hs"], max(pp["hs_e"], 0.05 * pp["hs"]), nmc).clip(0.5, 50)
    curves = []
    for i in range(nmc):
        if model == "quadratic":
            c = jeans_comps(z, row.rm, row.a, row.b, rz_draws[i, 1], hZ[i], hR[i], hs[i], law,
                            rz_b=rz_draws[i, 0], sigz_model=model,
                            sigz_q=sigz_draws[i, 0], sigz_s0sq=sigz_draws[i, 1])
        elif model == "tanh":
            c = jeans_comps(z, row.rm, row.a, row.b, rz_draws[i, 1], hZ[i], hR[i], hs[i], law,
                            rz_b=rz_draws[i, 0], sigz_model=model,
                            sigz_k=sigz_draws[i, 0], sigz_L=sigz_draws[i, 1], sigz_s0=sigz_draws[i, 2])
        else:
            c = jeans_comps(z, row.rm, sigz_draws[i, 0], sigz_draws[i, 1], rz_draws[i, 1],
                            hZ[i], hR[i], hs[i], law, rz_b=rz_draws[i, 0], sigz_model=model)
        curves.append(c["Sig"])
    A = np.vstack(curves)
    return np.nanmedian(A, 0), np.nanpercentile(A, 16, 0), np.nanpercentile(A, 84, 0)


def baryonic_sigma(R, z):
    """Baryonic Σ(R,|Z|) following Bland-Hawthorn & Gerhard (2016) as used in Cheng+2024.

    Per Cheng+2024 footnote 3: the gas disc is treated as a CONSTANT infinitely thin layer
    with Σ_gas = 13.2 M⊙/pc² (no radial scaling), contributing its full surface density for
    any Z > 0.  The stellar thin and thick discs use the standard cumulative integral with
    their respective exponential scale heights.
    """
    z = np.abs(np.asarray(z, dtype=float))
    # Thin stellar disc (hz=0.30 kpc): cumulative integral
    S_thin = 35.0 * np.exp(-(R - R0) / 2.60) * (1 - np.exp(-z / 0.30))
    # Thick stellar disc (hz=0.90 kpc): cumulative integral
    S_thick = 7.0 * np.exp(-(R - R0) / 3.60) * (1 - np.exp(-z / 0.90))
    # Gas disc: constant thin sheet (no radial dependence), Σ_gas = 13.2 M⊙/pc² for any Z > 0
    S_gas = np.where(z > 0, 13.2, 0.0)
    return S_thin + S_thick + S_gas


def shm_sigma(R, z):
    """SHM Σ(R,|Z|): baryonic components (same as baryonic_sigma) + NFW dark matter halo.

    Gas disc: constant thin sheet (Σ_gas = 13.2 M⊙/pc², no radial dependence).
    Stellar discs: cumulative integrals.  DM halo: NFW cumulative integral.
    """
    z = np.abs(np.asarray(z, dtype=float))
    # Thin stellar disc (hz=0.30 kpc): cumulative integral
    S_thin = 35.0 * np.exp(-(R - R0) / 2.60) * (1 - np.exp(-z / 0.30))
    # Thick stellar disc (hz=0.90 kpc): cumulative integral
    S_thick = 7.0 * np.exp(-(R - R0) / 3.60) * (1 - np.exp(-z / 0.90))
    # Gas disc: constant thin sheet
    S_gas = np.where(z > 0, 13.2, 0.0)
    S = S_thin + S_thick + S_gas
    # NFW dark matter halo: cumulative integral 2∫₀^Z ρ_NFW dz
    rho_s, r_s = 0.0084e9, 10.8
    for i, zv in enumerate(z):
        if zv <= 0:
            continue
        zg = np.linspace(0, zv, 400)
        r = np.sqrt(R ** 2 + zg ** 2)
        rho = rho_s * (R0 / r) * ((1 + R0 / r_s) / (1 + r / r_s)) ** 2
        S[i] += 2 * trapz(rho, zg) / 1e6
    return S


def _fig6_z_grid(zmax=4.5, n=400):
    """Fig. 6 reference grid with an explicit Z=0 point."""
    return np.r_[0.0, np.linspace(0.01, zmax, n)]


def _fig6_sheet_limit_sigma(func, R, z):
    """Plot the infinitely thin gas sheet as the Z->0+ limit.

    The physical cumulative integral is zero exactly at Z=0, but Cheng+2024
    Fig. 6 follows the conventional plotted SHM curve where the 13.2
    Msun/pc^2 gas sheet is already present at the mid-plane.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(func(R, z), dtype=float).copy()
    zero = z == 0.0
    if np.any(zero):
        y[zero] = np.asarray(func(R, np.full(int(zero.sum()), 1e-9)), dtype=float)
    return y


def get_solar_row(fits, R=8.5):
    if fits.empty:
        return None
    return fits.iloc[np.argmin(np.abs(fits.rm - R))]


def save_surface_table(fn, fk, outd, rng):
    rows = []
    for nm_p, fits, pp in [("thin", fn, TN), ("thick", fk, TK)]:
        for _, row in fits.iterrows():
            for zv in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
                if zv > pp["zmax"] + 0.01:
                    continue
                med, lo, hi = mc_sig(row, pp, np.array([zv]), nmc=150, rng=rng)
                rows.append(dict(pop=nm_p, R=row.rm, Z=zv,
                                 Sig=med[0], Sig_lo=lo[0], Sig_hi=hi[0],
                                 Kz=s2k(med[0])))
    df = pd.DataFrame(rows)
    path = outd / "surface_density_table.csv"
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"Table saved -> {path}")
    return df


def fit_quality_table(pf, fits, outd, suffix):
    rows = []
    for _, row in fits.iterrows():
        sub = pf[(pf.rl == row.rl) & np.isfinite(pf.sZ) & np.isfinite(pf.crz)].copy()
        if sub.empty:
            continue
        az = np.abs(sub.zm.values)
        sz_pred = sigz_model_eval(az, str(_row_get(row, "sigz_model", "linear")),
                                  dict(a=row.a, b=row.b,
                                       q=_row_get(row, "sigz_q", np.nan),
                                       s0sq=_row_get(row, "sigz_s0sq", np.nan),
                                       k=_row_get(row, "sigz_k", np.nan),
                                       L=_row_get(row, "sigz_L", np.nan),
                                       s0=_row_get(row, "sigz_s0", np.nan)))[0]
        sz_res = sub.sZ.values - sz_pred
        rz_res = sub.crz.values - (_row_get(row, "rz_b", 0.0) + row.m * sub.zm.values)
        def stats(res):
            res = np.asarray(res, dtype=float)
            res = res[np.isfinite(res)]
            if res.size == 0:
                return dict(n=0, median=np.nan, mad=np.nan, frac_above=np.nan,
                            frac_abs_le_1mad=np.nan, frac_abs_le_2mad=np.nan)
            med = float(np.nanmedian(res))
            mad = float(1.4826 * np.nanmedian(np.abs(res - med)))
            scale = mad if np.isfinite(mad) and mad > 0 else np.nanstd(res)
            return dict(
                n=int(res.size),
                median=med,
                mad=float(mad),
                frac_above=float(np.mean(res > 0)),
                frac_abs_le_1mad=float(np.mean(np.abs(res - med) <= scale)) if scale > 0 else np.nan,
                frac_abs_le_2mad=float(np.mean(np.abs(res - med) <= 2 * scale)) if scale > 0 else np.nan,
            )
        s = stats(sz_res)
        r = stats(rz_res)
        rows.append(dict(
            R=row.rm,
            n_points=int(len(sub)),
            sigmaZ_median_residual=s["median"],
            sigmaZ_mad_residual=s["mad"],
            sigmaZ_frac_above_line=s["frac_above"],
            sigmaZ_frac_within_1mad=s["frac_abs_le_1mad"],
            sigmaZ_frac_within_2mad=s["frac_abs_le_2mad"],
            sigmaRZ_median_residual=r["median"],
            sigmaRZ_mad_residual=r["mad"],
            sigmaRZ_frac_above_line=r["frac_above"],
            sigmaRZ_frac_within_1mad=r["frac_abs_le_1mad"],
            sigmaRZ_frac_within_2mad=r["frac_abs_le_2mad"],
        ))
    df = pd.DataFrame(rows)
    df.to_csv(outd / f"fit_quality_{suffix}.csv", index=False)
    return df


def _profile_overlay_df(path):
    df = pd.read_csv(path)
    if {"R_mid", "Z_median", "sigma_R", "sigma_phi", "sigma_Z"}.issubset(df.columns):
        return pd.DataFrame({
            "rm": df["R_mid"], "zm": df["Z_median"],
            "sR": df["sigma_R"], "sRe": df.get("sigma_R_err", np.nan),
            "sp": df["sigma_phi"], "spe": df.get("sigma_phi_err", np.nan),
            "sZ": df["sigma_Z"], "sZe": df.get("sigma_Z_err", np.nan),
            "crz": df["sigma_RZ2"], "crze": df.get("sigma_RZ2_err", np.nan),
        })
    return df


def load_overlay_sets(base_dir, keys=None):
    if base_dir is None:
        return []
    base = Path(base_dir)
    keys = set(keys) if keys is not None else None
    sets = []
    for key, sty in OVERLAY_STYLES.items():
        if keys is not None and key not in keys:
            continue
        d = base / key
        prof = d / f"all_no_mgfe_{key}_velocity_dispersion_profiles.csv"
        surf = d / f"paper_style_no_mgfe_{key}_surface_density_curves_exp.csv"
        item = dict(key=key, **sty)
        if prof.exists():
            item["profile"] = _profile_overlay_df(prof)
        if surf.exists():
            item["surface"] = pd.read_csv(surf)
        if "profile" in item or "surface" in item:
            sets.append(item)
    return sets


def _overlay_profiles_on_axis(ax, overlays, rl, rh, ycol):
    for ov in overlays:
        pf = ov.get("profile")
        if pf is None or ycol not in pf.columns:
            continue
        center = 0.5 * (rl + rh)
        tol = max(0.51, 0.5 * (rh - rl) + 1e-6)
        sub = pf[np.abs(pf.rm - center) <= tol]
        if sub.empty:
            continue
        ax.plot(sub.zm, sub[ycol], linestyle="none", marker=ov["marker"],
                ms=2.4, color=ov["color"], alpha=0.55, label=ov["label"])


def _overlay_surface_z(ax, overlays, rm, zmax=4.0, label_once=True, zmin=0.0):
    for ov in overlays:
        sf = ov.get("surface")
        if sf is None:
            continue
        rvals = sf.R_mid.dropna().unique()
        if len(rvals) == 0:
            continue
        nearest = rvals[np.argmin(np.abs(rvals - rm))]
        sub = sf[np.isclose(sf.R_mid, nearest)]
        sub = sub[(sub.Z_abs >= zmin) & (sub.Z_abs <= zmax) & np.isfinite(sub.Sigma_median_Msun_pc2)].sort_values("Z_abs")
        if sub.empty:
            continue
        ax.plot(sub.Z_abs, sub.Sigma_median_Msun_pc2, color=ov["color"],
                ls=ov["ls"], lw=1.35, alpha=0.9, label=ov["label"] if label_once else None)
        if "Sigma_p16_Msun_pc2" in sub.columns and "Sigma_p84_Msun_pc2" in sub.columns:
            lo = np.maximum(sub.Sigma_p16_Msun_pc2.values, 0.0)
            hi = sub.Sigma_p84_Msun_pc2.values
            bk = np.isfinite(lo) & np.isfinite(hi)
            if bk.sum() > 1:
                ax.fill_between(sub.Z_abs.values[bk], lo[bk], hi[bk],
                                color=ov["color"], alpha=0.15, linewidth=0)


def _overlay_surface_r(ax, overlays, zval):
    for ov in overlays:
        sf = ov.get("surface")
        if sf is None:
            continue
        # Use tolerance of 0.06 kpc to handle linspace that doesn't hit zval exactly,
        # then take the nearest Z point per R_mid
        near = sf[np.abs(sf.Z_abs - zval) <= 0.06]
        if near.empty:
            continue
        sub = (near.assign(_dist=np.abs(near.Z_abs - zval))
                   .sort_values("_dist")
                   .drop_duplicates(subset="R_mid")
                   .sort_values("R_mid"))
        if sub.empty:
            continue
        sub = sub.sort_values("R_mid")
        ax.plot(sub.R_mid, sub.Sigma_median_Msun_pc2, color=ov["color"],
                ls=ov["ls"], marker=ov["marker"], ms=3, lw=1.25,
                alpha=0.85, label=ov["label"])
        if "Sigma_p16_Msun_pc2" in sub.columns and "Sigma_p84_Msun_pc2" in sub.columns:
            lo = np.maximum(sub.Sigma_p16_Msun_pc2.values, 0.0)
            hi = sub.Sigma_p84_Msun_pc2.values
            bk = np.isfinite(lo) & np.isfinite(hi)
            if bk.sum() > 1:
                ax.fill_between(sub.R_mid.values[bk], lo[bk], hi[bk],
                                color=ov["color"], alpha=0.15, linewidth=0)


def plot_tilt_diagnostic(pn, pk, outd):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, nm, pf, pp in [(axes[0], "Thin", pn, TN), (axes[1], "Thick", pk, TK)]:
        ax.errorbar(pf.zm, pf.crz, yerr=pf.crze, fmt="o", ms=3, lw=0.7,
                    capsize=1.5, color=pp["c"], alpha=0.8)
        ax.axhline(0, color="0.45", lw=1, ls=":")
        ax.axvline(0, color="0.45", lw=1, ls=":")
        ax.set_title(f"{nm}: tilt sign diagnostic")
        ax.set_xlabel(r"$Z$ [kpc]")
        ax.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
    fig.tight_layout()
    fig.savefig(outd / "fig_tilt_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_velocity_figures(pn, pk, outd):
    ranges = [(3, 6, "fig2_velocity_dispersion_R3_6.png"),
              (6, 9, "fig3_velocity_dispersion_R6_9.png"),
              (9, 12, "fig4_velocity_dispersion_R9_12.png")]
    cols = [("sR", "sRe", r"$\sigma_R$"),
            ("sp", "spe", r"$\sigma_\phi$"),
            ("sZ", "sZe", r"$\sigma_Z$")]
    for r0, r1, fname in ranges:
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex=False)
        for j, rl in enumerate(range(r0, r1)):
            for i, (ycol, ecol, ylabel) in enumerate(cols):
                ax = axes[i, j]
                for pf, pp in [(pn, TN), (pk, TK)]:
                    sub = pf[pf.rl == rl]
                    if sub.empty:
                        continue
                    ax.errorbar(sub.zm, sub[ycol], yerr=sub[ecol], fmt="o",
                                ms=2.5, lw=0.6, capsize=1.2, color=pp["c"],
                                alpha=0.75, label=pp["lbl"] if (i == 0 and j == 0) else None)
                ax.axvline(0, color="0.75", lw=0.8)
                ax.set_title(f"R={rl}-{rl+1} kpc", fontsize=9)
                ax.tick_params(direction="in", top=True, right=True, labelsize=8)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == 2:
                    ax.set_xlabel(r"$Z$ [kpc]")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            axes[0, 0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(outd / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_velocity_figures_combined(pa, outd, overlays=None):
    overlays = overlays or []
    ranges = [(3, 6, "fig2_velocity_dispersion_R3_6.png"),
              (6, 9, "fig3_velocity_dispersion_R6_9.png"),
              (9, 12, "fig4_velocity_dispersion_R9_12.png")]
    cols = [("sR", "sRe", r"$\sigma_R$"),
            ("sp", "spe", r"$\sigma_\phi$"),
            ("sZ", "sZe", r"$\sigma_Z$")]
    for r0, r1, fname in ranges:
        bins = sorted([x for x in pa.rl.unique() if r0 <= x < r1])
        if not bins:
            continue
        ncol = len(bins)
        fig, axes = plt.subplots(3, ncol, figsize=(max(12, 3.2 * ncol), 10), sharex=False, squeeze=False)
        for j, rl in enumerate(bins):
            rh = float(pa.loc[pa.rl == rl, "rh"].iloc[0])
            for i, (ycol, ecol, ylabel) in enumerate(cols):
                ax = axes[i, j]
                sub = pa[pa.rl == rl]
                if not sub.empty:
                    ax.errorbar(sub.zm, sub[ycol], yerr=sub[ecol], fmt="o",
                                ms=2.5, lw=0.6, capsize=1.2, color=TA["c"],
                                alpha=0.75, label=TA["lbl"] if (i == 0 and j == 0) else None)
                _overlay_profiles_on_axis(ax, overlays, rl, rh, ycol)
                ax.axvline(0, color="0.75", lw=0.8)
                ax.set_title(f"R={rl:g}-{rh:g} kpc", fontsize=9)
                ax.tick_params(direction="in", top=True, right=True, labelsize=8)
                if j == 0:
                    ax.set_ylabel(ylabel)
                if i == 2:
                    ax.set_xlabel(r"$Z$ [kpc]")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            by_label = dict(zip(labels, handles))
            axes[0, 0].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(outd / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_hsigma(pn, pk, outd):
    fig, axes = plt.subplots(2, 1, figsize=(9, 10), sharex=True)
    for ax, nm, pf, pp, zts in [
        (axes[0], "Thin", pn, TN, [-1, 1]),
        (axes[1], "Thick", pk, TK, [-2, -1, 1, 2]),
    ]:
        hs, he, amps = fit_hs(pf, zts=zts)
        cols = plt.cm.plasma(np.linspace(0.1, 0.9, len(zts)))
        for zt, col in zip(zts, cols):
            pts = crz_at_z(pf, zt)
            if not pts:
                continue
            Rp, yp, ep = zip(*pts)
            ax.errorbar(Rp, yp, yerr=ep, fmt="o", color=col, ms=5,
                        capsize=2.5, lw=0.9, elinewidth=0.9, label=f"Z={zt:+g} kpc")
            if np.isfinite(hs) and float(zt) in amps:
                rr = np.linspace(min(Rp) - 0.5, max(Rp) + 0.5, 200)
                ax.plot(rr, amps[float(zt)] * np.exp(-(rr - 8.0) / hs), lw=2, color=col)
        ax.axhline(0, color="0.4", lw=1, ls=":")
        ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
        ax.set_title(f"{nm} disc: $h_\\sigma={hs:.2f}\\pm{he:.2f}$ kpc" if np.isfinite(hs) else f"{nm} disc")
        ax.legend(frameon=False, ncol=2, fontsize=10)
        ax.tick_params(direction="in", top=True, right=True)
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig5_hsigma_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fit_diagnostics(pn, pk, fn, fk, outd):
    for nm, pf, fits, pp in [("thin", pn, fn, TN), ("thick", pk, fk, TK)]:
        bins = sorted(fits.rl.unique())
        if not bins:
            continue
        nc = 4
        nr = int(np.ceil(len(bins) / nc))
        fig, ax2d = plt.subplots(nr * 2, nc, figsize=(nc * 3.5, nr * 5), squeeze=False)
        zln = np.linspace(-pp["zmax"], pp["zmax"], 300)
        for i, rl in enumerate(bins):
            row, col = divmod(i, nc)
            ax_s = ax2d[row * 2, col]
            ax_r = ax2d[row * 2 + 1, col]
            sub = pf[(pf.rl == rl) & (np.abs(pf.zm) <= pp["zmax"])]
            fit = fits[fits.rl == rl]
            if sub.empty or fit.empty:
                continue
            f = fit.iloc[0]
            rh = float(f.rh) if "rh" in f.index else rl + 1
            model = str(_row_get(f, "sigz_model", "linear"))
            sz_fit = sigz_model_eval(np.abs(zln), model,
                                     dict(a=f.a, b=f.b,
                                          q=_row_get(f, "sigz_q", np.nan),
                                          s0sq=_row_get(f, "sigz_s0sq", np.nan),
                                          k=_row_get(f, "sigz_k", np.nan),
                                          L=_row_get(f, "sigz_L", np.nan),
                                          s0=_row_get(f, "sigz_s0", np.nan)))[0]
            ax_s.errorbar(sub.zm, sub.sZ, yerr=sub.sZe, fmt="o",
                          ms=3.5, capsize=1.5, color="k", lw=0.8)
            ax_s.plot(zln, sz_fit, lw=2, color=pp["c"],
                      label=f"{model}, rmse={_row_get(f, 'sigz_rmse', np.nan):.2f}")
            ax_s.axvline(0, color="0.7", lw=0.8)
            ax_s.set_title(f"R={rl:g}-{rh:g}", fontsize=9)
            ax_s.set_ylabel(r"$\sigma_Z$ [km/s]", fontsize=8)
            ax_s.legend(fontsize=7, frameon=False)
            ax_s.set_ylim(0)
            ax_r.errorbar(sub.zm, sub.crz, yerr=sub.crze, fmt="o",
                          ms=3.5, capsize=1.5, color="k", lw=0.8)
            ax_r.plot(zln, _row_get(f, "rz_b", 0.0) + f.m * zln, lw=2, color="darkorange",
                      label=f"c={_row_get(f, 'rz_b', 0.0):.1f}, m={f.m:.1f}")
            ax_r.axhline(0, color="0.7", lw=0.8)
            ax_r.axvline(0, color="0.7", lw=0.8)
            ax_r.set_ylabel(r"$\sigma^2_{RZ}=\langle v_Rv_Z\rangle$", fontsize=8)
            ax_r.set_xlabel(r"$Z$ [kpc]", fontsize=8)
            ax_r.legend(fontsize=7, frameon=False)
        for ax in ax2d.ravel():
            ax.tick_params(labelsize=7)
        for j in range(len(bins), nr * nc):
            ax2d[divmod(j, nc)[0] * 2, divmod(j, nc)[1]].axis("off")
            ax2d[divmod(j, nc)[0] * 2 + 1, divmod(j, nc)[1]].axis("off")
        fig.tight_layout()
        fig.savefig(outd / f"fig_fitdiag_{nm}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def plot_sigz_model_selection(pf, fits, outd, suffix, pp):
    bins = sorted(fits.rl.unique())
    if not bins:
        return
    nc = 4
    nr = int(np.ceil(len(bins) / nc))
    fig, ax2d = plt.subplots(nr, nc, figsize=(nc * 3.6, nr * 2.7), squeeze=False)
    zln = np.linspace(-pp["zmax"], pp["zmax"], 300)
    colors = {"linear": "#2ca25f", "quadratic": "#756bb1", "tanh": "#d95f0e"}
    for i, rl in enumerate(bins):
        row, col = divmod(i, nc)
        ax = ax2d[row, col]
        sub = pf[(pf.rl == rl) & (np.abs(pf.zm) <= pp["zmax"])]
        fit = fits[fits.rl == rl]
        if sub.empty or fit.empty:
            ax.axis("off")
            continue
        f = fit.iloc[0]
        rh = float(f.rh) if "rh" in f.index else rl + 1
        ax.errorbar(sub.zm, sub.sZ, yerr=sub.sZe, fmt="o", ms=2.8,
                    capsize=1.2, color="k", lw=0.7, alpha=0.85)
        models = [
            ("linear", dict(a=f.a, b=f.b)),
            ("quadratic", dict(q=_row_get(f, "sigz_q", np.nan), s0sq=_row_get(f, "sigz_s0sq", np.nan))),
            ("tanh", dict(k=_row_get(f, "sigz_k", np.nan), L=_row_get(f, "sigz_L", np.nan), s0=_row_get(f, "sigz_s0", np.nan))),
        ]
        selected = str(_row_get(f, "sigz_model", "linear"))
        for model, params in models:
            vals = np.array(list(params.values()), dtype=float)
            if not np.all(np.isfinite(vals)):
                continue
            y = sigz_model_eval(np.abs(zln), model, params)[0]
            ax.plot(zln, y, color=colors[model], lw=2.1 if model == selected else 1.2,
                    alpha=1.0 if model == selected else 0.45,
                    ls="-" if model == selected else "--", label=model if i == 0 else None)
        ax.axvline(0, color="0.75", lw=0.7)
        ax.set_title(f"R={rl:g}-{rh:g}; best={selected}", fontsize=8)
        ax.tick_params(labelsize=7, direction="in", top=True, right=True)
        if col == 0:
            ax.set_ylabel(r"$\sigma_Z$ [km/s]", fontsize=8)
        if row == nr - 1:
            ax.set_xlabel(r"$Z$ [kpc]", fontsize=8)
    for j in range(len(bins), nr * nc):
        ax2d.ravel()[j].axis("off")
    handles, labels = ax2d[0, 0].get_legend_handles_labels()
    if handles:
        ax2d[0, 0].legend(handles, labels, frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outd / f"fig_sigz_model_selection_{suffix}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_fig6(fn, fk, outd, rng, plot_shm=True):
    fig, ax = plt.subplots(figsize=(9, 6))
    for fits, pp in [(fn, TN), (fk, TK)]:
        row = get_solar_row(fits)
        if row is None:
            continue
        z = np.linspace(0.02, pp["zmax"], 300)
        med, lo, hi = mc_sig(row, pp, z, nmc=400, rng=rng)
        ax.plot(z, med, ls=pp["ls"], color=pp["c"], lw=2.5, label=pp["lbl"])
        ax.fill_between(z, lo, hi, color=pp["c"], alpha=pp["alpha"])
    if plot_shm:
        zr = _fig6_z_grid()
        ax.plot(zr, _fig6_sheet_limit_sigma(shm_sigma, R0, zr),
                color="firebrick", lw=2, label="SHM")
    ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=13)
    ax.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=13)
    ax.set_xlim(0, 4.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    secax = ax.secondary_yaxis("right", functions=(s2k, k2s))
    secax.set_ylabel(r"$|K_Z|$ [km$^2$ s$^{-2}$ kpc$^{-1}$]", fontsize=12)
    ax.legend(frameon=False, fontsize=12)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outd / "fig6_solar_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig7(fn, fk, outd, rng):
    all_rm = sorted(set(list(fn.rm.values) + list(fk.rm.values)))
    if not all_rm:
        return
    nc = 2
    nr = int(np.ceil(len(all_rm) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(12, nr * 3.5), sharex=True, squeeze=False)
    axes = axes.ravel()
    for pi, rm in enumerate(all_rm):
        ax = axes[pi]
        rl = rm - 0.5
        for fits, pp in [(fn, TN), (fk, TK)]:
            sub = fits[fits.rl == rl]
            if sub.empty:
                continue
            row = sub.iloc[0]
            z = np.linspace(0.02, min(pp["zmax"], 4.0), 250)
            med, lo, hi = mc_sig(row, pp, z, nmc=250, rng=rng)
            ax.plot(z, med, ls=pp["ls"], color=pp["c"], lw=2,
                    label=pp["lbl"] if pi == 0 else None)
            ax.fill_between(z, lo, hi, color=pp["c"], alpha=pp["alpha"])
        zr = np.linspace(0.01, 4, 200)
        ax.plot(zr, shm_sigma(rm, zr), color="firebrick", lw=1.5,
                alpha=0.85, label="SHM" if pi == 0 else None)
        ax.text(0.97, 0.91, f"R = {rm:.1f} kpc", transform=ax.transAxes,
                ha="right", va="top", fontsize=10)
        ax.set_xlim(0, 4)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        secax = ax.secondary_yaxis("right", functions=(s2k, k2s))
        if pi % nc == 0:
            ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        if pi >= len(all_rm) - nc:
            ax.set_xlabel(r"$|Z|$ [kpc]")
        if pi % nc == nc - 1:
            secax.set_ylabel(r"$|K_Z|$")
    for pi in range(len(all_rm), len(axes)):
        axes[pi].axis("off")
    axes[0].legend(frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(outd / "fig7_sigma_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig8(fn, fk, outd, rng):
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    for ax, zv in zip(axes, [0.3, 1.0, 3.0]):
        for fits, pp in [(fn, TN), (fk, TK)]:
            if zv > pp["zmax"] + 0.01:
                continue
            xs, ys, ylo, yhi = [], [], [], []
            for _, row in fits.sort_values("rm").iterrows():
                med, lo, hi = mc_sig(row, pp, np.array([zv]), nmc=200, rng=rng)
                xs.append(row.rm); ys.append(med[0]); ylo.append(lo[0]); yhi.append(hi[0])
            if not xs:
                continue
            xs = np.asarray(xs); ys = np.asarray(ys); ylo = np.asarray(ylo); yhi = np.asarray(yhi)
            o = np.argsort(xs)
            ax.plot(xs[o], ys[o], ls=pp["ls"], color=pp["c"], lw=2.2,
                    marker="o", ms=5, label=pp["lbl"])
            ax.fill_between(xs[o], ylo[o], yhi[o], color=pp["c"], alpha=pp["alpha"])
        rr = np.linspace(4.5, 11.5, 120)
        ax.plot(rr, [shm_sigma(r, [zv])[0] for r in rr], color="firebrick", lw=1.8, label="SHM")
        ax.text(0.03, 0.91, f"$|Z|={zv:.1f}$ kpc", transform=ax.transAxes, fontsize=11)
        ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=10)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(s2k, k2s))
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig8_sigma_vs_R.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig9(fn, fk, outd, rng):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
    for ax, nm, fits, pp in [(axes[0], "Thin disc", fn, TN), (axes[1], "Thick disc", fk, TK)]:
        row = get_solar_row(fits)
        if row is None:
            continue
        z = np.linspace(0.02, pp["zmax"], 300)
        for law, ls, lbl in [("exp", "-", "Exponential nu"), ("sech2", "--", r"sech$^2$ nu")]:
            med, lo, hi = mc_sig(row, pp, z, nmc=400, law=law, rng=rng)
            ax.plot(z, med, ls=ls, color=pp["c"], lw=2.2, label=lbl)
            ax.fill_between(z, lo, hi, color=pp["c"], alpha=pp["alpha"] * 0.8)
        zr = np.linspace(0.01, pp["zmax"], 300)
        ax.plot(zr, shm_sigma(8.5, zr), color="firebrick", lw=1.8, label="SHM")
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        ax.set_title(f"{nm}: exp vs sech2")
        ax.legend(frameon=False, fontsize=10)
        ax.set_xlim(0, pp["zmax"])
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(s2k, k2s))
    fig.tight_layout()
    fig.savefig(outd / "fig9_exp_vs_sech2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def kg_sig2(z, Sd, rdm, hZ):
    z = np.asarray(z, dtype=float)
    Sdk = Sd * 1e6
    rdmk = rdm * 1e9
    t1 = 2 * np.pi * G * hZ * Sdk * (1 - 0.5 * np.exp(-z / hZ))
    t2 = 4 * np.pi * G * hZ * rdmk * (z + hZ)
    return t1 + t2


def fit_kg(absz, sig2, esig2, hZ, max_rel_err=0.5, n_zbins=25):
    """Fit KG disc+DM model to sigma_Z^2 vs |Z|.
    Bins data into n_zbins uniform |Z| bins and uses the median per bin as the
    fit input — this prevents low-|Z| data density from dominating the fit and
    ensures the curve passes through the full |Z| range (matching Cheng+2024 style).
    Returns both the binned data (for plotting) and the raw data.
    """
    ok = np.isfinite(absz) & np.isfinite(sig2) & (sig2 > 0) & np.isfinite(esig2)
    z_all, y_all, e_all = absz[ok], sig2[ok], esig2[ok]
    if len(z_all) < 5:
        return None

    if len(z_all) <= 30:
        # Already pre-binned — fit directly without further binning
        z_bin, y_bin = z_all, y_all
        e_bin = e_all if (e_all > 0).any() else np.ones_like(y_all) * np.std(y_all) * 0.1
    else:
        # Bin into uniform |Z| bins; use median per bin as the representative point
        zmax_v = z_all.max()
        edges = np.linspace(0, zmax_v, n_zbins + 1)
        z_bin, y_bin, e_bin = [], [], []
        for i in range(n_zbins):
            mask = (z_all >= edges[i]) & (z_all < edges[i + 1])
            if mask.sum() < 3:
                continue
            z_bin.append(np.median(z_all[mask]))
            y_bin.append(np.median(y_all[mask]))
            e_bin.append(np.std(y_all[mask]) / np.sqrt(mask.sum()))
        if len(z_bin) < 5:
            return None
        z_bin = np.array(z_bin); y_bin = np.array(y_bin); e_bin = np.array(e_bin)

    def mod(zv, Sd, rdm):
        return kg_sig2(zv, Sd, rdm, hZ)

    best, best_res = None, np.inf
    for p0 in [[50, 0.015], [80, 0.001], [35, 0.016], [100, 0.005], [20, 0.030]]:
        try:
            popt, pcov = curve_fit(mod, z_bin, y_bin, p0=p0,
                                   bounds=([0.1, 1e-6], [2000.0, 5.0]),
                                   maxfev=100000, method='trf')
            res = np.sum((y_bin - mod(z_bin, *popt)) ** 2)
            if res < best_res:
                best_res = res
                best = (popt, pcov)
        except Exception:
            continue
    if best is None:
        return None
    popt, pcov = best
    pe = np.sqrt(np.diag(pcov))
    # Return binned data for clean plot + raw data for reference
    return dict(Sd=popt[0], rdm=popt[1], Sd_e=pe[0], rdm_e=pe[1],
                z=z_bin, y=y_bin, e=e_bin,
                z_raw=z_all, y_raw=y_all, e_raw=e_all)


def fit_kg_exact(absz, sig2, esig2, hZ, D=0.4, alpha=2.0, hR=4.5, R=8.5,
                 n_zbins=25, z_fine_n=400):
    """Fit KG89 disc+DM model to σ²_z using EXACT eq 39 (K_z) + eq 50 (tilt Jeans).

    Free parameters
    ---------------
    K  : disc surface density [M_sun/pc²]  — paper's Σ_disc
    F  : DM volume density    [M_sun/pc³]  — paper's ρ_DM

    The force K_z(z) comes from eq 39 (√(z²+D²) disc term + linear DM term),
    and σ²_z(z) is integrated with the full S(R,z) tilt factor via eq 50.
    This reproduces Cheng+2024 Fig 10 exactly, including the tilt term.
    """
    ok = (np.isfinite(absz) & np.isfinite(sig2) & (sig2 > 0) & np.isfinite(esig2))
    z_all, y_all, e_all = absz[ok], sig2[ok], esig2[ok]
    if len(z_all) < 5:
        return None

    # Bin the data (same logic as fit_kg)
    if len(z_all) <= 30:
        z_bin = z_all; y_bin = y_all
        e_bin = (e_all if (e_all > 0).any()
                 else np.ones_like(y_all) * np.std(y_all) * 0.1)
    else:
        zmax_v = z_all.max()
        edges = np.linspace(0, zmax_v, n_zbins + 1)
        z_bin, y_bin, e_bin = [], [], []
        for i in range(n_zbins):
            mask = (z_all >= edges[i]) & (z_all < edges[i + 1])
            if mask.sum() < 3:
                continue
            z_bin.append(np.median(z_all[mask]))
            y_bin.append(np.median(y_all[mask]))
            e_bin.append(np.std(y_all[mask]) / np.sqrt(mask.sum()))
        if len(z_bin) < 5:
            return None
        z_bin = np.array(z_bin); y_bin = np.array(y_bin); e_bin = np.array(e_bin)

    # Integral grid MUST extend well past data so I[z_max]=0 is a valid boundary.
    # Rule: z_max >= max(data_max * 1.3, 12 * hZ) so ν(z_max) < e^{-12} ≈ 6e-6.
    z_lo   = max(float(z_bin.min()) * 0.5, 0.01)
    z_hi   = max(float(z_bin.max()) * 1.30, float(hZ) * 12.0)
    z_fine = np.linspace(z_lo, z_hi, z_fine_n)
    nu_fine = np.exp(-z_fine / float(hZ))

    def _model(zv, K, F):
        Kz_f = kg.kg_Kz_eq39(z_fine, max(float(K), 0.1), D, max(float(F), 1e-7))
        s2_f = kg.sigma_zz2_with_tilt_eq50(z_fine, R, nu_fine, Kz_f,
                                             alpha=alpha, hR=hR)
        return np.interp(np.asarray(zv, float), z_fine, np.clip(s2_f, 0, None))

    best, best_res = None, np.inf
    bounds_lo = np.array([0.1, 1e-7])
    bounds_hi = np.array([600.0, 0.15])
    for p0 in [[50, 0.010], [35, 0.015], [80, 0.008],
               [120, 0.005], [20, 0.020], [150, 0.003]]:
        try:
            sig_w = np.where(e_bin > 0, e_bin, y_bin * 0.1)
            popt, pcov = curve_fit(_model, z_bin, y_bin, p0=p0,
                                   bounds=(bounds_lo, bounds_hi),
                                   maxfev=300000, method="trf", sigma=sig_w)
            res = float(np.sum((y_bin - _model(z_bin, *popt)) ** 2))
            if res < best_res:
                best_res = res; best = (popt, pcov)
        except Exception:
            continue

    if best is None:
        return None
    popt, pcov = best
    pe = np.sqrt(np.diag(np.clip(pcov, 0, None)))
    return dict(K=float(popt[0]),   F=float(popt[1]),
                K_e=float(pe[0]),   F_e=float(pe[1]),
                Sd=float(popt[0]),  rdm=float(popt[1]),   # legacy keys
                Sd_e=float(pe[0]),  rdm_e=float(pe[1]),
                z=z_bin, y=y_bin, e=e_bin,
                z_raw=z_all, y_raw=y_all, e_raw=e_all,
                hZ=float(hZ), D=float(D), alpha=float(alpha), hR=float(hR),
                K_at_lower=bool(popt[0] <= bounds_lo[0] * 1.001),
                K_at_upper=bool(popt[0] >= bounds_hi[0] * 0.999),
                F_at_lower=bool(popt[1] <= bounds_lo[1] * 1.001),
                F_at_upper=bool(popt[1] >= bounds_hi[1] * 0.999))


def solar_sig2(pf, R_target=8.5):
    rl = int(R_target - 0.5)
    sub = pf[pf.rl == rl].copy()
    absz = np.abs(sub.zm.values)
    sig2 = sub.sZ.values ** 2
    esig2 = 2 * sub.sZ.values * sub.sZe.values
    o = np.argsort(absz)
    return absz[o], sig2[o], esig2[o]


def plot_fig10(pn, pk, outd):
    az_n, s2_n, es2_n = solar_sig2(pn)
    az_k, s2_k, es2_k = solar_sig2(pk)
    res_n = fit_kg(az_n, s2_n, es2_n, TN["hZ"])
    res_k = fit_kg(az_k, s2_k, es2_k, TK["hZ"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, nm, res, pp, az, s2, es2 in [
        (axes[0], "Thin disc", res_n, TN, az_n, s2_n, es2_n),
        (axes[1], "Thick disc", res_k, TK, az_k, s2_k, es2_k),
    ]:
        ax.errorbar(az, s2, yerr=es2, fmt="o", color=pp["c"], ms=4,
                    capsize=2, lw=0.9, elinewidth=0.9, label="Data")
        if res:
            zz = np.linspace(0, az.max() * 1.1, 300)
            lbl = (f'$\\Sigma_{{disk}}={res["Sd"]:.2f}\\pm{res["Sd_e"]:.2f}$\n'
                   f'$\\rho_{{dm}}={res["rdm"]:.4f}\\pm{res["rdm_e"]:.4f}$')
            ax.plot(zz, kg_sig2(zz, res["Sd"], res["rdm"], pp["hZ"]), "k-", lw=2, label=lbl)
        ax.set_xlabel(r"$|Z|$ [kpc]")
        ax.set_ylabel(r"$\sigma^2_Z$ [km$^2$ s$^{-2}$]")
        ax.set_title(f"R = 8.5 kpc {nm}")
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(outd / "fig10_KG_integral.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res_n, res_k


def write_consistency_summary(fn, fk, res_n, res_k, outd):
    lines = []
    lines.append("MATHEMATICAL CONSISTENCY SUMMARY")
    lines.append("")
    lines.append("1. Tilt slope sign:")
    lines.append("   With the adopted cylindrical convention, sigma_RZ^2 is expected to be odd in Z;")
    lines.append("   the fitted through-origin slope should be interpreted together with the velocity convention.")
    for nm, fits in [("Thin", fn), ("Thick", fk)]:
        pos = int((fits.m > 0).sum()) if not fits.empty else 0
        neg = int((fits.m < 0).sum()) if not fits.empty else 0
        lines.append(f"   {nm}: {pos}/{len(fits)} positive slopes, {neg}/{len(fits)} negative slopes")
    lines.append("")
    lines.append("2. Sigma_thick > Sigma_thin at solar circle:")
    rn_s = get_solar_row(fn)
    rk_s = get_solar_row(fk)
    if rn_s is not None and rk_s is not None:
        for zv in [0.3, 1.0, 2.0, 3.0]:
            Sn = None if zv > TN["zmax"] + 0.01 else sig_from_row(rn_s, TN, np.array([zv]))[0][0]
            Sk = None if zv > TK["zmax"] + 0.01 else sig_from_row(rk_s, TK, np.array([zv]))[0][0]
            lines.append(f"   |Z|={zv}: Sigma_thin={Sn if Sn is not None else 'N/A'}, Sigma_thick={Sk if Sk is not None else 'N/A'}")
    lines.append("")
    lines.append("3. K&G:")
    if res_n and res_k:
        lines.append(f"   Sigma_disk_thin={res_n['Sd']:.2f} Msun/pc^2, rho_dm_thin={res_n['rdm']:.4f}")
        lines.append(f"   Sigma_disk_thick={res_k['Sd']:.2f} Msun/pc^2, rho_dm_thick={res_k['rdm']:.4f}")
    (outd / "mathematical_consistency_summary.txt").write_text("\n".join(lines) + "\n")


def plot_tilt_diagnostic_combined(pa, outd):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(pa.zm, pa.crz, yerr=pa.crze, fmt="o", ms=3, lw=0.7,
                capsize=1.5, color=TA["c"], alpha=0.8)
    ax.axhline(0, color="0.45", lw=1, ls=":")
    ax.axvline(0, color="0.45", lw=1, ls=":")
    ax.set_title("All stars: tilt sign diagnostic")
    ax.set_xlabel(r"$Z$ [kpc]")
    ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fig.savefig(outd / "fig_tilt_diagnostic.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_hsigma_combined(pa, outd, overlay_dir=None, da=None,
                          chem_crz_thin=None, chem_crz_thick=None,
                          chem_selection="mgfe"):
    """Reproduce paper Fig 5: sigma_RZ^2 vs R at fixed |Z|, with exponential fit.

    Panel layout (when overlay_dir or chem_crz_* is given):
      Top:    All tracers (raw all-stars), h_sigma unconstrained
      Middle: Thin disc — selected chemical tracer, Z=±1 only
      Bottom: Thick disc — selected chemical tracer, Z=±1 and ±2

    When chem_crz_thin/thick DataFrames are provided they take priority over
    loading from CSV files in overlay_dir.
    """

    _OV_CMAP = [plt.cm.coolwarm(v) for v in np.linspace(0.05, 0.95, 4)]

    # ------------------------------------------------------------------
    # Helper: load CSV → (R, y, e, zv) arrays
    # ------------------------------------------------------------------
    def _load_pts(path):
        if not Path(path).exists():
            return np.array([]), np.array([]), np.array([]), np.array([])
        df = pd.read_csv(path)
        return df.R_mid.values, df.sigma_RZ2.values, df.sigma_RZ2_err.values, df.z_target.values

    # ------------------------------------------------------------------
    # Helper: fit h_sigma and draw, return (hb_c, he_c, amps)
    # ------------------------------------------------------------------
    def _fit_draw(ax, R_fit, y_fit, e_fit, zv_fit, zts, cmcols, ls="-", lw=2.2):
        hb, he, amps = _fit_hs_xy(R_fit, y_fit, e_fit, zv_fit, Rref=8.0)
        hb_c = hb if (np.isfinite(hb) and hb < 20.0) else np.nan
        he_c = he if np.isfinite(hb_c) else np.nan
        for zt, col in zip(zts, cmcols):
            if not (np.isfinite(hb_c) and float(zt) in amps):
                continue
            m = np.isclose(zv_fit, zt) & np.isfinite(R_fit)
            if m.sum() < 2:
                continue
            rr = np.linspace(R_fit[m].min() - 0.3, R_fit[m].max() + 0.3, 200)
            ax.plot(rr, amps[float(zt)] * np.exp(-(rr - 8.0) / hb_c),
                    lw=lw, ls=ls, color=col, zorder=3)
        return hb_c, he_c, amps

    # ------------------------------------------------------------------
    # Panel: combined thin/thick from raw or pooled CSV sources, binned by R
    # ------------------------------------------------------------------
    def _binned_panel(ax, paths_with_labels, nm, zts, cmcols,
                      precomputed_df=None):
        """
        Pool sigma_RZ^2 vs R data, bin by R per Z slice, fit exponential.
        If precomputed_df is supplied (from compute_chem_crz_from_raw) the CSV
        paths are ignored and the DataFrame is used directly.
        """
        if precomputed_df is not None and not precomputed_df.empty:
            # Data already R-binned (one point per cprof group per R-Z cell)
            R_c  = precomputed_df.R_mid.values
            y_c  = precomputed_df.sigma_RZ2.values
            e_c  = precomputed_df.sigma_RZ2_err.values
            zv_c = precomputed_df.z_target.values
            src_label = "raw"
        else:
            R_all, y_all, e_all, zv_all = [], [], [], []
            for (path, _) in paths_with_labels:
                Rs, ys, es, zvs = _load_pts(path)
                if len(Rs):
                    R_all.append(Rs); y_all.append(ys)
                    e_all.append(es); zv_all.append(zvs)
            if not R_all:
                ax.set_title(f"{nm}: no data", fontsize=10)
                return np.nan, np.nan, {}
            R_c  = np.concatenate(R_all);  y_c  = np.concatenate(y_all)
            e_c  = np.concatenate(e_all);  zv_c = np.concatenate(zv_all)
            src_label = "pooled CSV"

        R_edges = np.arange(4.0, 12.5, 1.0)
        R_fit_list, y_fit_list, e_fit_list, zv_fit_list = [], [], [], []

        for zt, col in zip(zts, cmcols):
            z_ok = np.isclose(zv_c, zt) & np.isfinite(R_c) & np.isfinite(y_c) & (e_c > 0)
            Rz, yz, ez = R_c[z_ok], y_c[z_ok], e_c[z_ok]

            if precomputed_df is not None and not precomputed_df.empty:
                # Already one point per R bin — no further binning needed
                ok = np.isfinite(yz)
                if ok.sum() < 2:
                    continue
                Rb, yb, eb = Rz[ok], yz[ok], ez[ok]
            else:
                Rb, yb, eb = [], [], []
                for rl in R_edges:
                    m = (Rz >= rl) & (Rz < rl + 1.0)
                    if m.sum() < 1:
                        continue
                    w = 1.0 / ez[m] ** 2
                    Rb.append(rl + 0.5)
                    yb.append(np.sum(w * yz[m]) / np.sum(w))
                    eb.append(1.0 / np.sqrt(np.sum(w)))
                if len(Rb) < 2:
                    continue
                Rb, yb, eb = np.array(Rb), np.array(yb), np.array(eb)

            ax.errorbar(Rb, yb, yerr=eb, fmt="o", color=col, ms=7,
                        capsize=3.5, lw=1.2, elinewidth=1.2,
                        alpha=0.92, label=f"Z={zt:+g} kpc", zorder=4)
            R_fit_list.append(Rb); y_fit_list.append(yb)
            e_fit_list.append(eb); zv_fit_list.append(np.full(len(Rb), float(zt)))

        if not R_fit_list:
            ax.set_title(f"{nm}: no bins", fontsize=10)
            return np.nan, np.nan, {}

        R_fit = np.concatenate(R_fit_list); y_fit = np.concatenate(y_fit_list)
        e_fit = np.concatenate(e_fit_list); zv_fit = np.concatenate(zv_fit_list)

        hb_c, he_c, amps = _fit_draw(ax, R_fit, y_fit, e_fit, zv_fit, zts, cmcols)

        ax.axhline(0, color="0.4", lw=1, ls=":")
        ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
        src_tag = ("raw stars" if src_label == "raw" else
                   ("MgFe CSV" if chem_selection == "mgfe" else "MgFe + $\\alpha$Fe pooled"))
        title = (f"{nm} ({src_tag}): $h_\\sigma={hb_c:.2f}\\pm{he_c:.2f}$ kpc"
                 if np.isfinite(hb_c) else f"{nm} ({src_tag}): $h_\\sigma$ unconstrained")
        ax.set_title(title, fontsize=10)
        ax.legend(frameon=False, ncol=2, fontsize=8)
        ax.tick_params(direction="in", top=True, right=True)
        return hb_c, he_c, amps

    # ------------------------------------------------------------------
    # Panel: vphi-selected thin/thick — unbiased because vphi ⊥ (vR, vZ)
    #   fast (vphi ≥ 218 km/s) ≈ 97% pure thin disc
    #   slow (130 ≤ vphi < 200 km/s) = thick disc-enhanced
    # ------------------------------------------------------------------
    def _vphi_panel(ax, da_dict):
        vp = da_dict.get("vp", da_dict.get("vphi"))
        R_edges = np.arange(4.0, 12.5, 1.0)

        VPH_FAST = 218.0   # thin disc-like cut  (km/s)
        VPH_SLOW_MAX = 200.0
        VPH_SLOW_MIN = 130.0

        n_fast = int((np.asarray(vp) >= VPH_FAST).sum())
        n_slow = int(((np.asarray(vp) >= VPH_SLOW_MIN) &
                      (np.asarray(vp) <  VPH_SLOW_MAX)).sum())
        print(f"  vphi-thin (≥{VPH_FAST}): {n_fast:,}  "
              f"vphi-thick ({VPH_SLOW_MIN}–{VPH_SLOW_MAX}): {n_slow:,}")

        # Thin-disc-like: Z=±1 kpc only (thin disc barely reaches Z=2 kpc)
        df_fast, df_slow_z1 = compute_vphi_crz_profiles(
            da_dict["R"], da_dict["Z"], da_dict["vR"], da_dict["vZ"], vp,
            z_targets=[-1, 1], R_edges=R_edges,
            vp_fast_min=VPH_FAST, vp_slow_max=VPH_SLOW_MAX, vp_slow_min=VPH_SLOW_MIN)

        # Thick-disc-like: Z=±1 and ±2 kpc
        _, df_slow_z2 = compute_vphi_crz_profiles(
            da_dict["R"], da_dict["Z"], da_dict["vR"], da_dict["vZ"], vp,
            z_targets=[-2, -1, 1, 2], R_edges=R_edges,
            vp_fast_min=VPH_FAST, vp_slow_max=VPH_SLOW_MAX, vp_slow_min=VPH_SLOW_MIN)

        df_fast.to_csv(outd / "vphi_fast_crz_profile.csv",  index=False)
        df_slow_z2.to_csv(outd / "vphi_slow_crz_profile.csv", index=False)

        fast_zts  = [-1, 1]
        slow_zts  = [-2, -1, 1, 2]
        fast_cols = [plt.cm.Oranges(0.55), plt.cm.Oranges(0.85)]
        slow_cols = [plt.cm.Blues(0.40), plt.cm.Blues(0.62),
                     plt.cm.Blues(0.78), plt.cm.Blues(0.95)]

        def _pool_plot(df, zts, cols, mkr, lbl_prefix):
            Rp, yp, ep, zvp = [], [], [], []
            for zt, col in zip(zts, cols):
                sub = df[np.isclose(df.z_target, zt)]
                if len(sub) < 2:
                    continue
                ax.errorbar(sub.R_mid, sub.sigma_RZ2, yerr=sub.sigma_RZ2_err,
                            fmt=mkr, color=col, ms=6, capsize=3, lw=1, elinewidth=1,
                            alpha=0.88, label=f"{lbl_prefix} Z={zt:+g}")
                Rp.extend(sub.R_mid.tolist()); yp.extend(sub.sigma_RZ2.tolist())
                ep.extend(sub.sigma_RZ2_err.tolist())
                zvp.extend([float(zt)] * len(sub))
            return np.array(Rp), np.array(yp), np.array(ep), np.array(zvp)

        Rf, yf, ef, zvf = _pool_plot(df_fast,   fast_zts, fast_cols, "o", r"$v_\phi\!\geq\!218$")
        Rs, ys, es, zvs = _pool_plot(df_slow_z2, slow_zts, slow_cols, "^", r"$v_\phi\!<\!200$")

        hb_f, he_f = np.nan, np.nan
        hb_s, he_s = np.nan, np.nan
        if len(Rf) >= 4:
            hb_f, he_f, _ = _fit_draw(ax, Rf, yf, ef, zvf,
                                        fast_zts, fast_cols, ls="-", lw=2.2)
        if len(Rs) >= 4:
            hb_s, he_s, _ = _fit_draw(ax, Rs, ys, es, zvs,
                                        slow_zts, slow_cols, ls="--", lw=2.2)

        ax.axhline(0, color="0.4", lw=1, ls=":")
        ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")

        def _hs(h, he):
            return f"$h_\\sigma={h:.2f}\\pm{he:.2f}$ kpc" if np.isfinite(h) else "unconstrained"

        ax.set_title(
            fr"All stars — $v_\phi$-selected: "
            fr"thin-like (solid) {_hs(hb_f, he_f)}   "
            fr"thick-like (dashed) {_hs(hb_s, he_s)}",
            fontsize=9)
        handles, labels = ax.get_legend_handles_labels()
        seen, uh, ul = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); uh.append(h); ul.append(l)
        ax.legend(uh, ul, frameon=False, ncol=2, fontsize=7)
        ax.tick_params(direction="in", top=True, right=True)
        return hb_f if np.isfinite(hb_f) else hb_s

    # ------------------------------------------------------------------
    # Fallback: raw all-stars panel (when da is None)
    # ------------------------------------------------------------------
    hs_all, he_all, amps_all = fit_hs(pa, zts=[-2, -1, 1, 2])
    hs_all_flat = np.isfinite(hs_all) and hs_all >= 20.0

    def _all_stars_panel(ax):
        cols_all = plt.cm.plasma(np.linspace(0.1, 0.9, 4))
        for zt, col in zip([-2, -1, 1, 2], cols_all):
            pts = crz_at_z(pa, zt)
            if not pts:
                continue
            Rp, yp, ep = zip(*pts)
            ax.errorbar(Rp, yp, yerr=ep, fmt="o", color=col, ms=5,
                        capsize=2.5, lw=0.9, elinewidth=0.9, label=f"Z={zt:+g} kpc")
            if np.isfinite(hs_all) and float(zt) in amps_all:
                rr = np.linspace(min(Rp) - 0.5, max(Rp) + 0.5, 200)
                ax.plot(rr, amps_all[float(zt)] * np.exp(-(rr - 8.0) / hs_all),
                        lw=2, ls="--" if hs_all_flat else "-", color=col)
        ax.axhline(0, color="0.4", lw=1, ls=":")
        ax.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]")
        if np.isfinite(hs_all):
            flat_note = " (flat/unconstrained)" if hs_all_flat else ""
            title = f"All tracers: $h_\\sigma={hs_all:.2f}\\pm{he_all:.2f}$ kpc{flat_note}"
        else:
            title = "All tracers: $h_\\sigma$ fit failed"
        ax.set_title(title, fontsize=11)
        ax.legend(frameon=False, ncol=2, fontsize=9)
        ax.tick_params(direction="in", top=True, right=True)

    # ------------------------------------------------------------------
    # Assemble figure
    # ------------------------------------------------------------------
    has_data = (chem_crz_thin is not None) or (overlay_dir is not None)
    if has_data:
        # CSV fallback paths (used only when chem_crz_thin/thick not provided)
        thin_paths = thick_paths = []
        if overlay_dir:
            p = Path(overlay_dir)
            thin_paths = [
                (str(p / "mgfe_feh_thin_chem" / "all_no_mgfe_mgfe_feh_thin_chem_sigma_RZ_vs_R_points_for_hsigma.csv"), "MgFe"),
            ]
            thick_paths = [
                (str(p / "mgfe_feh_thick_chem" / "all_no_mgfe_mgfe_feh_thick_chem_sigma_RZ_vs_R_points_for_hsigma.csv"), "MgFe"),
            ]
            if chem_selection == "combined":
                thin_paths.append(
                    (str(p / "alphafe_feh_thin_chem" / "all_no_mgfe_alphafe_feh_thin_chem_sigma_RZ_vs_R_points_for_hsigma.csv"), r"$\alpha$Fe"))
                thick_paths.append(
                    (str(p / "alphafe_feh_thick_chem" / "all_no_mgfe_alphafe_feh_thick_chem_sigma_RZ_vs_R_points_for_hsigma.csv"), r"$\alpha$Fe"))

        fig, axes = plt.subplots(3, 1, figsize=(9, 15), sharex=True)
        _all_stars_panel(axes[0])

        # Middle: thin — use raw computation if available, else CSV
        hs_n, he_n, _ = _binned_panel(
            axes[1], thin_paths, "Thin disc", [-1, 1], _OV_CMAP[1:3],
            precomputed_df=chem_crz_thin)

        # Bottom: thick — use raw computation if available, else CSV
        hs_k, he_k, _ = _binned_panel(
            axes[2], thick_paths, "Thick disc", [-2, -1, 1, 2], _OV_CMAP,
            precomputed_df=chem_crz_thick)

        axes[-1].set_xlabel(r"$R$ [kpc]")
        fig.tight_layout()
        fig.savefig(outd / "fig5_hsigma_fit.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        return hs_all, he_all

    # No overlay_dir: single all-stars panel
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _all_stars_panel(ax)
    ax.set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig5_hsigma_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return hs_all, he_all


def plot_fit_diagnostics_combined(pa, fa, outd, overlays=None):
    overlays = overlays or []
    bins = sorted(fa.rl.unique())
    if not bins:
        return
    nc = 4
    nr = int(np.ceil(len(bins) / nc))
    fig, ax2d = plt.subplots(nr * 2, nc, figsize=(nc * 3.5, nr * 5), squeeze=False)
    zln = np.linspace(-TA["zmax"], TA["zmax"], 300)
    for i, rl in enumerate(bins):
        row, col = divmod(i, nc)
        ax_s = ax2d[row * 2, col]
        ax_r = ax2d[row * 2 + 1, col]
        sub = pa[(pa.rl == rl) & (np.abs(pa.zm) <= TA["zmax"])]
        fit = fa[fa.rl == rl]
        if sub.empty or fit.empty:
            continue
        f = fit.iloc[0]
        rh = float(f.rh) if "rh" in f.index else rl + 1
        model = str(_row_get(f, "sigz_model", "linear"))
        sz_fit = sigz_model_eval(np.abs(zln), model,
                                 dict(a=f.a, b=f.b,
                                      q=_row_get(f, "sigz_q", np.nan),
                                      s0sq=_row_get(f, "sigz_s0sq", np.nan),
                                      k=_row_get(f, "sigz_k", np.nan),
                                      L=_row_get(f, "sigz_L", np.nan),
                                      s0=_row_get(f, "sigz_s0", np.nan)))[0]
        # Combined all-stars data
        ax_s.errorbar(sub.zm, sub.sZ, yerr=sub.sZe, fmt="o",
                      ms=2.5, capsize=1.2, color="0.35", lw=0.6, alpha=0.6, zorder=1)
        ax_s.plot(zln, sz_fit, lw=2, color=TA["c"], zorder=3,
                  label=f"{model}, rmse={_row_get(f, 'sigz_rmse', np.nan):.2f}")
        ax_r.errorbar(sub.zm, sub.crz, yerr=sub.crze, fmt="o",
                      ms=2.5, capsize=1.2, color="0.35", lw=0.6, alpha=0.6, zorder=1)
        ax_r.plot(zln, _row_get(f, "rz_b", 0.0) + f.m * zln, lw=2, color="darkorange", zorder=3,
                  label=f"c={_row_get(f, 'rz_b', 0.0):.1f}, m={f.m:.1f}")
        # Chemical thin/thick overlay data points
        ctr = 0.5 * (rl + rh)
        tol = max(0.51, 0.5 * (rh - rl) + 1e-6)
        for ov in overlays:
            pf = ov.get("profile")
            if pf is None:
                continue
            osub = pf[np.abs(pf.rm - ctr) <= tol]
            if osub.empty:
                continue
            osub_z = osub[np.abs(osub.zm) <= TA["zmax"]]
            ax_s.errorbar(osub_z.zm, osub_z.sZ, yerr=osub_z.sZe, fmt=ov["marker"],
                          ms=3, capsize=0, lw=0.5, color=ov["color"], alpha=0.7,
                          label=ov["label"] if i == 0 else None, zorder=2)
            ax_r.errorbar(osub_z.zm, osub_z.crz, yerr=osub_z.crze, fmt=ov["marker"],
                          ms=3, capsize=0, lw=0.5, color=ov["color"], alpha=0.7, zorder=2)
        ax_s.axvline(0, color="0.7", lw=0.8)
        ax_s.set_title(f"R={rl:g}-{rh:g}", fontsize=9)
        ax_s.set_ylabel(r"$\sigma_Z$ [km/s]", fontsize=8)
        ax_s.legend(fontsize=6, frameon=False)
        ax_s.set_ylim(0)
        ax_r.axhline(0, color="0.7", lw=0.8)
        ax_r.axvline(0, color="0.7", lw=0.8)
        ax_r.set_ylabel(r"$\sigma^2_{RZ}=\langle v_Rv_Z\rangle$", fontsize=8)
        ax_r.set_xlabel(r"$Z$ [kpc]", fontsize=8)
        ax_r.legend(fontsize=6, frameon=False)
    for ax in ax2d.ravel():
        ax.tick_params(labelsize=7)
    for j in range(len(bins), nr * nc):
        ax2d[divmod(j, nc)[0] * 2, divmod(j, nc)[1]].axis("off")
        ax2d[divmod(j, nc)[0] * 2 + 1, divmod(j, nc)[1]].axis("off")
    fig.tight_layout()
    fig.savefig(outd / "fig_fitdiag_all.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_fig6_combined(fa, outd, rng, overlays=None,
                        fa_lin=None, thick_lin_row=None, thin_lin_row=None,
                        mgfe_thick_lin_row=None):
    """
    Solar-radius surface density vs |Z|.
    fa_lin              : linear-σ_Z all-stars fits (for comparison)
    thick_lin_row       : pd.Series — solar-R row from combined-thick linear fits
    thin_lin_row        : pd.Series — solar-R row from combined-thin linear fits
    mgfe_thick_lin_row  : pd.Series — solar-R row from MgFe-only thick (Cheng exact, darkorange)
    """
    overlays = overlays or []
    fig, ax = plt.subplots(figsize=(9, 6))

    # ── Main all-stars result ─────────────────────────────────────────────
    row = get_solar_row(fa)
    if row is not None:
        z = np.linspace(0.02, TA["zmax"], 300)
        med, lo, hi = mc_sig(row, TA, z, nmc=400, rng=rng)
        model_label = str(_row_get(row, "sigz_model", "linear"))
        ax.plot(z, med, ls=TA["ls"], color=TA["c"], lw=2.5,
                label=TA["lbl"] + f" ({model_label} " + r"$\sigma_Z$)")
        ax.fill_between(z, lo, hi, color=TA["c"], alpha=TA["alpha"])

    # ── Linear all-stars (dashed, for comparison) ─────────────────────────
    if fa_lin is not None:
        row_lin = get_solar_row(fa_lin)
        if row_lin is not None:
            z = np.linspace(0.02, TA["zmax"], 300)
            med_l, lo_l, hi_l = mc_sig(row_lin, TA, z, nmc=400, rng=rng)
            ax.plot(z, med_l, ls=":", color=TA["c"], lw=2.0, alpha=0.7,
                    label=TA["lbl"] + " (linear σ_Z)")

    # ── SHM ────────────────────────────────────────────────────────────────
    zr = _fig6_z_grid()
    ax.plot(zr, _fig6_sheet_limit_sigma(shm_sigma, R0, zr),
            color="firebrick", lw=2, label="SHM")
    ax.plot(zr, _fig6_sheet_limit_sigma(baryonic_sigma, R0, zr),
            color="black", lw=1.8,
            label="Baryonic (BH\&G 2016)")

    # ── Chemical overlays from raw stars ──────────────────────────────────
    _overlay_surface_z(ax, overlays, 8.5, zmax=4.0, label_once=True)

    # ── Cheng+2024 comparison: thick disc with LINEAR σ_Z vs SHM ─────────
    if thick_lin_row is not None:
        z_tk = np.linspace(0.02, TK["zmax"], 300)
        med_tk, lo_tk, hi_tk = mc_sig(thick_lin_row, TK, z_tk, nmc=400, rng=rng)
        ax.plot(z_tk, med_tk, ls="--", color="#b2182b", lw=2.2,
                label="Combined thick (linear σ_Z, Cheng arg.)")
        ax.fill_between(z_tk, lo_tk, hi_tk, color="#b2182b", alpha=0.12)

    if thin_lin_row is not None:
        z_tn = np.linspace(0.02, TN["zmax"], 300)
        med_tn, lo_tn, hi_tn = mc_sig(thin_lin_row, TN, z_tn, nmc=400, rng=rng)
        ax.plot(z_tn, med_tn, ls="--", color="#2166ac", lw=2.2,
                label="Combined thin (linear σ_Z, Cheng arg.)")
        ax.fill_between(z_tn, lo_tn, hi_tn, color="#2166ac", alpha=0.12)

    # ── MgFe-only thick disc (Cheng exact: hZ=0.9, hR=2.0, hσ=5.03) — darkorange ──
    if mgfe_thick_lin_row is not None:
        z_mg = np.linspace(0.02, TK["zmax"], 300)
        med_mg, lo_mg, hi_mg = mc_sig(mgfe_thick_lin_row, TK, z_mg, nmc=400, rng=rng)
        ax.plot(z_mg, med_mg, ls="--", color="darkorange", lw=2.5,
                label="MgFe thick (linear σ_Z, Cheng+2024)", zorder=4)
        ax.fill_between(z_mg, lo_mg, hi_mg, color="darkorange", alpha=0.15)

    ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=13)
    ax.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=13)
    ax.set_xlim(0, 4.5)
    # Clamp y-axis to ~200 so SHM's rapid thin-disc rise (0→107 M☉/pc² over 4 kpc)
    # is visible on a scale comparable to the paper (0–120); data beyond 200 is clipped.
    _ydata_max = max(
        ax.get_ylim()[1],
        float(shm_sigma(R0, np.array([4.5]))[0]) * 1.8,
    )
    ax.set_ylim(0, min(_ydata_max, 200))
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(s2k, k2s))
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.grid(alpha=0.2)
    # annotation explaining the comparison
    ax.text(0.02, 0.80,
            "Dashed: linear σ_Z assumption (Cheng+2024)\n"
            "Thick disc + linear → expected to match SHM",
            transform=ax.transAxes, va="top", fontsize=8.5, color="0.35")
    fig.tight_layout()
    fig.savefig(outd / "fig6_solar_sigma.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig7_combined(fa, outd, rng, overlays=None, fa_lin=None,
                        fa_thin_lin=None, fa_thick_lin=None,
                        fa_mgfe_thick_lin=None):
    """
    Grid of Sigma vs |Z| for all R bins.
    fa_lin            : all-stars forced-linear sigma_Z (dark green dashed)
    fa_thin_lin       : combined thin disc forced-linear sigma_Z (blue dashed)
    fa_thick_lin      : combined thick disc forced-linear sigma_Z (red dashed)
    fa_mgfe_thick_lin : MgFe-only thick disc, Cheng exact params (darkorange dashed)
    """
    overlays = overlays or []
    all_rm = sorted(fa.rm.unique())
    if not all_rm:
        return
    nc = 2
    nr = int(np.ceil(len(all_rm) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(12, nr * 3.5), sharex=True, squeeze=False)
    axes = axes.ravel()
    for pi, rm in enumerate(all_rm):
        ax = axes[pi]
        z = np.linspace(0.01, TA["zmax"], 250)

        # Main all-stars curve (solid green)
        row = fa[fa.rm == rm].iloc[0]
        med, lo, hi = mc_sig(row, TA, z, nmc=250, rng=rng)
        model_label = str(_row_get(row, "sigz_model", "linear"))
        ax.plot(z, med, ls=TA["ls"], color=TA["c"], lw=2,
                label=f"All stars ({model_label} " + r"$\sigma_Z$)" if pi == 0 else None)
        ax.fill_between(z, lo, hi, color=TA["c"], alpha=TA["alpha"])

        # All-stars linear σ_Z (dark green dashed)
        if fa_lin is not None and not fa_lin.empty:
            rows_lin = fa_lin[np.isclose(fa_lin.rm, rm)]
            if not rows_lin.empty:
                med_l, lo_l, hi_l = mc_sig(rows_lin.iloc[0], TA, z, nmc=150, rng=rng)
                ax.plot(z, med_l, ls="--", color="#2d7507", lw=1.8, alpha=0.85,
                        label="All stars (linear σ_Z)" if pi == 0 else None)
                ax.fill_between(z, lo_l, hi_l, color="#2d7507", alpha=0.08)

        # Combined thin disc linear σ_Z (blue dashed)
        if fa_thin_lin is not None and not fa_thin_lin.empty:
            rows_tn = fa_thin_lin[np.isclose(fa_thin_lin.rm, rm)]
            if not rows_tn.empty:
                z_tn = np.linspace(0.01, TN["zmax"], 200)
                med_tn, lo_tn, hi_tn = mc_sig(rows_tn.iloc[0], TN, z_tn, nmc=150, rng=rng)
                ax.plot(z_tn, med_tn, ls="--", color="#2166ac", lw=1.8, alpha=0.85,
                        label="Thin disc (linear σ_Z)" if pi == 0 else None)
                ax.fill_between(z_tn, lo_tn, hi_tn, color="#2166ac", alpha=0.10)

        # Combined thick disc linear σ_Z (red dashed)
        if fa_thick_lin is not None and not fa_thick_lin.empty:
            rows_tk = fa_thick_lin[np.isclose(fa_thick_lin.rm, rm)]
            if not rows_tk.empty:
                z_tk = np.linspace(0.01, TK["zmax"], 200)
                med_tk, lo_tk, hi_tk = mc_sig(rows_tk.iloc[0], TK, z_tk, nmc=150, rng=rng)
                ax.plot(z_tk, med_tk, ls="--", color="#b2182b", lw=1.8, alpha=0.80,
                        label="Combined thick (linear σ_Z)" if pi == 0 else None)
                ax.fill_between(z_tk, lo_tk, hi_tk, color="#b2182b", alpha=0.10)

        # MgFe-only thick disc, Cheng exact (darkorange dashed) — Cheng+2024 comparison
        if fa_mgfe_thick_lin is not None and not fa_mgfe_thick_lin.empty:
            rows_mg = fa_mgfe_thick_lin[np.abs(fa_mgfe_thick_lin.rm - rm) < 0.55]
            if not rows_mg.empty:
                z_mg = np.linspace(0.01, TK["zmax"], 200)
                med_mg, lo_mg, hi_mg = mc_sig(rows_mg.iloc[0], TK, z_mg, nmc=150, rng=rng)
                ax.plot(z_mg, med_mg, ls="--", color="darkorange", lw=2.5, alpha=0.95,
                        label="MgFe thick (linear σ_Z, Cheng+2024)" if pi == 0 else None,
                        zorder=5)
                ax.fill_between(z_mg, lo_mg, hi_mg, color="darkorange", alpha=0.14)

        zr = np.linspace(0.01, 4, 200)
        ax.plot(zr, shm_sigma(rm, zr), color="firebrick", lw=1.5,
                alpha=0.85, label="SHM" if pi == 0 else None)
        ax.plot(zr, baryonic_sigma(rm, zr), color="black", lw=1.4,
                alpha=0.85, label="Baryonic (BH\&G 2016)" if pi == 0 else None)
        _overlay_surface_z(ax, overlays, rm, zmax=4.0, zmin=0.0, label_once=(pi == 0))
        ax.text(0.97, 0.91, f"R = {rm:.1f} kpc", transform=ax.transAxes,
                ha="right", va="top", fontsize=10)
        ax.set_xlim(0, 4)
        ax.set_ylim(bottom=0)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(s2k, k2s))
        if pi % nc == 0:
            ax.set_ylabel(r"$\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        if pi >= len(all_rm) - nc:
            ax.set_xlabel(r"$|Z|$ [kpc]")
    for pi in range(len(all_rm), len(axes)):
        axes[pi].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outd / "fig7_sigma_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig8_combined(fa, outd, rng, overlays=None):
    overlays = overlays or []
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    for ax, zv in zip(axes, [0.3, 1.0, 3.0]):
        xs, ys, ylo, yhi = [], [], [], []
        for _, row in fa.sort_values("rm").iterrows():
            med, lo, hi = mc_sig(row, TA, np.array([zv]), nmc=200, rng=rng)
            xs.append(row.rm); ys.append(med[0]); ylo.append(lo[0]); yhi.append(hi[0])
        xs = np.asarray(xs); ys = np.asarray(ys); ylo = np.asarray(ylo); yhi = np.asarray(yhi)
        o = np.argsort(xs)
        ax.plot(xs[o], ys[o], ls=TA["ls"], color=TA["c"], lw=2.2, marker="o", ms=5, label=TA["lbl"])
        ax.fill_between(xs[o], ylo[o], yhi[o], color=TA["c"], alpha=TA["alpha"])
        rr = np.linspace(4.5, 11.5, 120)
        ax.plot(rr, [shm_sigma(r, [zv])[0] for r in rr], color="firebrick", lw=1.8, label="SHM")
        _overlay_surface_r(ax, overlays, zv)
        ax.text(0.03, 0.91, f"$|Z|={zv:.1f}$ kpc", transform=ax.transAxes, fontsize=11)
        ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
        ax.set_ylim(bottom=0)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), frameon=False, fontsize=8)
        ax.tick_params(direction="in", top=True, right=True)
        ax.secondary_yaxis("right", functions=(s2k, k2s))
    axes[-1].set_xlabel(r"$R$ [kpc]")
    fig.tight_layout()
    fig.savefig(outd / "fig8_sigma_vs_R.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_fig9_combined(fa, outd, rng, overlays=None):
    overlays = overlays or []
    fig, ax = plt.subplots(figsize=(8, 5.5))
    row = get_solar_row(fa)
    if row is not None:
        z = np.linspace(0.02, TA["zmax"], 300)
        for law, ls, lbl in [("exp", "-", "Exponential nu"), ("sech2", "--", r"sech$^2$ nu")]:
            med, lo, hi = mc_sig(row, TA, z, nmc=400, law=law, rng=rng)
            ax.plot(z, med, ls=ls, color=TA["c"], lw=2.2, label=lbl)
            ax.fill_between(z, lo, hi, color=TA["c"], alpha=TA["alpha"] * 0.8)
    zr = np.linspace(0.01, TA["zmax"], 300)
    ax.plot(zr, shm_sigma(8.5, zr), color="firebrick", lw=1.8, label="SHM")
    _overlay_surface_z(ax, overlays, 8.5, zmax=4.0, label_once=True)
    ax.set_xlabel(r"$|Z|$ [kpc]")
    ax.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]")
    ax.set_title("All stars: exp vs sech2")
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0, TA["zmax"])
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    ax.secondary_yaxis("right", functions=(s2k, k2s))
    fig.tight_layout()
    fig.savefig(outd / "fig9_exp_vs_sech2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cheng_linear_comparison(fa_thin_lin, fa_thick_lin, outd, rng,
                                  overlays=None, chem_data=None,
                                  fa_mgfe_thick_lin=None):
    """
    Reproduce the core Cheng+2024 conclusion using linear sigma_Z + exponential disc.

    When chem_data is provided, also computes the pure MgFe-only thick disc
    (chem_mgfe==2, 71K stars, Cheng's exact hZ=0.9/hR=2.0/hs=5.03 kpc) to
    show the exact Cheng-matching result in Panel A.

    Saves: fig_cheng_linear_comparison.png
    """
    overlays = overlays or []

    # ── Pure MgFe thick disc (Cheng's exact selection and parameters) ────────
    pp_cheng_thick = dict(hZ=0.90, hR=2.0, hs=5.03, hs_e=1.36,
                          zmax=4.0, c="#d6604d", ls="-", ng=200, alpha=0.22)
    if fa_mgfe_thick_lin is None:
        fa_mgfe_thick_lin = pd.DataFrame()
    if fa_mgfe_thick_lin.empty and chem_data is not None and "chem_mgfe" in chem_data:
        mask_mgfe_thick = chem_data["chem_mgfe"] == CHEM_THICK
        n_mgfe = int(mask_mgfe_thick.sum())
        print(f"  Pure MgFe thick disc: {n_mgfe:,} stars → linear cprof...")
        if n_mgfe >= pp_cheng_thick["ng"]:
            prof_mgfe = _cprof_subset(chem_data, mask_mgfe_thick,
                                      pp_cheng_thick["ng"], 1.0, 100, rng)
            if not prof_mgfe.empty:
                fa_mgfe_thick_lin, _ = fit_bins(
                    prof_mgfe, zmx=pp_cheng_thick["zmax"],
                    robust=True, clip=3.0, sigz_model="linear",
                    pp=pp_cheng_thick)
                fa_mgfe_thick_lin.to_csv(
                    outd / "chem_mgfe_thick_only_linear_fits.csv", index=False)
                print(f"  MgFe-only thick: {len(fa_mgfe_thick_lin)} R-bins fitted")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_tk_z  = axes[0, 0]   # thick solar circle
    ax_tn_z  = axes[0, 1]   # thin  solar circle
    ax_r_low = axes[1, 0]   # vs R at |Z|=0.3 kpc
    ax_r_hi  = axes[1, 1]   # vs R at |Z|=3.0 kpc

    zr_full = np.linspace(0.01, 4.5, 400)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _sigma_vs_z(ax, fa, pp, label, color):
        row = get_solar_row(fa, R=8.5)
        if row is None:
            return
        Z_eval = np.linspace(0.02, pp["zmax"], 250)
        med, lo, hi = mc_sig(row, pp, Z_eval, nmc=400, rng=rng)
        ax.plot(Z_eval, med, color=color, lw=2.5, label=label)
        ax.fill_between(Z_eval, lo, hi, color=color, alpha=0.22)

    def _sigma_vs_R(ax, fa, pp, label, color, z_target):
        if fa.empty or z_target > pp["zmax"] + 0.01:
            return
        Rv, Sv, Slo, Shi = [], [], [], []
        for _, row in fa.iterrows():
            med, lo, hi = mc_sig(row, pp, np.array([z_target]), nmc=200, rng=rng)
            if np.isfinite(med[0]):
                Rv.append(float(row.rm))
                Sv.append(float(med[0])); Slo.append(float(lo[0])); Shi.append(float(hi[0]))
        if Rv:
            ax.plot(Rv, Sv, color=color, lw=2, marker="o", ms=4.5, label=label)
            ax.fill_between(Rv, Slo, Shi, color=color, alpha=0.20)

    # ── Panel A: thick disc, solar circle ────────────────────────────────────
    # Orange: pure MgFe-only thick disc (71,633 stars, Cheng's exact parameters)
    if fa_mgfe_thick_lin is not None and not fa_mgfe_thick_lin.empty:
        _sigma_vs_z(ax_tk_z, fa_mgfe_thick_lin, pp_cheng_thick,
                    r"MgFe-only thick (Cheng: $h_Z$=0.9, $h_R$=2.0, $h_\sigma$=5.03)", "#d6604d")
    # Pre-computed overlays (includes MgFe thick from APOGEE-quality selection)
    _overlay_surface_z(ax_tk_z, overlays, 8.5, zmax=4.0, label_once=True)
    ax_tk_z.plot(zr_full, shm_sigma(8.5, zr_full), color="firebrick", lw=2.5, label="SHM")
    ax_tk_z.plot(zr_full, baryonic_sigma(8.5, zr_full), color="black", lw=1.8,
                 label="Baryonic (BH\&G 2016)")
    ax_tk_z.axvspan(1.0, 4.5, alpha=0.08, color="green")
    ax_tk_z.axvline(1.0, color="green", lw=1.5, ls="--", alpha=0.7, label="|Z| = 1 kpc")
    ax_tk_z.text(1.15, 5, "Thick disc matches SHM\nhere (Cheng+2024)", fontsize=8,
                 color="darkgreen", va="bottom")
    ax_tk_z.set_xlabel(r"$|Z|$ [kpc]", fontsize=12)
    ax_tk_z.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=11)
    ax_tk_z.set_title("(A) Thick disc — solar circle\n"
                       "Chemical thick: APOGEE-like MgFe selection by default", fontsize=9)
    ax_tk_z.set_xlim(0, 4); ax_tk_z.set_ylim(bottom=0, top=300)
    ax_tk_z.legend(frameon=False, fontsize=7, ncol=2)
    ax_tk_z.tick_params(direction="in", top=True, right=True)
    ax_tk_z.secondary_yaxis("right", functions=(s2k, k2s))

    # ── Panel B: thin disc, linear sigma_Z, solar circle ─────────────────────
    _sigma_vs_z(ax_tn_z, fa_thin_lin, TN, "Thin disc (linear σ_Z)", "#2166ac")
    ax_tn_z.plot(zr_full, shm_sigma(8.5, zr_full), color="firebrick", lw=2, label="SHM")
    ax_tn_z.plot(zr_full, baryonic_sigma(8.5, zr_full), color="black", lw=1.6,
                 label="Baryonic (BH\&G 2016)")
    _overlay_surface_z(ax_tn_z, overlays, 8.5, zmax=1.0, label_once=True)
    ax_tn_z.axvspan(0, 0.3, alpha=0.08, color="red")
    ax_tn_z.axvline(0.3, color="red", lw=1.5, ls="--", alpha=0.7, label="|Z| = 0.3 kpc")
    ax_tn_z.text(0.01, ax_tn_z.get_ylim()[1]*0.05 if ax_tn_z.get_ylim()[1] > 0 else 0.5,
                 "Below SHM ✗", fontsize=9, color="red", va="bottom")
    ax_tn_z.set_xlabel(r"$|Z|$ [kpc]", fontsize=12)
    ax_tn_z.set_ylabel(r"$\Sigma(R_\odot,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=11)
    ax_tn_z.set_title("(B) Thin disc, linear $\sigma_Z$ — solar circle\n"
                       r"Below SHM near mid-plane", fontsize=10)
    ax_tn_z.set_xlim(0, 1); ax_tn_z.set_ylim(bottom=0)
    ax_tn_z.legend(frameon=False, fontsize=8)
    ax_tn_z.tick_params(direction="in", top=True, right=True)
    ax_tn_z.secondary_yaxis("right", functions=(s2k, k2s))

    # ── Panel C: vs R at |Z| = 0.3 kpc — near mid-plane ─────────────────────
    R_shm = np.linspace(4.0, 12.0, 200)
    shm_03 = np.array([shm_sigma(r, np.array([0.3]))[0] for r in R_shm])
    bar_03 = np.array([baryonic_sigma(r, np.array([0.3]))[0] for r in R_shm])
    ax_r_low.plot(R_shm, shm_03, color="firebrick", lw=2, label="SHM")
    ax_r_low.plot(R_shm, bar_03, color="black", lw=1.6, label="Baryonic (BH\&G 2016)")
    _sigma_vs_R(ax_r_low, fa_thin_lin,  TN, "Thin disc (linear)", "#2166ac", 0.3)
    _sigma_vs_R(ax_r_low, fa_thick_lin, TK, "Thick disc (linear)", "#b2182b", 0.3)
    ax_r_low.axvline(8.0, color="0.5", lw=1, ls=":", alpha=0.6)
    ax_r_low.text(8.05, 0, "Solar R", fontsize=7, color="0.5", va="bottom")
    ax_r_low.set_xlabel(r"$R$ [kpc]", fontsize=12)
    ax_r_low.set_ylabel(r"$\Sigma(R,0.3\,\mathrm{kpc})$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=11)
    ax_r_low.set_title(r"(C) $|Z| = 0.3$ kpc — all populations depart from SHM"
                        "\n(inner disc: thin disc below baryonic floor)", fontsize=10)
    ax_r_low.set_ylim(bottom=0)
    ax_r_low.legend(frameon=False, fontsize=8)
    ax_r_low.tick_params(direction="in", top=True, right=True)
    ax_r_low.secondary_yaxis("right", functions=(s2k, k2s))

    # ── Panel D: vs R at |Z| = 3.0 kpc — thick disc inner excess ─────────────
    shm_30 = np.array([shm_sigma(r, np.array([3.0]))[0] for r in R_shm])
    bar_30 = np.array([baryonic_sigma(r, np.array([3.0]))[0] for r in R_shm])
    ax_r_hi.plot(R_shm, shm_30, color="firebrick", lw=2, label="SHM")
    ax_r_hi.plot(R_shm, bar_30, color="black", lw=1.6, label="Baryonic (BH\&G 2016)")
    _sigma_vs_R(ax_r_hi, fa_thin_lin,  TN, "Thin disc (linear)",  "#2166ac", 3.0)
    _sigma_vs_R(ax_r_hi, fa_thick_lin, TK, "Thick disc (linear)", "#b2182b", 3.0)
    ax_r_hi.axvline(8.0, color="0.5", lw=1, ls=":", alpha=0.6)
    ax_r_hi.axvspan(4.0, 8.0, alpha=0.04, color="orange")
    ax_r_hi.text(4.2, 0, "Inner Galaxy\nexcess ✗", fontsize=8, color="orange", va="bottom")
    ax_r_hi.set_xlabel(r"$R$ [kpc]", fontsize=12)
    ax_r_hi.set_ylabel(r"$\Sigma(R,3.0\,\mathrm{kpc})$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=11)
    ax_r_hi.set_title(r"(D) $|Z| = 3.0$ kpc — thick disc exceeds SHM in inner Galaxy"
                       "\n(possible dark disc signature; Purcell+2009)", fontsize=10)
    ax_r_hi.set_ylim(bottom=0)
    ax_r_hi.legend(frameon=False, fontsize=8)
    ax_r_hi.tick_params(direction="in", top=True, right=True)
    ax_r_hi.secondary_yaxis("right", functions=(s2k, k2s))

    fig.suptitle(
        "Cheng+2024 Conclusion — Linear $\\sigma_Z$ + Exponential Disc Assumption\n"
        "Thick disc at |Z|>1 kpc (solar circle): agrees with SHM.  "
        "All other cases: depart from SHM.",
        fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(outd / "fig_cheng_linear_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: fig_cheng_linear_comparison.png")


def compute_thick_disc_kg_profile(chem_da, zmax=1.5, n_zbins=12, rmin=7.0, rmax=10.0, ng=150):
    """Compute sigma_Z^2 vs |Z| for MgFe-only thick disc from raw chemistry NPZ.
    Pools stars over rmin-rmax kpc. Returns:
      (z_bin, sig2_bin, z_grp, sig2_grp, es2_grp)
    where z_grp/sig2_grp are group-level estimates (background scatter, Cheng style)
    and z_bin/sig2_bin are uniform |Z|-bin medians used for fitting.
    """
    mask = ((chem_da['chem_mgfe'] == CHEM_THICK) &
            (chem_da['R'] >= rmin) & (chem_da['R'] <= rmax))
    if mask.sum() < ng:
        return None, None, None, None, None
    Z = chem_da['Z'][mask]
    vZ = chem_da['vZ'][mask]
    az = np.abs(Z)

    # Group-level estimates sorted by |Z| (background scatter like Cheng's fig10)
    order = np.argsort(az)
    az_s, vZ_s = az[order], vZ[order]
    n_grp = len(az_s) // ng
    zg, yg = [], []
    for g in range(n_grp):
        sl = slice(g * ng, (g + 1) * ng)
        zg.append(float(np.median(az_s[sl])))
        yg.append(float(np.var(vZ_s[sl])))
    zg, yg = np.array(zg), np.array(yg)

    # Bootstrap error for each group (rough estimate: Poisson-like 1/sqrt(N))
    eg = yg / np.sqrt(ng / 2.0)  # approximate standard error on variance estimate

    # Uniform |Z|-bin medians (used for fit — each bin contributes equally)
    edges = np.linspace(0, zmax, n_zbins + 1)
    zb, yb = [], []
    for i in range(n_zbins):
        m = (az >= edges[i]) & (az < edges[i + 1])
        if m.sum() < 20:
            continue
        # Use group-based median within the bin for robustness
        vZ_bin = vZ[m]
        n_g_bin = len(vZ_bin) // ng
        if n_g_bin >= 2:
            ord_b = np.argsort(np.abs(Z[m]))
            grp_vars = [np.var(vZ_bin[ord_b[j*ng:(j+1)*ng]]) for j in range(n_g_bin)]
            yb.append(float(np.median(grp_vars)))
        else:
            yb.append(float(np.var(vZ_bin)))
        zb.append(0.5 * (edges[i] + edges[i + 1]))

    return np.array(zb), np.array(yb), zg, yg, eg


def plot_sigz_surface_density_grid(fa_thin_lin, fa_thick_lin, outd, rng):
    """
    Grid figure: Sigma(R, |Z|) from linear / quadratic / tanh sigma_Z fits.
    One panel per R bin.  Combined thin (blue shades) and combined thick (red shades),
    with solid/dashed/dash-dot line styles for the three sigma_Z models.
    Requires all three model parameter sets to be stored in the input DataFrames
    (true for CSVs produced by fit_bins with choice='best' or any choice, as all
    three fits are always saved alongside the BIC winner).
    """
    if fa_thin_lin.empty and fa_thick_lin.empty:
        return

    all_rm = sorted(set(
        list(fa_thin_lin["rm"].unique()) + list(fa_thick_lin["rm"].unique())
    ))
    if not all_rm:
        return

    nc = 2
    nr = int(np.ceil(len(all_rm) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, nr * 3.2), squeeze=False)
    axes_flat = axes.ravel()

    # Model definitions: (sigz_model key, linestyle, short label)
    model_specs = [
        ("linear",    "-",  "Linear"),
        ("quadratic", "--", "Quadratic"),
        ("tanh",      "-.", "Tanh"),
    ]
    thin_color  = "#2166ac"   # blue  — combined thin
    thick_color = "#cb181d"   # red   — combined thick

    for pi, rm in enumerate(all_rm):
        ax = axes_flat[pi]

        thin_rows  = fa_thin_lin[np.isclose(fa_thin_lin["rm"],  rm)]
        thick_rows = fa_thick_lin[np.isclose(fa_thick_lin["rm"], rm)]

        if not thin_rows.empty:
            rl = float(thin_rows.iloc[0]["rl"]); rh = float(thin_rows.iloc[0]["rh"])
        elif not thick_rows.empty:
            rl = float(thick_rows.iloc[0]["rl"]); rh = float(thick_rows.iloc[0]["rh"])
        else:
            ax.set_visible(False); continue

        ax.set_title(f"R = {rl:.1f}–{rh:.1f} kpc", fontsize=9)

        z_tn = np.linspace(0.02, TN["zmax"], 200)
        z_tk = np.linspace(0.02, TK["zmax"], 200)

        for model_name, ls, mlabel in model_specs:
            show_label = (pi == 0)

            # ── Combined thin ──────────────────────────────────────────────
            if not thin_rows.empty:
                row = thin_rows.iloc[0].copy()
                row["sigz_model"] = model_name
                # Skip if required params are NaN
                _ok = True
                if model_name == "quadratic" and (
                    not np.isfinite(float(_row_get(row, "sigz_q", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_s0sq", np.nan)))
                ):
                    _ok = False
                elif model_name == "tanh" and (
                    not np.isfinite(float(_row_get(row, "sigz_k", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_L", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_s0", np.nan)))
                ):
                    _ok = False
                if _ok:
                    try:
                        med, lo, hi = mc_sig(row, TN, z_tn, nmc=150, rng=rng)
                        ax.plot(z_tn, med, ls=ls, color=thin_color, lw=1.8,
                                label=f"Thin – {mlabel}" if show_label else None)
                        ax.fill_between(z_tn, lo, hi, color=thin_color, alpha=0.12)
                    except Exception:
                        pass

            # ── Combined thick ─────────────────────────────────────────────
            if not thick_rows.empty:
                row = thick_rows.iloc[0].copy()
                row["sigz_model"] = model_name
                _ok = True
                if model_name == "quadratic" and (
                    not np.isfinite(float(_row_get(row, "sigz_q", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_s0sq", np.nan)))
                ):
                    _ok = False
                elif model_name == "tanh" and (
                    not np.isfinite(float(_row_get(row, "sigz_k", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_L", np.nan))) or
                    not np.isfinite(float(_row_get(row, "sigz_s0", np.nan)))
                ):
                    _ok = False
                if _ok:
                    try:
                        med, lo, hi = mc_sig(row, TK, z_tk, nmc=150, rng=rng)
                        ax.plot(z_tk, med, ls=ls, color=thick_color, lw=1.8,
                                label=f"Thick – {mlabel}" if show_label else None)
                        ax.fill_between(z_tk, lo, hi, color=thick_color, alpha=0.12)
                    except Exception:
                        pass

        ax.set_ylim(bottom=0)
        ax.set_xlim(0, max(TN["zmax"], TK["zmax"]))
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.20)
        if pi % nc == 0:
            ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=9)
        if pi >= (nr - 1) * nc:
            ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=9)

    # Hide unused panels
    for pi in range(len(all_rm), len(axes_flat)):
        axes_flat[pi].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        r"Surface density $\Sigma(R,|Z|)$: Linear vs Quadratic vs Tanh $\sigma_Z$" + "\n"
        r"Blue = chemical thin · Red = chemical thick",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = outd / "fig_sigz_model_sigma_comparison_grid.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _density_bin_points(R, Z, rl, rh, zmax, dz, nmin):
    """Binned uncorrected number-density points in one radial annulus."""
    R = np.asarray(R, dtype=float)
    Zabs = np.abs(np.asarray(Z, dtype=float))
    z_edges = np.arange(0.0, zmax + dz * 0.5, dz)
    if z_edges[-1] < zmax:
        z_edges = np.r_[z_edges, zmax]
    rows = []
    for zl, zh in zip(z_edges[:-1], z_edges[1:]):
        m = (R >= rl) & (R < rh) & (Zabs >= zl) & (Zabs < zh)
        n = int(m.sum())
        if n < nmin:
            continue
        volume = np.pi * (rh ** 2 - rl ** 2) * (zh - zl) * 2.0
        rows.append(dict(z_mid=0.5 * (zl + zh), z_lo=zl, z_hi=zh,
                         N=n, volume=volume,
                         lnnu=np.log(n / volume),
                         lnnu_err=1.0 / np.sqrt(max(n, 1))))
    return pd.DataFrame(rows)


def _density_window_cells(R, Z, rl, rh, zmax, dz, rbw, nmin):
    """2D (R, |Z|) cells for local exp(R,Z) density-gradient fits."""
    R = np.asarray(R, dtype=float)
    Zabs = np.abs(np.asarray(Z, dtype=float))
    r_edges = np.arange(rl, rh + rbw * 0.5, rbw)
    z_edges = np.arange(0.0, zmax + dz * 0.5, dz)
    rows = []
    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        for z0, z1 in zip(z_edges[:-1], z_edges[1:]):
            m = (R >= r0) & (R < r1) & (Zabs >= z0) & (Zabs < z1)
            n = int(m.sum())
            if n < nmin:
                continue
            volume = np.pi * (r1 ** 2 - r0 ** 2) * (z1 - z0) * 2.0
            rows.append(dict(R_mid=0.5 * (r0 + r1), z_mid=0.5 * (z0 + z1),
                             N=n, volume=volume,
                             lnnu=np.log(n / volume),
                             lnnu_err=1.0 / np.sqrt(max(n, 1))))
    return pd.DataFrame(rows)


def _weighted_lstsq(X, y, yerr):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if ok.sum() < X.shape[1] + 1:
        return None
    X, y, yerr = X[ok], y[ok], yerr[ok]
    sw = 1.0 / yerr
    beta, _, _, _ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return beta


def _log_sech2_profile(z, hZ):
    x = np.clip(np.asarray(z, dtype=float) / (2.0 * float(hZ)), -50.0, 50.0)
    return -2.0 * np.log(np.cosh(x))


def _fit_density_kind(points, kind, pp, R=None, Z=None, rl=None, rh=None,
                      zmax=None, dz=None, rbw=None, nmin=None):
    if points is None or points.empty or len(points) < 3:
        return dict(ok=False, reason="too few vertical density bins")

    z = points.z_mid.values
    y = points.lnnu.values
    e = points.lnnu_err.values
    rm = 0.5 * (float(rl) + float(rh))

    if kind == "exp_vertical":
        X = np.column_stack([np.ones(len(points)), z])
        beta = _weighted_lstsq(X, y, e)
        if beta is None:
            return dict(ok=False, reason="exp vertical least-squares failed")
        A, bZ = [float(v) for v in beta]
        return dict(ok=True, A=A, bZ=bZ, bR=-1.0 / pp["hR"],
                    hZ=(-1.0 / bZ if bZ < 0 else np.nan),
                    hR=pp["hR"], reason="")

    if kind == "sech2_vertical":
        def objective(hZ):
            shape = _log_sech2_profile(z, hZ)
            w = 1.0 / np.maximum(e, 1e-6) ** 2
            A = np.sum(w * (y - shape)) / np.sum(w)
            return float(np.sum(w * (y - A - shape) ** 2))

        lo = 0.05 if pp["zmax"] <= 1.1 else 0.10
        hi = 5.0 if pp["zmax"] <= 1.1 else 10.0
        res = minimize_scalar(objective, bounds=(lo, hi), method="bounded")
        if not res.success:
            return dict(ok=False, reason="sech2 vertical optimization failed")
        hZ = float(res.x)
        shape = _log_sech2_profile(z, hZ)
        w = 1.0 / np.maximum(e, 1e-6) ** 2
        A = float(np.sum(w * (y - shape)) / np.sum(w))
        return dict(ok=True, A=A, bZ=np.nan, bR=-1.0 / pp["hR"],
                    hZ=hZ, hR=pp["hR"], reason="")

    if kind == "exp_zr":
        if R is None or Z is None:
            return dict(ok=False, reason="raw R/Z unavailable for exp_zr")
        half_width = max(1.0, float(rh) - float(rl))
        rlw = max(3.0, float(rl) - half_width)
        rhw = min(12.5, float(rh) + half_width)
        cells = _density_window_cells(R, Z, rlw, rhw, zmax, dz, rbw, nmin)
        if cells.empty or len(cells) < 8:
            return dict(ok=False, reason="too few 2D density cells")
        X = np.column_stack([
            np.ones(len(cells)),
            cells.R_mid.values - rm,
            cells.z_mid.values,
        ])
        beta = _weighted_lstsq(X, cells.lnnu.values, cells.lnnu_err.values)
        if beta is None:
            return dict(ok=False, reason="exp(R,Z) least-squares failed")
        A, bR, bZ = [float(v) for v in beta]
        return dict(ok=True, A=A, bZ=bZ, bR=bR,
                    hZ=(-1.0 / bZ if bZ < 0 else np.nan),
                    hR=(-1.0 / bR if bR < 0 else np.nan),
                    reason="")

    raise ValueError(f"Unknown density kind: {kind}")


def fit_binwise_number_density_models(chem_da, thin_mask, thick_mask,
                                      fa_thin_lin, fa_thick_lin, outd,
                                      rbw=0.5):
    """Fit uncorrected number-density models in each radial bin.

    These are diagnostic fits to raw catalogue counts. They are not
    selection-function corrected.
    """
    outd = Path(outd)
    kinds = ["exp_vertical", "exp_zr", "sech2_vertical"]
    pop_specs = [
        ("thin", thin_mask, TN, fa_thin_lin, 0.10, 30),
        ("thick", thick_mask, TK, fa_thick_lin, 0.20, 20),
    ]
    point_rows, param_rows = [], []

    for pop, mask, pp, fits, dz, nmin in pop_specs:
        if fits is None or fits.empty or mask is None:
            continue
        R = np.asarray(chem_da["R"], dtype=float)[mask]
        Z = np.asarray(chem_da["Z"], dtype=float)[mask]
        for _, fitrow in fits.sort_values("rm").iterrows():
            rl, rh, rm = float(fitrow.rl), float(fitrow.rh), float(fitrow.rm)
            points = _density_bin_points(R, Z, rl, rh, pp["zmax"], dz, nmin)
            for _, prow in points.iterrows():
                d = prow.to_dict()
                d.update(pop=pop, rl=rl, rh=rh, rm=rm, zmax=pp["zmax"])
                point_rows.append(d)
            for kind in kinds:
                res = _fit_density_kind(points, kind, pp, R=R, Z=Z,
                                        rl=rl, rh=rh, zmax=pp["zmax"],
                                        dz=dz, rbw=rbw, nmin=nmin)
                res.update(pop=pop, kind=kind, rl=rl, rh=rh, rm=rm,
                           zmax=pp["zmax"], n_density_bins=int(len(points)),
                           n_stars=int(((R >= rl) & (R < rh) &
                                        (np.abs(Z) <= pp["zmax"])).sum()))
                param_rows.append(res)

    points_df = pd.DataFrame(point_rows)
    params_df = pd.DataFrame(param_rows)
    points_df.to_csv(outd / "number_density_binwise_points.csv", index=False)
    params_df.to_csv(outd / "number_density_binwise_fit_params.csv", index=False)
    return points_df, params_df


def _density_model_lnnu(kind, params, z):
    z = np.asarray(z, dtype=float)
    A = float(params["A"])
    if kind in ("exp_vertical", "exp_zr"):
        return A + float(params["bZ"]) * z
    if kind == "sech2_vertical":
        return A + _log_sech2_profile(z, float(params["hZ"]))
    raise ValueError(kind)


def plot_binwise_number_density_fits(points_df, params_df, outd, kind):
    if points_df.empty or params_df.empty:
        return
    all_rm = sorted(points_df.rm.dropna().unique())
    if not all_rm:
        return
    nc = 2
    nr = int(np.ceil(len(all_rm) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, nr * 3.1), squeeze=False)
    axes_flat = axes.ravel()
    colors = {"thin": "#2166ac", "thick": "#cb181d"}
    labels = {
        "exp_vertical": r"vertical exponential $\nu(|Z|)$",
        "exp_zr": r"local double exponential $\nu(R,|Z|)$",
        "sech2_vertical": r"vertical sech$^2$ $\nu(|Z|)$",
    }
    for pi, rm in enumerate(all_rm):
        ax = axes_flat[pi]
        sub_points = points_df[np.isclose(points_df.rm, rm)]
        if sub_points.empty:
            ax.set_visible(False)
            continue
        rl = float(sub_points.rl.iloc[0])
        rh = float(sub_points.rh.iloc[0])
        ax.set_title(f"R = {rl:.1f}-{rh:.1f} kpc", fontsize=9)
        for pop in ("thin", "thick"):
            pnts = sub_points[sub_points["pop"] == pop]
            pars = params_df[(params_df.kind == kind) &
                             (params_df["pop"] == pop) &
                             np.isclose(params_df.rm, rm)]
            if not pnts.empty:
                ax.errorbar(pnts.z_mid, pnts.lnnu, yerr=pnts.lnnu_err,
                            fmt="o", ms=3.0, color=colors[pop], alpha=0.75,
                            label=pop if pi == 0 else None)
            if not pars.empty and bool(pars.iloc[0].ok):
                pp = TN if pop == "thin" else TK
                z = np.linspace(0.0, pp["zmax"], 200)
                y = _density_model_lnnu(kind, pars.iloc[0], z)
                ax.plot(z, y, color=colors[pop], lw=1.8)
        ax.set_xlim(0, 4.0)
        ax.grid(alpha=0.18)
        ax.tick_params(direction="in", top=True, right=True)
        if pi % nc == 0:
            ax.set_ylabel(r"$\ln(N/V)$ [arb.]", fontsize=9)
        if pi >= (nr - 1) * nc:
            ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=9)
    for pi in range(len(all_rm), len(axes_flat)):
        axes_flat[pi].set_visible(False)
    handles, leg_labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, frameon=False, ncol=2,
                   loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Bin-wise uncorrected number-density fits: " + labels[kind],
                 fontsize=11, y=1.01)
    fig.tight_layout()
    out = Path(outd) / f"fig_number_density_binwise_fits_{kind}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def _density_params_for_bin(params_df, kind, pop, rm):
    sub = params_df[(params_df.kind == kind) &
                    (params_df["pop"] == pop) &
                    np.isclose(params_df.rm, rm)]
    if sub.empty or not bool(sub.iloc[0].ok):
        return None
    return sub.iloc[0]


def _sig_with_binwise_density(row, pp, z, density_params, kind, model_name):
    row = row.copy()
    row["sigz_model"] = model_name
    if kind == "sech2_vertical":
        hZ = float(density_params["hZ"])
        dlnz = lambda _R, zz: -np.tanh(abs(float(zz)) / (2.0 * hZ)) / hZ
    else:
        dlnz = float(density_params["bZ"])
    dlnr = float(density_params["bR"])
    c = jeans_comps_general_density(
        z, row.rm, row.a, row.b, row.m, pp["hs"], dlnz, dlnr,
        rz_b=_row_get(row, "rz_b", 0.0), **_row_sigz_params(row))
    return c["Sig"]


def _sigz_model_params_available(row, model_name):
    if model_name == "quadratic":
        return (np.isfinite(float(_row_get(row, "sigz_q", np.nan))) and
                np.isfinite(float(_row_get(row, "sigz_s0sq", np.nan))))
    if model_name == "tanh":
        return (np.isfinite(float(_row_get(row, "sigz_k", np.nan))) and
                np.isfinite(float(_row_get(row, "sigz_L", np.nan))) and
                np.isfinite(float(_row_get(row, "sigz_s0", np.nan))))
    return True


def plot_sigz_surface_density_grid_binwise_density(fa_thin_lin, fa_thick_lin,
                                                   density_params, outd, kind):
    if density_params is None or density_params.empty:
        return
    all_rm = sorted(set(
        list(fa_thin_lin["rm"].unique()) + list(fa_thick_lin["rm"].unique())
    ))
    if not all_rm:
        return

    nc = 2
    nr = int(np.ceil(len(all_rm) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, nr * 3.2), squeeze=False)
    axes_flat = axes.ravel()
    model_specs = [
        ("linear", "-", "Linear"),
        ("quadratic", "--", "Quadratic"),
        ("tanh", "-.", "Tanh"),
    ]
    colors = {"thin": "#2166ac", "thick": "#cb181d"}
    kind_labels = {
        "exp_vertical": "bin-wise vertical exponential density",
        "exp_zr": "bin-wise local exp(R,|Z|) density",
        "sech2_vertical": r"bin-wise vertical sech$^2$ density",
    }

    for pi, rm in enumerate(all_rm):
        ax = axes_flat[pi]
        panel_vals = []
        thin_rows = fa_thin_lin[np.isclose(fa_thin_lin["rm"], rm)]
        thick_rows = fa_thick_lin[np.isclose(fa_thick_lin["rm"], rm)]
        if not thin_rows.empty:
            rl, rh = float(thin_rows.iloc[0].rl), float(thin_rows.iloc[0].rh)
        elif not thick_rows.empty:
            rl, rh = float(thick_rows.iloc[0].rl), float(thick_rows.iloc[0].rh)
        else:
            ax.set_visible(False)
            continue
        ax.set_title(f"R = {rl:.1f}-{rh:.1f} kpc", fontsize=9)
        for pop, rows, pp in [("thin", thin_rows, TN), ("thick", thick_rows, TK)]:
            if rows.empty:
                continue
            dens = _density_params_for_bin(density_params, kind, pop, rm)
            if dens is None:
                continue
            z = np.linspace(0.02, pp["zmax"], 200)
            for model_name, ls, mlabel in model_specs:
                row = rows.iloc[0].copy()
                if not _sigz_model_params_available(row, model_name):
                    continue
                try:
                    sig = _sig_with_binwise_density(row, pp, z, dens, kind, model_name)
                except Exception:
                    continue
                panel_vals.extend(np.asarray(sig)[np.isfinite(sig)].tolist())
                ax.plot(z, sig, color=colors[pop], ls=ls, lw=1.8,
                        label=f"{pop.capitalize()} - {mlabel}" if pi == 0 else None)
        ax.axhline(0, color="0.35", lw=0.8, ls=":")
        ax.set_xlim(0, max(TN["zmax"], TK["zmax"]))
        if panel_vals:
            ymin = min(0.0, np.nanmin(panel_vals))
            ymax = max(1.0, np.nanmax(panel_vals))
            pad = 0.08 * max(ymax - ymin, 1.0)
            ax.set_ylim(ymin - pad, ymax + pad)
        ax.tick_params(direction="in", top=True, right=True)
        ax.grid(alpha=0.20)
        if pi % nc == 0:
            ax.set_ylabel(r"$\Sigma(R,|Z|)$ [$M_\odot\,\mathrm{pc}^{-2}$]", fontsize=9)
        if pi >= (nr - 1) * nc:
            ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=9)

    for pi in range(len(all_rm), len(axes_flat)):
        axes_flat[pi].set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3,
                   loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        r"Surface density with " + kind_labels[kind] + "\n"
        r"Uncorrected number-density fits; negative values are shown",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    out = Path(outd) / f"fig_sigz_model_sigma_comparison_grid_binwise_density_{kind}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_fig10_combined(pa, outd, overlay_dir=None, chem_da=None):
    """KG integral method (Kuijken & Gilmore 1989a).
    When overlay_dir is given, uses pre-computed thin/thick chemical disc profiles at
    R=8.5 kpc and fits separate disc+DM models, producing a 2-panel figure like the paper.
    Thin disc is restricted to |Z| < 1 kpc per the paper's analysis cut."""

    def _load_solar_prof(prof_path, hZ, zmax, zmin=0.0):
        if not Path(prof_path).exists():
            return None, None, None
        pf = _profile_overlay_df(prof_path)
        rvals = pf.rm.dropna().unique()
        nearest = rvals[np.argmin(np.abs(rvals - 8.5))]
        sub = pf[np.isclose(pf.rm, nearest)].copy()
        sub = sub[np.isfinite(sub.sZ) & (sub.sZ > 0) &
                  (np.abs(sub.zm) >= zmin) & (np.abs(sub.zm) <= zmax)]
        if sub.empty:
            return None, None, None
        az = np.abs(sub.zm.values)
        sig2 = sub.sZ.values ** 2
        esig2 = 2.0 * sub.sZ.values * sub.sZe.values
        o = np.argsort(az)
        return az[o], sig2[o], esig2[o]

    # For fig10, always use MgFe-only precomputed profiles (Cheng+2024 exact selection).
    # Combined αFe+MgFe thick disc includes halo contamination at large |Z| that
    # breaks the KG fit. MgFe-only gives clean σ_Z profiles matching Cheng's fig10.
    ov = Path(overlay_dir) if overlay_dir else None
    thin_path  = str(ov / "mgfe_feh_thin_chem"  / "all_no_mgfe_mgfe_feh_thin_chem_velocity_dispersion_profiles.csv")  if ov else None
    thick_path = str(ov / "mgfe_feh_thick_chem" / "all_no_mgfe_mgfe_feh_thick_chem_velocity_dispersion_profiles.csv") if ov else None

    # Thin disc: use the selected chemical thin profile from outd when available.
    thin_p = outd / "chem_combined_thin_velocity_dispersion_profiles.csv"
    if thin_p.exists():
        print("[fig10] Thin disc: selected chemical profile from outd.")
        az_n, s2_n, es2_n = _load_solar_prof(str(thin_p), TN["hZ"], zmax=1.0, zmin=0.0)
    else:
        az_n, s2_n, es2_n = solar_sig2(pa)

    # Thick disc: raw NPZ MgFe-only group-level estimates (R=7-10 kpc, zmax=1.5 kpc).
    # These are more robust than the per-star CSV at low z.
    az_k_grp = s2_k_grp = es2_k_grp = None
    if chem_da is not None:
        print("[fig10] Thick disc: raw NPZ MgFe-only (R=7-10 kpc, zmax=1.5 kpc).")
        result_k = compute_thick_disc_kg_profile(chem_da, zmax=1.5, n_zbins=12)
        az_k, s2_k, az_k_grp, s2_k_grp, es2_k_grp = result_k
        es2_k = np.zeros_like(s2_k) if s2_k is not None else None
        if az_k is None:
            chem_da = None
    if chem_da is None:
        if thick_path and Path(thick_path).exists():
            print("[fig10] Thick disc: MgFe-only precomputed profile from overlay_dir.")
            az_k, s2_k, es2_k = _load_solar_prof(thick_path, TK["hZ"], zmax=1.5, zmin=0.0)
        else:
            thick_p = outd / "chem_combined_thick_velocity_dispersion_profiles.csv"
            az_k, s2_k, es2_k = _load_solar_prof(str(thick_p), TK["hZ"], zmax=1.5, zmin=0.0) if thick_p.exists() else (az_n, s2_n, es2_n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    res_n, res_k = None, None

    # Group-level scatter for each panel (thick disc uses raw NPZ groups; thin uses profile CSV groups)
    grp_data = {
        "Thin disc":  (None, None, None),  # thin uses profile CSV which is already group-level
        "Thick disc": (az_k_grp, s2_k_grp, es2_k_grp),
    }

    for ax, nm, pp, az, s2, es2 in [
        (axes[0], "Thin disc",  TN, az_n, s2_n, es2_n),
        (axes[1], "Thick disc", TK, az_k, s2_k, es2_k),
    ]:
        if az is None:
            ax.set_visible(False)
            continue
        # Use the exact KG89 force and tilt-integral model for both populations.
        # A simplified moment formula is useful only as a diagnostic elsewhere; silently
        # falling back to it here makes the figure cease to be a KG89 reproduction.
        res = fit_kg_exact(az, s2, es2, pp["hZ"],
                           hR=pp.get("hR", TA["hR"]), R=8.5,
                           D=0.4, alpha=2.0)
        if res is None:
            print(f"[fig10] WARNING: exact KG89 fit failed for {nm}; no simplified fallback used.")
        elif res.get("K_at_lower") or res.get("K_at_upper") or res.get("F_at_lower") or res.get("F_at_upper"):
            flags = [name for name in ("K_at_lower", "K_at_upper", "F_at_lower", "F_at_upper")
                     if res.get(name)]
            print(f"[fig10] WARNING: exact KG89 fit for {nm} hit parameter boundary: {', '.join(flags)}")

        # Background scatter — group-level sigma_Z^2 points (Cheng style)
        az_g, s2_g, es2_g = grp_data[nm]
        if az_g is not None and len(az_g) > 0:
            # Thick disc: show individual group-level points as background scatter with error bars
            ok_g = np.isfinite(az_g) & np.isfinite(s2_g) & (s2_g > 0)
            ax.errorbar(az_g[ok_g], s2_g[ok_g], yerr=es2_g[ok_g],
                        fmt="o", color=pp["c"], ms=4, alpha=0.55,
                        capsize=1.5, lw=0.7, elinewidth=0.7, zorder=1)
        else:
            # Thin disc: profile CSV rows are already group-level — show as scatter
            ok_raw = np.isfinite(az) & np.isfinite(s2) & (s2 > 0)
            ax.errorbar(az[ok_raw], s2[ok_raw], yerr=es2[ok_raw],
                        fmt="o", color=pp["c"], ms=4, alpha=0.55,
                        capsize=1.5, lw=0.7, elinewidth=0.7, zorder=1)

        if res is not None:
            # Binned medians on top with larger markers and error bars
            ax.errorbar(res["z"], res["y"], yerr=res["e"], fmt="o", color=pp["c"],
                        ms=7, capsize=3, lw=1.5, elinewidth=1.5,
                        label="Data (binned median)", zorder=3,
                        markeredgewidth=1.0, markeredgecolor="white")
        if nm == "Thin disc":
            res_n = res
        else:
            res_k = res
        if res is not None:
            zmax_curve = (az_g[ok_g].max() if (az_g is not None and len(az_g) > 0 and ok_g.any())
                          else az[np.isfinite(az)].max()) * 1.05
            # Exact KG89 eq 39+50 with tilt; integral grid extends beyond plotted data.
            z_hi_crv = max(zmax_curve + 0.1, float(res["hZ"]) * 12.0)
            zz_fine  = np.linspace(0.01, z_hi_crv, 600)
            nu_zz    = np.exp(-zz_fine / float(res["hZ"]))
            Kz_zz    = kg.kg_Kz_eq39(zz_fine, res["K"], res.get("D", 0.4), res["F"])
            s2_zz    = kg.sigma_zz2_with_tilt_eq50(zz_fine, 8.5, nu_zz, Kz_zz,
                                                     alpha=res.get("alpha", 2.0),
                                                     hR=res.get("hR", pp.get("hR", TA["hR"])))
            s2_zz    = np.clip(s2_zz, 0, None)
            lbl = (f'$\\Sigma_{{disc}}={res["K"]:.1f}\\pm{res["K_e"]:.1f}$ $M_\\odot$ pc$^{{-2}}$\n'
                   f'$\\rho_{{DM}}={res["F"]:.4f}\\pm{res["F_e"]:.4f}$ $M_\\odot$ pc$^{{-3}}$\n'
                   r'(KG89 eq 39$+$50, tilt $\mathcal{S}$)')
            if res.get("K_at_lower") or res.get("K_at_upper") or res.get("F_at_lower") or res.get("F_at_upper"):
                lbl += "\n(boundary solution)"
            zz_plot = np.linspace(0.01, zmax_curve, 300)
            s2_plot = np.interp(zz_plot, zz_fine, s2_zz)
            ax.plot(zz_plot, s2_plot, "k-", lw=2, label=lbl, zorder=4)
        ax.set_xlabel(r"$|Z|$ [kpc]", fontsize=12)
        ax.set_ylabel(r"$\sigma^2_Z$ [km$^2$ s$^{-2}$]", fontsize=12)
        ax.set_title(f"R = 8.5 kpc {nm}", fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
        ax.tick_params(direction="in", top=True, right=True)
        # X-axis: thick disc fit is restricted to 1.5 kpc; thin disc to 1.0 kpc
        fit_zmax = 1.0 if nm == "Thin disc" else 1.5
        # Show data to max group range; extend x-axis to match Cheng's style
        data_zmax = (az_g[ok_g].max() if (az_g is not None and len(az_g) > 0 and ok_g.any())
                     else az[np.isfinite(az)].max() if az is not None else fit_zmax)
        ax.set_xlim(0, min(data_zmax * 1.03, fit_zmax * 2.5))
        # Y-axis: p2-p98 of displayed data
        if az_g is not None and len(az_g) > 0:
            # Use only data in the displayed x-range
            vis = ok_g & (az_g <= fit_zmax * 2.5)
            ylo = max(0, np.percentile(s2_g[vis], 2) * 0.92) if vis.any() else 0
            yhi = np.percentile(s2_g[vis], 98) * 1.12 if vis.any() else None
        elif res is not None:
            ylo = max(0, res["y"].min() * 0.85)
            yhi = res["y"].max() * 1.20
        else:
            ylo, yhi = 0, None
        if yhi is not None:
            ax.set_ylim(ylo, yhi)

    fig.tight_layout()
    fig.savefig(outd / "fig10_KG_integral.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res_n, res_k


def save_surface_table_combined(fa, outd, rng):
    rows = []
    for _, row in fa.iterrows():
        for zv in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            med, lo, hi = mc_sig(row, TA, np.array([zv]), nmc=150, rng=rng)
            rows.append(dict(pop="all", R=row.rm, Z=zv,
                             Sig=med[0], Sig_lo=lo[0], Sig_hi=hi[0],
                             Kz=s2k(med[0])))
    df = pd.DataFrame(rows)
    path = outd / "surface_density_table.csv"
    df.to_csv(path, index=False, float_format="%.3f")
    print(f"Table saved -> {path}")
    return df


def save_eq1_components_combined(fa, outd, density_result=None, density_mode="fixed"):
    """Save Eq.1 term breakdown. Includes density-gradient diagnostics when density_result given."""
    rows = []
    dr = density_result or {}
    hZ_used = TA["hZ"]
    hR_used = TA["hR"]
    hs_used = TA["hs"]
    use_general = (density_mode in ("fitted_global", "fitted_local")
                   and dr.get("local_grad") is not None)

    for _, row in fa.iterrows():
        for zv in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            if use_general:
                lg = dr["local_grad"]
                dlnnu_dz, dlnnu_dR = lg(row.rm, zv)
                c = jeans_comps_general_density(
                    np.array([zv]), row.rm, row.a, row.b, row.m, hs_used,
                    dlnnu_dz, dlnnu_dR,
                    rz_b=_row_get(row, "rz_b", 0.0), **_row_sigz_params(row))
                dlnnu_dz_out = float(c["dlnnu_dz"][0])
                dlnnu_dR_out = float(c["dlnnu_dR"])
                hZ_loc = float(c["hZ_eff_local"][0]) if np.isfinite(c["hZ_eff_local"][0]) else hZ_used
                hR_loc = float(c["hR_eff_local"]) if np.isfinite(c["hR_eff_local"]) else hR_used
            else:
                c = jeans_comps(np.array([zv]), row.rm, row.a, row.b, row.m,
                                hZ_used, hR_used, hs_used,
                                rz_b=_row_get(row, "rz_b", 0.0), **_row_sigz_params(row))
                dlnnu_dz_out = -1.0 / hZ_used
                dlnnu_dR_out = -1.0 / hR_used
                hZ_loc = hZ_used
                hR_loc = hR_used

            rows.append(dict(R=row.rm, Z=zv,
                             sigma_Z_model=str(_row_get(row, "sigz_model", "linear")),
                             density_mode=density_mode,
                             Sigma=c["Sig"][0],
                             density_term=c["den_term"][0],
                             gradient_term=c["grad_term"][0],
                             tilt_term=c["tilt_term"][0],
                             sigma_Z=c["sz"][0],
                             sigma_RZ2=c["srz2"][0],
                             tilt_coefficient=float(c["tc"]) if np.ndim(c["tc"]) == 0 else float(c["tc"][0]),
                             dlnnu_dZ=dlnnu_dz_out,
                             dlnnu_dR=dlnnu_dR_out,
                             hZ_eff_used=hZ_loc,
                             hR_eff_used=hR_loc,
                             hs_used=hs_used))
    df = pd.DataFrame(rows)
    df.to_csv(outd / "eq1_components_all.csv", index=False, float_format="%.6f")
    return df


def write_consistency_summary_combined(fa, res, outd):
    lines = ["MATHEMATICAL CONSISTENCY SUMMARY", "", "1. Tilt slope sign:"]
    pos = int((fa.m > 0).sum()) if not fa.empty else 0
    neg = int((fa.m < 0).sum()) if not fa.empty else 0
    lines.append("   sigma_RZ^2 is fitted as an odd function with zero intercept.")
    lines.append(f"   All stars: {pos}/{len(fa)} positive slopes, {neg}/{len(fa)} negative slopes")
    lines.append("")
    lines.append("2. K&G:")
    # res may be a single dict (all stars) or a tuple/list (thin, thick)
    if isinstance(res, (tuple, list)):
        res_n, res_k = res[0], res[1]
        if res_n:
            lines.append(f"   Sigma_disk_thin={res_n['Sd']:.2f} Msun/pc^2, rho_dm_thin={res_n['rdm']:.4f}")
        if res_k:
            lines.append(f"   Sigma_disk_thick={res_k['Sd']:.2f} Msun/pc^2, rho_dm_thick={res_k['rdm']:.4f}")
    elif res:
        lines.append(f"   Sigma_disk_all={res['Sd']:.2f} Msun/pc^2, rho_dm_all={res['rdm']:.4f}")
    (outd / "mathematical_consistency_summary.txt").write_text("\n".join(lines) + "\n")


def plot_appendix_sigrz_combined(pa, fa, outd, overlays=None):
    """Paper-style appendix figure (Fig A1/A2): per-R-bin panels with
    σ_Z vs Z (upper) and σ²_RZ vs Z (lower), OLS linear fit for σ_RZ,
    and chemical thin/thick overlay populations as coloured markers.
    Unweighted fits to go through the bulk of the data."""
    overlays = overlays or []
    bins = sorted(fa.rl.unique())
    if not bins:
        return
    nc = 4
    nr = int(np.ceil(len(bins) / nc))
    fig, ax2d = plt.subplots(nr * 2, nc,
                             figsize=(nc * 3.8, nr * 5.2), squeeze=False)
    zln = np.linspace(-TA["zmax"], TA["zmax"], 400)

    ov_colors = {"mgfe_feh_thin_chem": "#2166ac", "mgfe_feh_thick_chem": "#b2182b",
                 "alphafe_feh_thin_chem": "#f4a582", "alphafe_feh_thick_chem": "#053061"}
    ov_markers = {"mgfe_feh_thin_chem": "^", "mgfe_feh_thick_chem": "v",
                  "alphafe_feh_thin_chem": "s", "alphafe_feh_thick_chem": "D"}

    for i, rl in enumerate(bins):
        row, col = divmod(i, nc)
        ax_s = ax2d[row * 2, col]
        ax_r = ax2d[row * 2 + 1, col]

        sub = pa[(pa.rl == rl) & (np.abs(pa.zm) <= TA["zmax"])]
        fit = fa[fa.rl == rl]
        if sub.empty or fit.empty:
            ax_s.axis("off"); ax_r.axis("off"); continue
        f = fit.iloc[0]
        rh = float(f.rh) if "rh" in f.index else rl + 1

        # --- σ_Z panel ---
        model = str(_row_get(f, "sigz_model", "linear"))
        sz_fit = sigz_model_eval(np.abs(zln), model,
                                 dict(a=f.a, b=f.b,
                                      q=_row_get(f, "sigz_q", np.nan),
                                      s0sq=_row_get(f, "sigz_s0sq", np.nan),
                                      k=_row_get(f, "sigz_k", np.nan),
                                      L=_row_get(f, "sigz_L", np.nan),
                                      s0=_row_get(f, "sigz_s0", np.nan)))[0]
        ax_s.errorbar(sub.zm, sub.sZ, yerr=sub.sZe, fmt="o",
                      ms=2.0, capsize=1.0, color="#4dac26", lw=0.5, alpha=0.55, zorder=1)
        ax_s.plot(zln, sz_fit, lw=1.8, color="k", zorder=3)

        # --- σ²_RZ panel: plain OLS fit without error weighting ---
        rz_m_ols, rz_b_ols, _ = wls(sub.zm.values, sub.crz.values)
        ax_r.errorbar(sub.zm, sub.crz, yerr=sub.crze, fmt="o",
                      ms=2.0, capsize=1.0, color="#4dac26", lw=0.5, alpha=0.55, zorder=1)
        ax_r.plot(zln, rz_b_ols + rz_m_ols * zln, lw=1.8, color="k", zorder=3,
                  label=(rf"$\sigma_{{RZ}}=({rz_b_ols:.1f})+({rz_m_ols:.1f})Z$"))
        ax_r.axhline(0, color="0.65", lw=0.7)
        ax_r.axvline(0, color="0.65", lw=0.7)

        # --- overlay thin/thick chemical subpopulations ---
        ctr = 0.5 * (rl + rh)
        tol = max(0.51, 0.5 * (rh - rl) + 1e-6)
        for ov in overlays:
            pf = ov.get("profile")
            if pf is None:
                continue
            key = ov.get("key", "")
            col_ov = ov_colors.get(key, ov["color"])
            mk_ov  = ov_markers.get(key, ov["marker"])
            osub = pf[np.abs(pf.rm - ctr) <= tol]
            if osub.empty:
                continue
            osub_z = osub[np.abs(osub.zm) <= TA["zmax"]]
            lbl = ov["label"] if i == 0 else None
            ax_s.errorbar(osub_z.zm, osub_z.sZ, yerr=osub_z.sZe,
                          fmt=mk_ov, ms=3.5, capsize=0, lw=0.0,
                          color=col_ov, alpha=0.75, label=lbl, zorder=2)
            ax_r.errorbar(osub_z.zm, osub_z.crz, yerr=osub_z.crze,
                          fmt=mk_ov, ms=3.5, capsize=0, lw=0.0,
                          color=col_ov, alpha=0.75, zorder=2)

        ax_s.axvline(0, color="0.65", lw=0.7)
        ax_s.set_title(f"({chr(97+i)}) {rl:g}$<R<${rh:g} kpc", fontsize=8)
        ax_s.tick_params(labelsize=7, direction="in", top=True, right=True)
        ax_r.tick_params(labelsize=7, direction="in", top=True, right=True)
        ax_r.legend(fontsize=5.5, frameon=False)
        if col == 0:
            ax_s.set_ylabel(r"$\sigma_Z$ [km s$^{-1}$]", fontsize=8)
            ax_r.set_ylabel(r"$\sigma^2_{RZ}$ [km$^2$ s$^{-2}$]", fontsize=8)
        if row == nr - 1:
            ax_r.set_xlabel(r"$Z$ [kpc]", fontsize=8)

    for j in range(len(bins), nr * nc):
        ax2d[divmod(j, nc)[0] * 2, divmod(j, nc)[1]].axis("off")
        ax2d[divmod(j, nc)[0] * 2 + 1, divmod(j, nc)[1]].axis("off")

    if overlays and i == 0:
        handles, labels = ax2d[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax2d[0, 0].legend(by_label.values(), by_label.keys(),
                              frameon=False, fontsize=6, loc="upper right")

    fig.suptitle("Appendix: vertical & tilt velocity dispersions per R bin "
                 "(black = all-star fits; coloured = chemical sub-populations)",
                 fontsize=9, y=1.002)
    fig.tight_layout()
    fig.savefig(outd / "figA_dispersion_per_bin.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# KG89 INTEGRATION — equations 22–56
# All functions here call kg1989_complete (imported as kg at the top).
# No mock/synthetic data.  Every path requires the real NPZ catalogue.
# ===========================================================================

# ---------------------------------------------------------------------------
# Validation figure: KG89 Fig 2  —  Kz → Σ conversion for double-exp disc
# (eq 27 exact Hankel-transform force vs eq 28 analytic Σ)
# ---------------------------------------------------------------------------

def plot_kg89_fig2_kz_sigma(outd, R=7.8, hR=4.5, z0=0.30, rho_d=1.0):
    """Reproduce KG89 Fig 2 using eq 27 (exact Kz) and eq 28 (exact Σ).

    Parameters
    ----------
    R, hR, z0 : geometry of the double-exponential disc [kpc]
    rho_d     : midplane density [M_sun/pc³]; used as unit — result is ratio

    Saves: kg_fig2_Kz_to_Sigma.png, kg_fig2_frac_deviation.png
    Returns max fractional |Kz/(2πG) − Σ| / Σ over z = 0–4 kpc.
    """
    z = np.linspace(1e-3, 4.0, 200)

    Sigma_exact = kg.dexp_sigma_eq28(R, z, rho_d, hR, z0)           # eq 28, M_sun/pc²
    Kz_exact    = np.array([abs(kg.dexp_Kz_eq27(R, zi, rho_d, hR, z0))
                             for zi in z])                            # eq 27, (km/s)²/kpc
    # |Kz|/(2πG) in M_sun/pc²
    Kz_over_TG = Kz_exact / TG / 1e6

    # Fractional deviation  Δ = (|Kz|/(2πG) − Σ) / Σ
    frac = (Kz_over_TG - Sigma_exact) / np.where(Sigma_exact > 0, Sigma_exact, 1e-30)

    # ── Panel 1: Σ and |Kz|/(2πG) vs z ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(z * 1000, Sigma_exact, "k-",  lw=2.0,
            label=r"$\Sigma(z)$ eq 28 (exact)")
    ax.plot(z * 1000, Kz_over_TG,  "k--", lw=1.6,
            label=r"$|K_z|/(2\pi G)$ eq 27 (Hankel)")
    ax.set_xlabel("z  [pc]", fontsize=12)
    ax.set_ylabel(r"$\Sigma,\ |K_z|/(2\pi G)$  [arb. M$_\odot$ pc$^{-2}$]",
                  fontsize=11)
    ax.set_title(rf"KG89 Fig 2 — double-exp disc "
                 rf"($h_R={hR}$, $z_0={z0*1000:.0f}$ pc, $R={R}$ kpc)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(direction="in", top=True, right=True)

    # ── Panel 2: fractional deviation ───────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(z * 1000, frac * 100.0, "b-", lw=2.0)
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("z  [pc]", fontsize=12)
    ax2.set_ylabel(r"$\Delta\Sigma/\Sigma$ = $(|K_z|/(2\pi G) - \Sigma)/\Sigma$  [%]",
                   fontsize=11)
    ax2.set_title("Fractional Oort-correction need (eq 24/36)", fontsize=11)
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(outd / "kg_fig2_Kz_to_Sigma.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG Fig 2 saved → {outd / 'kg_fig2_Kz_to_Sigma.png'}")
    return float(np.nanmax(np.abs(frac)))


# ---------------------------------------------------------------------------
# Validation figure: KG89 Fig 7  —  tilt function T(R, z)  (eq 48)
# ---------------------------------------------------------------------------

def plot_kg89_fig7_tilt(outd, R0_plot=7.8, alpha=2.0, hR=4.5):
    """Reproduce KG89 Fig 7: T(R,z) tilt function at R = R0 (eq 48).

    Also shows how T varies with α (velocity-ellipsoid axis ratio).

    Saves: kg_fig7_T_tilt.png
    Returns min T at R0 (paper: should be negative for plausible params).
    """
    z = np.linspace(1e-3, 8.0, 500)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Panel 1: T(R0, z) for KG89 canonical parameters ────────────────────
    T_kg = kg.T_tilt_eq48(R0_plot, z, alpha=alpha, hR=hR)
    ax   = axes[0]
    ax.plot(z, T_kg, "k-", lw=2.0,
            label=rf"$\alpha={alpha}$, $h_R={hR}$ kpc")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("z  [kpc]", fontsize=12)
    ax.set_ylabel(r"$T(R_0,z)$  [kpc$^{-1}$]", fontsize=12)
    ax.set_title(rf"KG89 Fig 7 — tilt function "
                 rf"$T(R_0={R0_plot}\ \mathrm{{kpc}},z)$ (eq 48)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(direction="in", top=True, right=True)

    # ── Panel 2: sensitivity to α ────────────────────────────────────────────
    ax2 = axes[1]
    for al, ls in zip([1.5, 2.0, 2.5, 3.0], ["-", "--", "-.", ":"]):
        T_a = kg.T_tilt_eq48(R0_plot, z, alpha=al, hR=hR)
        ax2.plot(z, T_a, lw=1.8, ls=ls, label=rf"$\alpha={al}$")
    ax2.axhline(0, color="0.6", lw=0.8)
    ax2.set_xlabel("z  [kpc]", fontsize=12)
    ax2.set_ylabel(r"$T(R_0,z)$  [kpc$^{-1}$]", fontsize=12)
    ax2.set_title(r"Sensitivity to velocity-ellipsoid axis ratio $\alpha$",
                  fontsize=11)
    ax2.legend(frameon=False, fontsize=10)
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fig.savefig(outd / "kg_fig7_T_tilt.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG Fig 7 saved → {outd / 'kg_fig7_T_tilt.png'}")
    return float(np.nanmin(T_kg))


# ---------------------------------------------------------------------------
# Full KG89 ML-DF pipeline  —  Figure 8 flowchart, rigorous implementation
# ---------------------------------------------------------------------------

def run_kg89_ml_pipeline(data_dict, outd, tag="",
                          K_lo=20.0, K_hi=100.0, nK=25,
                          D=0.4, alpha=2.0, hR_tilt=None,
                          sigma_lnz=0.15, sigma_v=4.0,
                          R_target=8.5, R_width=0.5,
                          nz_grid=251, nz_ll=31, nv_ll=31,
                          z_min_ll=0.05, z_max_ll=4.0):
    """Full KG89 maximum-likelihood DF pipeline following Fig 8 flowchart.

    Flowchart steps implemented (exactly as described in the paper):

      1. Measure ν(z)         — fit exponential to |Z| histogram from real data
      2. Adopt model potentials — eq 39/40, K-grid, F from eq 43
      3. σ_Rz tilt in Jeans   — predict σ_zz(z), K_z,eff(z) via eq 50, 52
      4. Abel inversion        — ν(ψ_eff) → f_z(E_{z,eff}) via eq 21
      5. Observe f^obs(v_z|z)  — real (z, v_z) pairs from data_dict
      6. Error convolution     — F^mod(v_z;z) via eq 56–57
      7. Max likelihood        — eq 53–55 + rotation-curve constraint eq 43
      8. Best ψ(z), K_z(z)    — from best-fit K
      9. Σ(z) from eq 24       — with Oort (A²−B²) correction

    Parameters
    ----------
    data_dict : dict with arrays R, Z, vZ (and vp/vphi for Oort constants)
    outd      : Path — directory for output files
    tag       : string label for filenames / plot titles
    K_lo, K_hi, nK : grid range for trial disc surface density [M_sun/pc²]
    D         : disc equivalent scaleheight for eq 40 [kpc]
    alpha     : velocity-ellipsoid axis ratio for tilt eq 50/52
    hR_tilt   : radial scalelength for tilt eq 48/51 [kpc]; defaults to TA['hR']
    sigma_lnz : fractional distance error (log-normal σ)
    sigma_v   : velocity-measurement error [km/s]
    R_target, R_width : solar-annulus definition [kpc]
    nz_grid   : points in z grid for potential + DF build
    nv_grid   : points in v_z grid for f_z evaluation
    nz_ll, nv_ll : quadrature points for likelihood integration
    z_min_ll, z_max_ll : z range for likelihood integration [kpc]

    Returns dict with keys:
      K_best, F_best, K_grid, logL,
      z, Kz, Sigma, Sigma_oort,
      psi, Kzeff,
      nu_func, hZ_tracer, oort,
      Ez_grid, fz_grid
    """
    outd = Path(outd)
    if hR_tilt is None:
        hR_tilt = TA["hR"]

    R_arr  = np.asarray(data_dict["R"],  float)
    Z_arr  = np.asarray(data_dict["Z"],  float)
    vZ_arr = np.asarray(data_dict["vZ"], float)
    vp_key = "vp" if "vp" in data_dict else "vphi"
    vp_arr = np.asarray(data_dict[vp_key], float) if vp_key in data_dict else None

    # ── Step 1: Measure ν(z) from data  ─────────────────────────────────────
    print(f"  [KG89 Fig8 step 1] Measuring ν(z) from solar annulus "
          f"R={R_target}±{R_width} kpc ...")
    nu_func, hZ_tracer = kg.fit_nu_from_data(
        Z_arr, R_arr,
        R_target=R_target, R_width=R_width,
        z_min=0.1, z_max=4.0, n_bins=30)
    print(f"    fitted hZ_tracer = {hZ_tracer:.3f} kpc")

    # Select stars in solar annulus for the likelihood
    sel = (np.abs(R_arr - R_target) < R_width) & np.isfinite(Z_arr) & np.isfinite(vZ_arr)
    z_data  = np.abs(Z_arr[sel])
    vz_data = vZ_arr[sel]
    print(f"    {sel.sum():,} stars in solar annulus for ML")

    # ── Oort constants from v_phi data  ──────────────────────────────────────
    if vp_arr is not None:
        oort = kg.oort_constants_from_data(R_arr, vp_arr, R0_loc=R_target)
    else:
        oort = dict(A=14.82, B=-12.37, Vc=220.0, dVc_dR=0.0, A2mB2=0.0,
                    note="no vphi array: IAU 1985 values used")
    print(f"    Oort A={oort['A']:.2f}, B={oort['B']:.2f} km/s/kpc, "
          f"A²−B²={oort['A2mB2']:.3f} (km/s/kpc)²")
    A2mB2 = oort["A2mB2"]

    # ── Step 2: Adopt model potentials  ──────────────────────────────────────
    K_grid  = np.linspace(K_lo, K_hi, nK)
    z_grid  = np.linspace(1e-3, z_max_ll * 1.1, nz_grid)
    nu_grid = nu_func(z_grid)

    print(f"  [KG89 Fig8 step 2] Scanning {nK} trial disc densities "
          f"K = {K_lo:.0f}–{K_hi:.0f} M_sun/pc² ...")

    logL      = np.full(nK, -np.inf)
    best_data = {}

    for i, K in enumerate(K_grid):
        # Rotation-curve constraint (eq 43): F = 0.041 − 0.0094 × K/36.7
        F_kg, _ = kg.rotation_curve_constraint_eq43(K / KG_SIGMA_UNIT)
        F_phys  = max(float(F_kg), 1e-5) * KG_RHO_UNIT   # M_sun/pc³

        # ── Step 3: σ_zz(z) and K_z,eff(z) with tilt  (eq 50, 52) ──────────
        Kz_trial  = kg.kg_Kz_eq39(z_grid, K, D, F_phys)             # eq 39
        Kzeff_arr = kg.Kz_eff_eq52(z_grid, R_target, nu_grid,
                                    Kz_trial, alpha=alpha, hR=hR_tilt)   # eq 52
        # Effective potential from integrating K_z,eff  (used in Abel inversion)
        psi_eff   = kg.effective_psi_from_Kzeff(z_grid, Kzeff_arr)  # ∫K_z,eff dz

        # ── Step 4: Abel inversion ν(ψ_eff) → f_z(E_{z,eff})  (eq 21) ──────
        order  = np.argsort(psi_eff)
        psi_s  = psi_eff[order]
        nu_s   = nu_grid[order]
        # Keep only strictly increasing ψ
        uniq   = np.concatenate(([True], np.diff(psi_s) > 1e-10))
        psi_s, nu_s = psi_s[uniq], nu_s[uniq]
        if len(psi_s) < 10:
            continue

        Ez_grid_i = np.linspace(0.0, psi_s[-1] * 0.97, 160)
        fz_grid_i = kg.abel_invert_df_eq21(Ez_grid_i, psi_s, nu_s)  # eq 21
        fz_grid_i = np.clip(fz_grid_i, 0.0, None)

        # ── Steps 5–7: observe data, convolve, maximise likelihood  ──────────
        # Build a fast psi function for the likelihood call
        psi_eff_interp = interp1d(z_grid, psi_eff, kind="linear",
                                   bounds_error=False,
                                   fill_value=(psi_eff[0], psi_eff[-1]))

        ll = kg.log_likelihood_eq53_55(
            z_data, vz_data,
            Ez_grid_i, fz_grid_i,
            psi_eff_interp,
            sigma_lnz=sigma_lnz, sigma_v=sigma_v,
            nz=nz_ll, nv=nv_ll,
            z_int=(z_min_ll, z_max_ll))
        logL[i] = ll

        if ll > max(logL[:i], default=-np.inf) or i == 0:
            best_data = dict(K=K, F_phys=F_phys, F_kg=F_kg,
                             psi_eff=psi_eff, Kz=Kz_trial,
                             Kzeff=Kzeff_arr,
                             Ez_grid=Ez_grid_i, fz_grid=fz_grid_i,
                             psi_func=psi_eff_interp)

        if (i + 1) % 5 == 0:
            print(f"    K={K:.1f}: log L = {ll:.1f}  (best so far: "
                  f"{logL[:i+1][np.isfinite(logL[:i+1])].max():.1f})")

    # ── Step 8: Best parametric description of ψ(z), K_z(z)  ────────────────
    finite_mask = np.isfinite(logL)
    if not finite_mask.any():
        raise RuntimeError(
            "KG89 ML pipeline: no finite log-likelihood for any trial K. "
            "Check that the data arrays contain real stars and that "
            "sigma_lnz / sigma_v are non-zero.")

    best_idx  = int(np.nanargmax(logL))
    K_best    = float(K_grid[best_idx])
    F_kg_best, _ = kg.rotation_curve_constraint_eq43(K_best / KG_SIGMA_UNIT)
    F_best    = max(float(F_kg_best), 1e-5) * KG_RHO_UNIT

    print(f"  [KG89 Fig8 step 8] Best K = {K_best:.2f} M_sun/pc², "
          f"F = {F_best:.5f} M_sun/pc³")

    z_plot   = np.linspace(0.02, z_max_ll, 300)
    Kz_best  = kg.kg_Kz_eq39(z_plot, K_best, D, F_best)
    psi_best = kg.psi_disc_halo_eq40(z_plot, K_best, D, F_best)

    # Recompute full K_z,eff and σ_zz at the best K
    nu_plot   = nu_func(z_plot)
    Kzeff_best = kg.Kz_eff_eq52(z_plot, R_target, nu_plot,
                                  Kz_best, alpha=alpha, hR=hR_tilt)

    # ── Step 9: Σ(z) from eq 24 with Oort correction  ────────────────────────
    Sigma_raw  = kg.surface_density_eq24(Kz_best, z_plot, A2mB2=0.0)
    Sigma_oort = kg.surface_density_eq24(Kz_best, z_plot, A2mB2=A2mB2)
    print(f"  [KG89 Fig8 step 9] Σ(1 kpc) = {np.interp(1.0, z_plot, Sigma_oort):.1f} "
          f"M_sun/pc² (Oort-corrected)")

    result = dict(
        K_best=K_best, F_best=F_best, K_grid=K_grid, logL=logL,
        z=z_plot, Kz=Kz_best, psi=psi_best, Kzeff=Kzeff_best,
        Sigma=Sigma_raw, Sigma_oort=Sigma_oort,
        nu_func=nu_func, hZ_tracer=hZ_tracer, oort=oort, A2mB2=A2mB2,
        Ez_grid=best_data.get("Ez_grid"), fz_grid=best_data.get("fz_grid"),
        alpha=alpha, D=D, R_target=R_target, tag=tag)

    # Save CSVs
    import pandas as pd
    pd.DataFrame(dict(K=K_grid, logL=logL)).to_csv(
        outd / f"kg89_ml_logL_{tag}.csv".replace(" ", "_"), index=False)
    pd.DataFrame(dict(z=z_plot, Kz=Kz_best, psi=psi_best,
                      Kzeff=Kzeff_best,
                      Sigma=Sigma_raw, Sigma_oort=Sigma_oort)).to_csv(
        outd / f"kg89_ml_best_model_{tag}.csv".replace(" ", "_"), index=False)
    print(f"    CSVs saved to {outd}")

    return result


def plot_kg89_ml_summary(res, outd):
    """Three-panel summary figure for the KG89 ML pipeline result.

    Panel (a): log L vs trial K (eq 55)
    Panel (b): Best Kz(z) from eq 39 and K_z,eff(z) from eq 52
    Panel (c): Σ(z) from eq 24, plain and Oort-corrected

    Parameters
    ----------
    res  : dict returned by run_kg89_ml_pipeline
    outd : output directory Path
    """
    tag   = res.get("tag", "")
    z     = res["z"]
    K_grid = res["K_grid"]
    logL  = res["logL"]
    K_best = res["K_best"]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # (a) Likelihood
    ax = axes[0]
    finite = np.isfinite(logL)
    ax.plot(K_grid[finite], logL[finite] - logL[finite].max(),
            "o-", color="#2166ac", ms=5, lw=1.8)
    ax.axvline(K_best, color="firebrick", ls="--", lw=1.5,
               label=fr"$K_\mathrm{{best}}={K_best:.1f}\ M_\odot\,\mathrm{{pc}}^{{-2}}$")
    ax.set_xlabel(r"Trial $K$ [M$_\odot$ pc$^{-2}$]", fontsize=12)
    ax.set_ylabel(r"$\log L - \log L_{\max}$  (eq 55)", fontsize=12)
    ax.set_title("(a) ML over disc-halo potential grid\n"
                 r"+ rotation-curve constraint (eq 43)", fontsize=10)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(bottom=min(-60, float(np.nanmin(logL[finite] - logL[finite].max())) * 1.1))
    ax.tick_params(direction="in", top=True, right=True)

    # (b) Kz and K_z,eff
    ax2 = axes[1]
    ax2.plot(z, -res["Kz"]   / TG / 1e6, "k-",  lw=2.0,
             label=r"$|K_z|/(2\pi G)$ eq 39")
    ax2.plot(z, -res["Kzeff"] / TG / 1e6, "b--", lw=1.6,
             label=r"$|K_{z,\mathrm{eff}}|/(2\pi G)$ eq 52 (tilt)")
    ax2.set_xlabel("z  [kpc]", fontsize=12)
    ax2.set_ylabel(r"$|K_z|/(2\pi G)$  [M$_\odot$ pc$^{-2}$]", fontsize=12)
    ax2.set_title(r"(b) Best-fit $K_z$ and $K_{z,\mathrm{eff}}$" + "\n(eq 39, 52)",
                  fontsize=10)
    ax2.legend(frameon=False, fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(direction="in", top=True, right=True)

    # (c) Σ(z)
    ax3 = axes[2]
    ax3.plot(z, res["Sigma"],      "k-",  lw=2.0, label=r"$\Sigma$ eq 24")
    ax3.plot(z, res["Sigma_oort"], "r--", lw=1.6,
             label=(r"$\Sigma$ eq 24 + Oort corr."
                    + f"\n$A^2-B^2={res['A2mB2']:.3f}$"))
    ax3.set_xlabel("z  [kpc]", fontsize=12)
    ax3.set_ylabel(r"$\Sigma(R_\odot,z)$  [M$_\odot$ pc$^{-2}$]", fontsize=12)
    ax3.set_title("(c) Integral surface density (eq 24)\n"
                  "with Oort-constant correction", fontsize=10)
    ax3.legend(frameon=False, fontsize=10)
    ax3.set_ylim(bottom=0)
    ax3.tick_params(direction="in", top=True, right=True)

    A_val = res["oort"].get("A", np.nan)
    B_val = res["oort"].get("B", np.nan)
    fig.suptitle(
        f"KG89 ML-DF pipeline — {tag}\n"
        rf"$K_\mathrm{{best}}={K_best:.1f}\ M_\odot\,\mathrm{{pc}}^{{-2}}$, "
        rf"$\alpha={res['alpha']:.1f}$, $D={res['D']:.2f}$ kpc   "
        rf"$|$ Oort: $A={A_val:.2f}$, $B={B_val:.2f}$ km/s/kpc",
        fontsize=10, y=1.02)
    fig.tight_layout()
    fname = f"kg89_ml_pipeline_summary_{tag}.png".replace(" ", "_")
    fig.savefig(outd / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG89 ML summary → {outd / fname}")


def plot_kg89_sigma_comparison(res, outd, fa=None, pp=None):
    """Compare KG89 ML Σ(z) with differential-Jeans Σ(z) (if fa is given).

    Saves: kg89_sigma_comparison_{tag}.png
    """
    tag  = res.get("tag", "")
    z    = res["z"]
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.fill_between(z, res["Sigma_oort"], res["Sigma"],
                    alpha=0.18, color="#2166ac",
                    label="Oort correction band (eq 24)")
    ax.plot(z, res["Sigma_oort"], "b-",  lw=2.2,
            label=fr"KG89 ML Σ (eq 24, Oort-corr.)")
    ax.plot(z, res["Sigma"],      "b--", lw=1.4,
            label="KG89 ML Σ (eq 24, no corr.)")

    if fa is not None and pp is not None:
        z_jeans = np.linspace(0.1, z[-1], 120)
        for _, row in fa.iterrows():
            if not np.isclose(row.rm, res["R_target"], atol=0.6):
                continue
            S_j = sig_from_row(row, pp, z_jeans)[0]
            ax.plot(z_jeans, S_j, "k-", lw=1.8, alpha=0.75,
                    label=f"Jeans Σ (eq 1, R={row.rm:.1f} kpc)")
            break

    ax.set_xlabel("z  [kpc]", fontsize=12)
    ax.set_ylabel(r"$\Sigma(R_\odot,z)$  [M$_\odot$ pc$^{-2}$]", fontsize=12)
    ax.set_title(f"KG89 ML vs Jeans Σ — {tag}", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)
    fig.tight_layout()
    fname = f"kg89_sigma_comparison_{tag}.png".replace(" ", "_")
    fig.savefig(outd / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG89 Σ comparison → {outd / fname}")


def plot_kg89_oort_rotcurve(oort, K_best, outd, tag=""):
    """Plot the Oort-constant rotation curve and the eq 43 F(K) constraint.

    Saves: kg89_oort_rotcurve_{tag}.png
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Rotation curve from Oort constants
    ax = axes[0]
    if "r_cen" in oort and "vc_med" in oort:
        ax.plot(oort["r_cen"], oort["vc_med"], "o", color="#2166ac",
                ms=5, label="median $V_c(R)$ from data")
    A, B = oort.get("A", np.nan), oort.get("B", np.nan)
    Vc   = oort.get("Vc", 220.0)
    ax.axhline(Vc, color="k", ls="--", lw=1.2,
               label=fr"$V_c={Vc:.0f}$ km/s at $R_0$")
    ax.set_xlabel("R  [kpc]", fontsize=12)
    ax.set_ylabel(r"$V_c$ [km/s]", fontsize=12)
    ax.set_title(
        fr"Oort constants: $A={A:.2f}$, $B={B:.2f}$ km/s/kpc",
        fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    ax.tick_params(direction="in", top=True, right=True)

    # (b) KG89 eq 43 rotation-curve constraint line
    ax2 = axes[1]
    K_arr   = np.linspace(10, 120, 200)
    F_arr   = np.array([max(kg.rotation_curve_constraint_eq43(K / KG_SIGMA_UNIT)[0], 0.0)
                        * KG_RHO_UNIT for K in K_arr])
    ax2.plot(K_arr, F_arr * 1000, "k-", lw=2.0,
             label="eq 43: $F = (0.041 - 0.0094K') \\times 0.367$")
    ax2.axvline(K_best, color="firebrick", ls="--", lw=1.5,
                label=fr"$K_\mathrm{{best}}={K_best:.1f}$")
    ax2.set_xlabel(r"$K$ [M$_\odot$ pc$^{-2}$]", fontsize=12)
    ax2.set_ylabel(r"$F$ [M$_\odot$ pc$^{-3}$] $\times 10^3$", fontsize=12)
    ax2.set_title("Rotation-curve constraint (KG89 eq 43)", fontsize=11)
    ax2.legend(frameon=False, fontsize=10)
    ax2.tick_params(direction="in", top=True, right=True)

    fig.tight_layout()
    fname = f"kg89_oort_rotcurve_{tag}.png".replace(" ", "_")
    fig.savefig(outd / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG89 Oort+rotcurve → {outd / fname}")


# ---------------------------------------------------------------------------
# Extended KG89 diagnostics — equations not previously used in pipeline
# ---------------------------------------------------------------------------

def _load_kg89_ml_result(outd, tag):
    """Load K_best, F_best, D=0.4 (default), and best-model DataFrame from CSVs."""
    ll_path = outd / f"kg89_ml_logL_{tag}.csv".replace(" ", "_")
    bm_path = outd / f"kg89_ml_best_model_{tag}.csv".replace(" ", "_")
    if not ll_path.exists() or not bm_path.exists():
        raise FileNotFoundError(
            f"KG89 ML CSVs not found for tag='{tag}' in {outd}. "
            "Run --kg89-only first to generate them.")
    ll = pd.read_csv(ll_path)
    K_best = float(ll.loc[ll["logL"].idxmax(), "K"])
    F_kg, _ = kg.rotation_curve_constraint_eq43(K_best / KG_SIGMA_UNIT)
    F_best   = max(float(F_kg), 1e-5) * KG_RHO_UNIT
    bm = pd.read_csv(bm_path)
    return K_best, F_best, bm


def _get_solar_sigma2_profile(outd, tag, n_zbins=18):
    """Load and bin σ²_z(z) at the solar ring for a given population tag.

    Returns (z_bin, sig2_bin, esig2_bin) sorted ascending in |z|,
    or (None, None, None) if data are missing.
    """
    vd_map = {
        "all_stars":       outd / "profile_all.csv",
        "chem_thin":       outd / "chem_combined_thin_velocity_dispersion_profiles.csv",
        "chem_thick":      outd / "chem_combined_thick_velocity_dispersion_profiles.csv",
        "mgfe_thick_only": outd / "chem_mgfe_thick_only_velocity_dispersion_profiles.csv",
    }
    vd_path = vd_map.get(tag, outd / "profile_all.csv")
    if not vd_path.exists():
        return None, None, None
    vd = pd.read_csv(vd_path)

    if "sigma_Z" in vd.columns:   # chem VD format
        solar = vd[(vd["R_mid"] >= 7.75) & (vd["R_mid"] <= 9.25)].copy()
        solar["absz"] = solar["Z_median"].abs()
        solar = solar[(solar["absz"] > 0.05)
                      & np.isfinite(solar["sigma_Z"]) & (solar["sigma_Z"] > 0)]
        z_raw  = solar["absz"].values
        s2_raw = solar["sigma_Z"].values ** 2
        e2_raw = 2.0 * solar["sigma_Z"].values * solar["sigma_Z_err"].values
    elif "sZ" in vd.columns:      # profile_all format
        solar = vd[(vd["rm"] >= 8.0) & (vd["rm"] <= 9.0)].copy()
        solar["absz"] = solar["zm"].abs()
        solar = solar[(solar["absz"] > 0.05)
                      & np.isfinite(solar["sZ"]) & (solar["sZ"] > 0)]
        z_raw  = solar["absz"].values
        s2_raw = solar["sZ"].values ** 2
        e2_raw = 2.0 * solar["sZ"].values * solar["sZe"].values
    else:
        return None, None, None

    if len(z_raw) < 5:
        return None, None, None

    zmax  = min(float(z_raw.max()), 3.5)
    edges = np.linspace(0.05, zmax, n_zbins + 1)
    z_b, s2_b, e2_b = [], [], []
    for i in range(n_zbins):
        m = (z_raw >= edges[i]) & (z_raw < edges[i + 1])
        if m.sum() < 3:
            continue
        z_b.append(float(np.median(z_raw[m])))
        s2_b.append(float(np.median(s2_raw[m])))
        e2_b.append(float(np.std(s2_raw[m]) / np.sqrt(m.sum())))
    if len(z_b) < 3:
        return None, None, None
    return np.array(z_b), np.array(s2_b), np.array(e2_b)


def plot_kg89_sigma2_exact(outd, tag, pp, D=0.4, alpha=2.0, R_target=8.5):
    """Fixed fig10: σ²_z(z) using EXACT KG89 eq 39 (K_z) + eq 50 (tilt Jeans).

    Compares three curves:
      • Data binned points (solar ring VD profile)
      • OLD simplified kg_sig2 formula (no tilt, no √(z²+D²) disc term)
      • NEW: no-tilt Jeans with eq 39 K_z (dotted)
      • NEW: full eq 39 + eq 50 with tilt integrating factor (solid)

    Saves: fig_kg89_sigma2_exact_{tag}.png
    """
    try:
        K_best, F_best, bm = _load_kg89_ml_result(outd, tag)
    except FileNotFoundError as e:
        print(f"  [sigma2_exact] {e}")
        return

    hZ = pp.get("hZ", TA["hZ"])
    hR = pp.get("hR", TA["hR"])
    z_bin, s2_bin, e2_bin = _get_solar_sigma2_profile(outd, tag)
    if z_bin is None:
        print(f"  [sigma2_exact] no VD profile data for tag={tag}")
        return

    z_fine  = np.linspace(0.02, min(float(z_bin.max()) * 1.1, 3.5), 300)
    nu_fine = np.exp(-z_fine / hZ)
    Kz_fine = kg.kg_Kz_eq39(z_fine, K_best, D, F_best)          # eq 39

    # Exact σ²_zz with tilt — eq 39 + eq 50
    s2_exact = kg.sigma_zz2_with_tilt_eq50(z_fine, R_target, nu_fine,
                                             Kz_fine, alpha=alpha, hR=hR)

    # No-tilt Jeans with eq 39 K_z (plain integral, no S factor)
    integrand = nu_fine * Kz_fine
    I_nt = np.zeros_like(z_fine)
    for i in range(len(z_fine) - 2, -1, -1):
        I_nt[i] = (I_nt[i + 1]
                   + 0.5 * (integrand[i] + integrand[i + 1]) * (z_fine[i + 1] - z_fine[i]))
    s2_notilt = -I_nt / np.where(nu_fine > 1e-30, nu_fine, 1e-30)

    # Old simplified kg_sig2 best-fit
    e_safe = e2_bin if (e2_bin > 0).any() else np.ones_like(s2_bin) * 100.0
    res_old = fit_kg(z_bin, s2_bin, e_safe, hZ)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.errorbar(z_bin, s2_bin / 1e3, yerr=e2_bin / 1e3,
                fmt="o", color="k", ms=5, capsize=3, lw=1.2, zorder=5,
                label="Data (binned, solar ring)")

    if res_old is not None:
        z_pl = np.linspace(0.02, min(float(z_bin.max()) * 1.1, 3.5), 300)
        ax.plot(z_pl, kg_sig2(z_pl, res_old["Sd"], res_old["rdm"], hZ) / 1e3,
                "--", color="#d73027", lw=2.0, alpha=0.85,
                label=(rf"Simplified kg\_sig2 (old): "
                       rf"$\Sigma_d={res_old['Sd']:.0f}$ M$_\odot$/pc²"))

    ax.plot(z_fine, s2_notilt / 1e3, ":", color="#4393c3", lw=2.0,
            label=f"Eq 39 K_z + no-tilt Jeans (K={K_best:.0f})")

    ax.plot(z_fine, s2_exact / 1e3,  "-", color="#1a6fba", lw=2.4,
            label=f"Eq 39+50 exact with tilt (K={K_best:.0f})")

    # Check whether K_best landed at the edge of the scan grid
    ll_path = outd / f"kg89_ml_logL_{tag}.csv".replace(" ", "_")
    _at_boundary = False
    _boundary_msg = ""
    if ll_path.exists():
        _ll = pd.read_csv(ll_path)
        _K_lo_used, _K_hi_used = float(_ll["K"].min()), float(_ll["K"].max())
        _step = (_K_hi_used - _K_lo_used) / max(len(_ll) - 1, 1)
        if K_best <= _K_lo_used + _step * 0.6:
            _at_boundary = True
            _boundary_msg = (f"⚠ K_best={K_best:.1f} at LOWER grid edge (K_lo={_K_lo_used:.1f})\n"
                             "True peak may be at even lower K — model may not suit\nthis mixed population")
        elif K_best >= _K_hi_used - _step * 0.6:
            _at_boundary = True
            _boundary_msg = (f"⚠ K_best={K_best:.1f} at UPPER grid edge (K_hi={_K_hi_used:.1f})\n"
                             f"Re-run with --kg89-K-hi {int(_K_hi_used*1.5)}")
    if _at_boundary:
        ax.text(0.97, 0.05, _boundary_msg,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color="firebrick",
                bbox=dict(facecolor="lightyellow", edgecolor="firebrick", alpha=0.7))

    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=13)
    ax.set_ylabel(r"$\sigma_z^2$ [$10^3$ (km/s)²]", fontsize=13)
    ax.set_title(
        f"KG89 exact $\\sigma_z^2(z)$ — {tag.replace('_', ' ')}\n"
        r"Solid blue: eq 39+50 (KG89 exact);  dotted: eq 39 no-tilt;  "
        r"dashed red: simplified kg\_sig2",
        fontsize=9.5)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlim(left=0)
    fig.tight_layout()

    fname = f"fig_kg89_sigma2_exact_{tag}.png".replace(" ", "_")
    fig.savefig(outd / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG89 exact σ²_z  → {outd / fname}")


def plot_kg89_complete_diagnostics(outd, tag, pp,
                                    D=0.4, alpha=2.0, R_target=8.5,
                                    A2mB2=0.0):
    """Six-panel diagnostic figure using KG89 equations not previously plotted.

    Panels (all using eq numbers not called in the original pipeline):
      A  K_z decomposition (eq 39 disc term, DM term, eq 52 tilt correction)
      B  Matter density ρ(z) from Poisson eq 23, isothermal eq 37, DM=F, eq 44
      C  Jeans self-consistency: K_z,eff model (eq 52) vs data-implied (eq 45)
      D  Velocity-ellipsoid tilt profile: T eq 48, S eq 51, σ²_Rz eq 46
      E  Oort correction: Σ with/without (eq 24), fractional error (eq 36)
      F  Effective halo density comparison: eq 43 vs eq 44

    Saves: fig_kg89_diagnostics_{tag}.png
    """
    try:
        K_best, F_best, bm = _load_kg89_ml_result(outd, tag)
    except FileNotFoundError as e:
        print(f"  [diagnostics] {e}")
        return

    hZ = pp.get("hZ", TA["hZ"])
    hR = pp.get("hR", TA["hR"])

    # Load saved model arrays
    z_fine       = np.asarray(bm["z"],         float)
    Kz_arr       = np.asarray(bm["Kz"],        float)
    Kzeff_arr    = np.asarray(bm["Kzeff"],     float)
    psi_arr      = np.asarray(bm["psi"],       float)
    Sigma_arr    = np.asarray(bm["Sigma"],     float)
    Sigma_oort   = np.asarray(bm["Sigma_oort"], float)
    nu_arr       = np.exp(-z_fine / hZ)

    # ── Eq 39 decomposition: disc vs DM vs tilt ──────────────────────────────
    K_kpc  = K_best * 1e6
    F_kpc  = F_best * 1e9
    Kz_disc = -(TG * K_kpc * z_fine / np.sqrt(z_fine**2 + float(D)**2))
    Kz_dm   = -(2.0 * TG * F_kpc * z_fine)
    Kz_tilt = Kzeff_arr - Kz_arr    # T·σ²_zz correction term (eq 52)

    # ── Eq 50: σ²_zz with tilt (recomputed for full accuracy) ────────────────
    s2_zz = kg.sigma_zz2_with_tilt_eq50(z_fine, R_target, nu_arr, Kz_arr,
                                          alpha=alpha, hR=hR)

    # ── Eq 23: local matter density ρ(z) from Poisson ────────────────────────
    dKz_dz      = np.gradient(Kz_arr,  z_fine)
    dKz_disc_dz = np.gradient(Kz_disc, z_fine)
    rho_total = kg.local_density_eq23(dKz_dz,      A2mB2=A2mB2)  # eq 23
    rho_disc  = kg.local_density_eq23(dKz_disc_dz, A2mB2=0.0)
    rho_dm    = np.full_like(z_fine, F_best)                       # constant DM

    # ── Eq 37: isothermal tracer density ─────────────────────────────────────
    rho0_iso = max(float(np.clip(rho_total[2], 0, None)), 0.02)
    s2_z0    = float(s2_zz[2]) if s2_zz[2] > 1.0 else 300.0
    rho_iso  = kg.isothermal_density_eq37(psi_arr, rho0_iso, s2_z0)  # eq 37

    # ── Eq 44: effective halo density (alternative formula) ─────────────────
    Sigma_mid    = float(np.interp(1.0, z_fine, Sigma_arr))
    rho_eff_44   = kg.effective_halo_density_eq44(Sigma_mid)           # eq 44

    # ── Eq 48, 51: tilt function T and integrating factor S ─────────────────
    T_arr = kg.T_tilt_eq48(R_target, z_fine, alpha=alpha, hR=hR)      # eq 48
    S_arr = kg.S_integrating_factor_eq51(R_target, z_fine,
                                          alpha=alpha, hR=hR)           # eq 51

    # ── Eq 46: σ²_Rz cross-term ──────────────────────────────────────────────
    s2_Rz = np.array([kg.sigma2_Rz_eq46(R_target, float(zi),
                                          float(s2i), alpha=alpha)
                       for zi, s2i in zip(z_fine, s2_zz)])              # eq 46

    # ── Eq 36: fractional Oort error on Σ ────────────────────────────────────
    Sig_safe = np.where(Sigma_arr > 0.1, Sigma_arr, 0.1)
    frac_oort = kg.fractional_kz_error_eq36(Sigma_arr, Sig_safe,
                                              A2mB2=A2mB2, z=z_fine)    # eq 36

    # ── Eq 45: observed K_z,eff from binned VD profile ───────────────────────
    z_bin, s2_bin, e2_bin = _get_solar_sigma2_profile(outd, tag)
    Kzeff_obs = None
    if z_bin is not None and len(z_bin) >= 4:
        nu_obs    = np.exp(-z_bin / hZ)
        Kzeff_obs = kg.Kzeff_jeans_eq45(s2_bin, nu_obs, z_bin)          # eq 45

    # ── Eq 43 vs 44: rotation-curve vs formula comparison ───────────────────
    K_scan  = np.linspace(5, 130, 200)
    F_43    = np.array([max(kg.rotation_curve_constraint_eq43(k / KG_SIGMA_UNIT)[0], 0.0)
                        * KG_RHO_UNIT for k in K_scan])
    F_44    = np.array([kg.effective_halo_density_eq44(k) for k in K_scan])

    # ──────────────── Plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()
    tag_nice = tag.replace("_", " ")

    # ── Panel A: K_z decomposition ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(z_fine, -Kz_arr    / 1e3, "k-",  lw=2.2,
            label=r"$|K_z|$ eq 39 total")
    ax.plot(z_fine, -Kzeff_arr / 1e3, "r--", lw=2.0,
            label=r"$|K_{z,\rm eff}|$ eq 52 (+tilt)")
    ax.plot(z_fine, -Kz_disc   / 1e3, color="#2166ac",  ls="-.", lw=1.7,
            label=r"Disc: $2\pi GKz/\sqrt{z^2+D^2}$ (eq 39)")
    ax.plot(z_fine, -Kz_dm     / 1e3, color="#4dac26",  ls="-.", lw=1.7,
            label=r"DM: $4\pi GFz$ (eq 39)")
    ax.plot(z_fine, -Kz_tilt   / 1e3, color="#b2abd2",  ls=":",  lw=1.7,
            label=r"Tilt correction $T\sigma_{zz}^2$ (eq 52$-$39)")
    ax.axvline(1.0, color="gray", ls=":", alpha=0.4)
    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=11)
    ax.set_ylabel(r"Force [10³ (km/s)²/kpc]", fontsize=11)
    ax.set_title(fr"(A) $K_z$ decomposition  eq 39/52 — K={K_best:.0f}", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, min(3.5, float(z_fine.max())))
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)

    # ── Panel B: Matter density profile ─────────────────────────────────────
    ax = axes[1]
    rho_t = np.clip(rho_total, 1e-6, None)
    rho_d = np.clip(rho_disc,  1e-6, None)
    ax.semilogy(z_fine, rho_t,  "k-",  lw=2.2,
                label=r"$\rho_{\rm total}$ eq 23 (Poisson)")
    ax.semilogy(z_fine, rho_d,  "b--", lw=1.8,
                label=r"$\rho_{\rm disc}$ (disc term eq 39)")
    ax.semilogy(z_fine, rho_dm, "g:",  lw=1.8,
                label=fr"$\rho_{{DM}}=F={F_best*1e3:.2f}\times10^{{-3}}$ M$_\odot$/pc³ (eq 43)")
    ax.semilogy(z_fine, np.clip(rho_iso, 1e-6, None), "r-.", lw=1.8,
                label=r"$\rho_{\rm iso}$ eq 37 (isothermal model)")
    ax.axhline(rho_eff_44, color="darkorange", ls="--", lw=1.4,
               label=fr"$\rho_{{eff}}$ eq 44 = {rho_eff_44:.4f} M$_\odot$/pc³")
    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=11)
    ax.set_ylabel(r"$\rho(z)$ [M$_\odot$/pc³]", fontsize=11)
    ax.set_title("(B) Matter density — eq 23, 37, 44", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlim(0, min(3.5, float(z_fine.max())))
    ax.tick_params(direction="in", top=True, right=True)

    # ── Panel C: Jeans self-consistency ─────────────────────────────────────
    ax = axes[2]
    ax.plot(z_fine, -Kzeff_arr / 1e3, "r-",  lw=2.2,
            label=r"Model $|K_{z,\rm eff}|$ eq 52")
    ax.plot(z_fine, -Kz_arr    / 1e3, "k--", lw=1.5, alpha=0.55,
            label=r"$|K_z|$ eq 39")
    if Kzeff_obs is not None:
        ok = (np.isfinite(Kzeff_obs) & (z_bin > 0.1) & (z_bin < 3.0)
              & (Kzeff_obs < 0))
        if ok.sum() >= 2:
            ax.errorbar(z_bin[ok], -Kzeff_obs[ok] / 1e3,
                        fmt="o", color="#2166ac", ms=5, lw=1.3, capsize=2,
                        label=r"Data-implied $|K_{z,\rm eff}|$ eq 45")
    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=11)
    ax.set_ylabel(r"$|K_{z,\rm eff}|$ [10³ (km/s)²/kpc]", fontsize=11)
    ax.set_title("(C) Jeans self-consistency: model eq 52 vs data eq 45", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlim(0, min(3.0, float(z_fine.max())))
    ax.set_ylim(bottom=0)
    ax.tick_params(direction="in", top=True, right=True)

    # ── Panel D: Velocity-ellipsoid tilt ────────────────────────────────────
    ax = axes[3]
    ax2 = ax.twinx()
    l1, = ax.plot(z_fine, T_arr, "b-",  lw=2.0,
                  label=r"$T(R,z)$ eq 48 [kpc⁻¹]")
    l2, = ax.plot(z_fine, S_arr, "g--", lw=2.0,
                  label=r"$S(R,z)$ eq 51 [—]")
    l3, = ax2.plot(z_fine, s2_Rz, "r-.", lw=1.8,
                   label=r"$\sigma_{Rz}^2$ eq 46 [(km/s)²]")
    tilt_frac = np.abs(Kz_tilt) / np.where(np.abs(Kz_arr) > 0.01, np.abs(Kz_arr), 0.01)
    l4, = ax2.plot(z_fine, tilt_frac * 100, "m:", lw=1.5,
                   label=r"Tilt fraction $|T\sigma_{zz}^2/K_z|$ [%]")
    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=11)
    ax.set_ylabel(r"$T$ [kpc⁻¹]  /  $S$ [—]", fontsize=11)
    ax2.set_ylabel(r"$\sigma_{Rz}^2$ [(km/s)²]  /  tilt frac [%]",
                   fontsize=10, color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax.set_title(r"(D) Velocity-ellipsoid tilt — eq 46, 48, 51", fontsize=10)
    ax.legend([l1, l2, l3, l4], [l.get_label() for l in [l1, l2, l3, l4]],
              frameon=False, fontsize=7.5)
    ax.set_xlim(0, min(3.5, float(z_fine.max())))
    ax.tick_params(direction="in", top=True)

    # ── Panel E: Oort correction on Σ(z) ────────────────────────────────────
    ax = axes[4]
    ax.plot(z_fine, Sigma_arr,  "k-",  lw=2.2,
            label=r"$\Sigma(z)$ eq 24, no Oort ($A^2-B^2=0$)")
    ax.plot(z_fine, Sigma_oort, "r--", lw=2.2,
            label=r"$\Sigma(z)$ eq 24 with Oort correction")
    ax_r = ax.twinx()
    frac_pct = np.abs(Sigma_oort - Sigma_arr) / np.where(Sigma_arr > 0.1, Sigma_arr, 0.1) * 100
    ax_r.plot(z_fine, frac_pct, "b:",  lw=1.7,
              label=r"$|\Delta\Sigma/\Sigma|$ eq 36 [%]")
    ax_r.plot(z_fine, np.abs(frac_oort) * 100, "c-.", lw=1.4, alpha=0.7,
              label=r"eq 36 analytical ΔΣ/Σ [%]")
    ax.set_xlabel(r"$|z|$ [kpc]", fontsize=11)
    ax.set_ylabel(r"$\Sigma(z)$ [M$_\odot$/pc²]", fontsize=11)
    ax_r.set_ylabel(r"Oort correction [%]", fontsize=11, color="b")
    ax_r.tick_params(axis="y", labelcolor="b")
    ax.set_title("(E) Oort correction — eq 24, 36", fontsize=10)
    lines_e = ax.get_lines() + ax_r.get_lines()
    ax.legend(lines_e, [l.get_label() for l in lines_e],
              frameon=False, fontsize=7.5, loc="upper left")
    ax.set_xlim(0, min(3.5, float(z_fine.max())))
    ax.tick_params(direction="in", top=True)

    # ── Panel F: Effective halo density eq 43 vs eq 44 ──────────────────────
    ax = axes[5]
    ax.plot(K_scan, F_43 * 1e3, "k-",  lw=2.2,
            label=r"$F$ eq 43: $0.041-0.0094K'$ (rotation curve constraint)")
    ax.plot(K_scan, F_44 * 1e3, "r--", lw=2.0,
            label=r"$\rho_{\rm eff}$ eq 44: $0.015-0.0047(\Sigma/50)$")
    ax.axvline(K_best, color="firebrick", ls=":", lw=1.6,
               label=fr"$K_{{best}}={K_best:.0f}$ M$_\odot$/pc²")
    ax.axhline(F_best * 1e3, color="gray", ls=":", lw=1.2,
               label=fr"$F_{{best}}={F_best*1e3:.2f}\times10^{{-3}}$")
    ax.axhline(rho_eff_44 * 1e3, color="darkorange", ls="--", lw=1.3,
               label=fr"$\rho_{{eff,44}}={rho_eff_44*1e3:.2f}\times10^{{-3}}$")
    ax.set_xlabel(r"$K$ [M$_\odot$/pc²]", fontsize=11)
    ax.set_ylabel(r"Halo density [10⁻³ M$_\odot$/pc³]", fontsize=11)
    ax.set_title("(F) Halo density: eq 43 (rotation curve) vs eq 44", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_xlim(K_scan[0], K_scan[-1])
    ax.tick_params(direction="in", top=True, right=True)

    fig.suptitle(
        f"KG89 Extended Diagnostics — {tag_nice}\n"
        fr"$K_{{best}}={K_best:.1f}$ M$_\odot$/pc²,  "
        fr"$F_{{best}}={F_best*1e3:.2f}\times10^{{-3}}$ M$_\odot$/pc³,  "
        fr"$h_Z={hZ:.3f}$ kpc,  $h_R={hR:.3f}$ kpc",
        fontsize=11, fontweight="bold", y=1.005)
    fig.tight_layout()

    fname = f"fig_kg89_diagnostics_{tag}.png".replace(" ", "_")
    fig.savefig(outd / fname, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  KG89 extended diagnostics → {outd / fname}")


# ---------------------------------------------------------------------------
# Convenience wrapper: run the full KG89 suite in one call
# ---------------------------------------------------------------------------

def run_kg89_full_suite(data_dict, fa, pp, outd, tag="",
                         K_lo=20.0, K_hi=100.0, nK=25,
                         D=0.4, alpha=2.0, hR_tilt=None,
                         sigma_lnz=0.15, sigma_v=4.0,
                         R_target=8.5, R_width=0.5):
    """Run KG89 validation figures (Fig 2, Fig 7) and the ML-DF pipeline (Fig 8).

    Returns the result dict from run_kg89_ml_pipeline.
    """
    print(f"\n{'='*70}")
    print(f"KG89 (1989) pipeline — {tag or 'all-stars'}")
    print(f"{'='*70}")

    print("  [KG89] Reproducing Fig 2 (Kz → Σ, eq 27/28) ...")
    frac_dev = plot_kg89_fig2_kz_sigma(outd)
    print(f"    max |Kz/(2πG) − Σ|/Σ = {frac_dev:.4f}")

    print("  [KG89] Reproducing Fig 7 (tilt T(R,z), eq 48) ...")
    T_min = plot_kg89_fig7_tilt(outd, R0_plot=R_target, alpha=alpha,
                                 hR=hR_tilt if hR_tilt else TA["hR"])
    print(f"    min T(R0,z) = {T_min:.4f} kpc⁻¹  (expect < 0)")

    print("  [KG89] Running ML-DF pipeline (Fig 8 flowchart) ...")
    res = run_kg89_ml_pipeline(
        data_dict, outd, tag=tag,
        K_lo=K_lo, K_hi=K_hi, nK=nK,
        D=D, alpha=alpha, hR_tilt=hR_tilt,
        sigma_lnz=sigma_lnz, sigma_v=sigma_v,
        R_target=R_target, R_width=R_width)

    plot_kg89_ml_summary(res, outd)
    plot_kg89_sigma_comparison(res, outd, fa=fa, pp=pp)
    plot_kg89_oort_rotcurve(res["oort"], res["K_best"], outd, tag=tag)

    # Extended diagnostics: equations not called in the original pipeline
    print("  [KG89] Extended diagnostics (eq 23, 36, 37, 44, 45, 46, 48, 51) ...")
    A2mB2 = float(res.get("A2mB2", 0.0))
    plot_kg89_sigma2_exact(outd, tag, pp, D=D, alpha=alpha, R_target=R_target)
    plot_kg89_complete_diagnostics(outd, tag, pp, D=D, alpha=alpha,
                                    R_target=R_target, A2mB2=A2mB2)

    print(f"{'='*70}\n")
    return res


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True,
                   help="Real NPZ cache with R,Z,vR,vZ,vphi/vp. chem_mgfe is required only for --mode chem_mgfe.")
    p.add_argument("--mode", choices=["chem_mgfe", "combined"], default="chem_mgfe",
                   help="chem_mgfe reproduces thin/thick notebook split; combined treats all stars as one tracer.")
    p.add_argument("--combined-hZ", type=float, default=0.70)
    p.add_argument("--combined-hR", type=float, default=2.30)
    p.add_argument("--combined-group-size", type=int, default=2000)
    p.add_argument("--combined-density-mode",
                   choices=["fixed", "fitted_global", "fitted_local"],
                   default="fitted_global",
                   help="How to model all-star tracer density in combined mode. "
                        "fixed: use hZ/hR as given; "
                        "fitted_global: fit effective hZ_eff/hR_eff from star counts; "
                        "fitted_local: use smoothed local gradients from star counts.")
    p.add_argument("--r-bin-width", type=float, default=1.0)
    p.add_argument("--fit-clip", type=float, default=3.0)
    p.add_argument("--sigz-model", choices=["linear", "best", "quadratic", "tanh"], default="linear",
                   help="sigma_Z model used in Eq.(1). Default 'linear' is the Cheng+2024 main-text assumption; "
                        "'best' is a diagnostic BIC selection among linear/quadratic/tanh.")
    p.add_argument("--chem-selection", choices=["mgfe", "combined"], default="mgfe",
                   help="Chemical thin/thick selection for overlay and Cheng comparison paths. "
                        "Default 'mgfe' follows Cheng+2024 Fig. 1; 'combined' pools MgFe and alpha-Fe labels for diagnostics.")
    p.add_argument("--no-robust-fit", action="store_true")
    p.add_argument("--r-bin-width-fine", type=float, default=0.25,
                   help="Finer R-bin width (kpc) for a second, denser analysis alongside the main one.")
    p.add_argument("--overlay-chem-dir",
                   default="/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/cheng2024_direct_chemical_abundance_only_surface_density_20260601")
    p.add_argument("--chem-cache",
                   default="/user/sutirtha/RC_FINAL_PAPER/GRAND_RC_LS/cheng2024_direct_chemical_abundance_only_surface_density_20260601/reduced_catalogue_all_no_mgfe_cheng2024_corrected.npz",
                   help="Chemistry-tagged NPZ (with chem_mgfe/chem_alpha) produced by surface_density.py. "
                        "When provided, all chemical overlay plots are computed from raw stars instead of loading precomputed CSVs.")
    p.add_argument("--no-overlays", action="store_true")
    p.add_argument("--out-dir", default="./cheng2024_figs")
    p.add_argument("--no-shm", action="store_true")
    p.add_argument("--nboot", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)

    # ── KG89 (Kuijken & Gilmore 1989) pipeline flags ─────────────────────────
    p.add_argument("--kg89-ml", action="store_true",
                   help="Run full KG89 ML-DF pipeline (Fig 8 flowchart, eq 18-21, 39-43, 53-57).")
    p.add_argument("--kg89-K-lo", type=float, default=20.0,
                   help="Lower bound of trial disc surface density grid K [M_sun/pc²].")
    p.add_argument("--kg89-K-hi", type=float, default=100.0,
                   help="Upper bound of trial disc surface density grid K [M_sun/pc²].")
    p.add_argument("--kg89-nK", type=int, default=25,
                   help="Number of K grid points for ML scan.")
    p.add_argument("--kg89-D", type=float, default=0.4,
                   help="Disc equivalent scaleheight D for KG89 eq 40 [kpc].")
    p.add_argument("--kg89-alpha", type=float, default=2.0,
                   help="Velocity-ellipsoid axis ratio α = σ_R/σ_z for tilt eq 48/50/51/52.")
    p.add_argument("--kg89-hR-tilt", type=float, default=None,
                   help="Radial scalelength for tilt eq 48/51 [kpc]. "
                        "Defaults to --combined-hR / TN['hR'] depending on mode.")
    p.add_argument("--kg89-sigma-lnz", type=float, default=0.15,
                   help="Fractional distance error σ(ln z) for error convolution (eq 56-57).")
    p.add_argument("--kg89-sigma-v", type=float, default=4.0,
                   help="Velocity measurement error σ_v [km/s] for error convolution (eq 56-57).")
    p.add_argument("--kg89-R-target", type=float, default=8.5,
                   help="Centre of solar annulus for KG89 tracer selection [kpc].")
    p.add_argument("--kg89-R-width", type=float, default=0.5,
                   help="Half-width of solar annulus for KG89 tracer selection [kpc].")
    p.add_argument("--kg89-only", action="store_true",
                   help="Skip ALL existing analysis. Load profile_all.csv, linear_fits_all.csv, "
                        "allstar_density_eff_scalelengths.csv etc. from --out-dir and run ONLY "
                        "the KG89 ML-DF pipeline (Fig 8 flowchart). No existing plots are "
                        "overwritten. --cache is still required for the raw (R,Z,vZ) stellar data.")

    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    outd = Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)

    print(f"G = {G:.4e}, 2piG = {TG:.4e} kpc(km/s)^2/Msun")
    print("Sigma [Msun/pc^2] -> Kz [km^2/s^2/kpc]: 1 ->", f"{s2k(1):.4e}")

    # ── KG89-only: load existing CSVs, run KG89 on EVERY sub-population ─────
    if getattr(args, "kg89_only", False):
        # Require at least the all-stars fits table
        fa_all_path = outd / "linear_fits_all.csv"
        if not fa_all_path.exists():
            raise FileNotFoundError(
                f"--kg89-only requires pre-computed {fa_all_path}.\n"
                "Run the full pipeline first so CSVs are present in --out-dir.")

        # Load full chemistry-tagged NPZ (real stars only, no fallback)
        chem_da = load_chem_data(Path(args.cache))
        n_total = len(chem_da["R"])
        all_mask = np.ones(n_total, dtype=bool)
        thin_mask, thick_mask = get_chem_masks(chem_da, args.chem_selection)
        cm = chem_da.get("chem_mgfe", np.zeros(n_total, dtype=np.uint8))
        mgfe_thick_mask = (cm == CHEM_THICK)

        # Build pp for all-stars from fitted scalelengths CSV
        pp_all = dict(TA)
        scl_path = outd / "allstar_density_eff_scalelengths.csv"
        if scl_path.exists():
            scl = pd.read_csv(scl_path)
            if not scl.empty:
                pp_all["hZ"] = float(scl["hZ_eff"].iloc[0])
                pp_all["hR"] = float(scl["hR_eff"].iloc[0])
        hs_path = outd / "hsigma_fit.csv"
        if hs_path.exists():
            hs_df = pd.read_csv(hs_path)
            if not hs_df.empty:
                pp_all["hs"] = float(hs_df["h_sigma"].iloc[0])

        # Each entry: (tag, star_mask, linear_fits_csv_path, pp_dict)
        sub_populations = [
            ("all_stars",      all_mask,       fa_all_path,                                   pp_all),
            ("chem_thin",      thin_mask,      outd / "chem_combined_thin_linear_fits.csv",   dict(TN)),
            ("chem_thick",     thick_mask,     outd / "chem_combined_thick_linear_fits.csv",  dict(TK)),
            ("mgfe_thick_only",mgfe_thick_mask,outd / "chem_mgfe_thick_only_linear_fits.csv", dict(TK)),
        ]

        for tag, mask, fa_path, pp in sub_populations:
            n_sub = int(mask.sum())
            if not fa_path.exists():
                print(f"[KG89-only] skipping {tag}: {fa_path.name} not found")
                continue
            if n_sub < 200:
                print(f"[KG89-only] skipping {tag}: only {n_sub} stars in solar annulus")
                continue
            fa = pd.read_csv(fa_path)
            # Subset raw stellar arrays to this population
            sub = {k: chem_da[k][mask] for k in ("R", "Z", "vR", "vZ", "vp")
                   if k in chem_da}
            print(f"\n[KG89-only] {tag}: {n_sub:,} stars, "
                  f"hZ={pp['hZ']:.3f} hR={pp['hR']:.3f} hs={pp['hs']:.3f} kpc, "
                  f"{len(fa)} R-bins from {fa_path.name}")
            # Keep global TA in sync so sig_from_row inside plots uses correct pp
            TA["hZ"] = pp["hZ"]; TA["hR"] = pp["hR"]; TA["hs"] = pp["hs"]
            try:
                run_kg89_full_suite(
                    data_dict=sub, fa=fa, pp=pp, outd=outd,
                    tag=tag,
                    K_lo=args.kg89_K_lo, K_hi=args.kg89_K_hi, nK=args.kg89_nK,
                    D=args.kg89_D, alpha=args.kg89_alpha,
                    hR_tilt=(args.kg89_hR_tilt if args.kg89_hR_tilt is not None
                             else pp["hR"]),
                    sigma_lnz=args.kg89_sigma_lnz, sigma_v=args.kg89_sigma_v,
                    R_target=args.kg89_R_target, R_width=args.kg89_R_width)
            except Exception as _e:
                print(f"  [KG89 WARNING] {tag}: {_e}")
                import traceback; traceback.print_exc()

        print(f"\nAll KG89 sub-population figures saved to: {outd.resolve()}")
        return

    if args.mode == "combined":
        TA["hZ"] = args.combined_hZ
        TA["hR"] = args.combined_hR
        TA["ng"] = args.combined_group_size
        da = load_combined_data(Path(args.cache))

        # ── Chemical raw computation ────────────────────────────────────────
        chem_crz_thin = chem_crz_thick = None
        thick_lin_row = thin_lin_row = None
        fa_thick_lin = fa_thin_lin = pd.DataFrame()
        fa_mgfe_thick_lin_main = pd.DataFrame()   # pure MgFe-only thick, Cheng params
        chem_path = Path(args.chem_cache) if args.chem_cache else None
        if not args.no_overlays and chem_path and chem_path.exists():
            print(f"Loading chemistry-tagged catalogue: {chem_path}")
            chem_da = load_chem_data(chem_path)
            thin_mask, thick_mask = get_chem_masks(chem_da, args.chem_selection)
            chem_label = "MgFe-only" if args.chem_selection == "mgfe" else "MgFe+alpha pooled"
            print(f"  {chem_label} thin:  {int(thin_mask.sum()):,} stars")
            print(f"  {chem_label} thick: {int(thick_mask.sum()):,} stars")
            print("Computing chemical sub-sample overlays from raw stars...")
            overlays_computed, chem_raw_profs = compute_chem_overlays_from_raw(
                chem_da, thin_mask, thick_mask,
                nboot=args.nboot, rbw=args.r_bin_width, rng=rng,
                sigz_model=args.sigz_model, clip=args.fit_clip,
                nmc=400, outd=outd,
                selection_label=("MgFe" if args.chem_selection == "mgfe" else "Combined"))

            # ── Finer R-bin full analysis (same pipeline as standard) ─────────
            if args.r_bin_width_fine < args.r_bin_width:
                print(f"Computing fine-bin chemical overlays (rbw={args.r_bin_width_fine} kpc)...")
                fine_outd = outd / "fine_bins"
                fine_outd.mkdir(parents=True, exist_ok=True)
                overlays_fine, _ = compute_chem_overlays_from_raw(
                    chem_da, thin_mask, thick_mask,
                    nboot=args.nboot, rbw=args.r_bin_width_fine, rng=rng,
                    sigz_model=args.sigz_model, clip=args.fit_clip,
                    nmc=400, outd=fine_outd,
                    selection_label=("MgFe" if args.chem_selection == "mgfe" else "Combined"))
                for ov in overlays_fine:
                    ov["label"] += f" (rbw={args.r_bin_width_fine})"
                    ov["ls"] = ":"
                overlays_computed = overlays_computed + overlays_fine

            # ── Linear fits: combined thin/thick + pure MgFe-only thick ──────
            thick_lin_row = thin_lin_row = None
            thick_prof = chem_raw_profs.get("combined_thick", pd.DataFrame())
            thin_prof  = chem_raw_profs.get("combined_thin",  pd.DataFrame())
            if not thick_prof.empty:
                print("  σ_Z fit for thick disc (forced linear; Cheng+2024 main-text assumption)...")
                fa_thick_lin, _ = fit_bins(thick_prof, zmx=TK["zmax"], robust=True,
                                           clip=args.fit_clip, sigz_model="linear", pp=TK)
                fa_thick_lin.to_csv(outd / "chem_combined_thick_linear_fits.csv", index=False)
                thick_lin_row = get_solar_row(fa_thick_lin)
            if not thin_prof.empty:
                print("  σ_Z fit for thin disc (forced linear; Cheng+2024 main-text assumption)...")
                fa_thin_lin, _ = fit_bins(thin_prof, zmx=TN["zmax"], robust=True,
                                          clip=args.fit_clip, sigz_model="linear", pp=TN)
                fa_thin_lin.to_csv(outd / "chem_combined_thin_linear_fits.csv", index=False)
                thin_lin_row = get_solar_row(fa_thin_lin)

            # Keep the BIC-selected alternatives as explicitly named diagnostics.
            if not thick_prof.empty:
                fa_thick_best, qa_thick_best = fit_bins(
                    thick_prof, zmx=TK["zmax"], robust=True,
                    clip=args.fit_clip, sigz_model="best", pp=TK)
                fa_thick_best.to_csv(outd / "chem_combined_thick_best_fits.csv", index=False)
                qa_thick_best.to_csv(outd / "chem_combined_thick_sigz_model_selection.csv", index=False)
            if not thin_prof.empty:
                fa_thin_best, qa_thin_best = fit_bins(
                    thin_prof, zmx=TN["zmax"], robust=True,
                    clip=args.fit_clip, sigz_model="best", pp=TN)
                fa_thin_best.to_csv(outd / "chem_combined_thin_best_fits.csv", index=False)
                qa_thin_best.to_csv(outd / "chem_combined_thin_sigz_model_selection.csv", index=False)

            # ── Pure MgFe-only thick disc (Cheng's exact selection + parameters) ──
            # chem_mgfe==2 only: 71,633 stars, hZ=0.90, hR=2.0, hσ=5.03 kpc
            fa_mgfe_thick_lin_main = pd.DataFrame()
            _pp_cheng_tk = dict(hZ=0.90, hR=2.0, hs=5.03, hs_e=1.36,
                                zmax=4.0, c="#d6604d", ls="-", ng=200, alpha=0.22)
            if "chem_mgfe" in chem_da:
                mask_mgfe_only = chem_da["chem_mgfe"] == CHEM_THICK
                n_mgfe_tk = int(mask_mgfe_only.sum())
                print(f"  Pure MgFe-only thick: {n_mgfe_tk:,} stars → linear cprof...")
                if n_mgfe_tk >= 200:
                    prof_m = _cprof_subset(chem_da, mask_mgfe_only, 200,
                                           args.r_bin_width, args.nboot, rng,
                                           label="MgFe-thick")
                    if not prof_m.empty:
                        pd.DataFrame({
                            "R_mid": prof_m.rm,       "Z_median": prof_m.zm,
                            "sigma_R": prof_m.sR,     "sigma_R_err": prof_m.sRe,
                            "sigma_phi": prof_m.sp,   "sigma_phi_err": prof_m.spe,
                            "sigma_Z": prof_m.sZ,     "sigma_Z_err": prof_m.sZe,
                            "sigma_RZ2": prof_m.crz,  "sigma_RZ2_err": prof_m.crze,
                        }).to_csv(outd / "chem_mgfe_thick_only_velocity_dispersion_profiles.csv",
                                  index=False)
                        fa_mgfe_thick_lin_main, _ = fit_bins(
                            prof_m, zmx=4.0, robust=True, clip=args.fit_clip,
                            sigz_model="linear", pp=_pp_cheng_tk)
                        fa_mgfe_thick_lin_main.to_csv(
                            outd / "chem_mgfe_thick_only_linear_fits.csv", index=False)
                        print(f"  MgFe-only thick linear: {len(fa_mgfe_thick_lin_main)} R-bins")

            print("Computing sigma_RZ^2 vs R for h_sigma panels...")
            chem_crz_thin, chem_crz_thick = compute_chem_crz_from_raw(
                chem_da, thin_mask, thick_mask,
                nboot=args.nboot, rbw=args.r_bin_width, rng=rng)
            chem_crz_thin.to_csv(outd / "chem_combined_thin_crz_for_hsigma.csv", index=False)
            chem_crz_thick.to_csv(outd / "chem_combined_thick_crz_for_hsigma.csv", index=False)
            # Also load the precomputed CSV overlays and merge all into one list
            overlay_keys = (["mgfe_feh_thin_chem", "mgfe_feh_thick_chem"]
                            if args.chem_selection == "mgfe" else None)
            overlays_precomputed = load_overlay_sets(args.overlay_chem_dir, keys=overlay_keys)
            overlays = overlays_precomputed + overlays_computed
        else:
            # Fall back to precomputed CSV overlays only
            overlay_keys = (["mgfe_feh_thin_chem", "mgfe_feh_thick_chem"]
                            if args.chem_selection == "mgfe" else None)
            overlays = [] if args.no_overlays else load_overlay_sets(args.overlay_chem_dir, keys=overlay_keys)
        # ────────────────────────────────────────────────────────────────────

        if overlays:
            print("Overlays ready:", ", ".join(ov["label"] for ov in overlays))
        print("Computing combined velocity dispersion profile...")
        profile_path = outd / "profile_all.csv"
        if profile_path.exists():
            pa = pd.read_csv(profile_path)
            print(f"Using existing combined profile: {profile_path}")
        else:
            print(f"Computing all-stars velocity dispersion profile (ng={TA['ng']}, rbw={args.r_bin_width})...")
            pa = cprof(da, ng=TA["ng"], rbw=args.r_bin_width, nb=args.nboot, rng=rng, label="all-stars")
            pa.to_csv(profile_path, index=False)
        print(f"All-star groups: {len(pa):,}")
        ov_dir = None if args.no_overlays else args.overlay_chem_dir

        # --- All-star effective tracer density fitting ---
        density_mode = args.combined_density_mode
        density_result = None
        if density_mode != "fixed":
            print(f"Fitting all-star effective tracer density (mode={density_mode})...")
            density_result = fit_allstar_density_gradients(
                da["R"], da["Z"],
                rl0=4.0, rl1=12.0, rbw=args.r_bin_width,
                zmax=TA["zmax"], dz=0.3, R0=R0,
                Nmin=50, n_boot=200, rng=rng, outd=outd)
            hZ_fit = density_result["hZ_eff"]
            hR_fit = density_result["hR_eff"]
            print(f"  hZ_eff = {hZ_fit:.3f} kpc  (input was {args.combined_hZ:.3f})")
            print(f"  hR_eff = {hR_fit:.3f} kpc  (input was {args.combined_hR:.3f})")
            if np.isfinite(hZ_fit) and hZ_fit > 0:
                TA["hZ"] = hZ_fit
                TA["hR"] = hR_fit
                # Propagate bootstrap arrays for mc_sig sampling
                TA["hZ_boot"] = density_result["hZ_boot"]
                TA["hR_boot"] = density_result["hR_boot"]
            plot_allstar_density_diagnostics(density_result, outd)
            pd.DataFrame([dict(density_mode=density_mode,
                               hZ_eff=TA["hZ"], hR_eff=TA["hR"],
                               hZ_lo=density_result["hZ_lo"], hZ_hi=density_result["hZ_hi"],
                               hR_lo=density_result["hR_lo"], hR_hi=density_result["hR_hi"],
                               n_bins=density_result["n_bins"])
                           ]).to_csv(outd / "allstar_density_eff_scalelengths.csv", index=False)
        else:
            print(f"All-star density mode=fixed: using hZ={TA['hZ']:.2f}, hR={TA['hR']:.2f} kpc")

        plot_tilt_diagnostic_combined(pa, outd)
        plot_velocity_figures_combined(pa, outd, overlays=overlays)

        print("Fitting h_sigma (thin/thick from overlay chemical profiles)...")
        hs_a, hse_a = plot_hsigma_combined(
            pa, outd,
            overlay_dir=ov_dir if chem_crz_thin is None else None,
            da=da,
            chem_crz_thin=chem_crz_thin,
            chem_crz_thick=chem_crz_thick,
            chem_selection=args.chem_selection)
        TA["hs"] = hs_a if np.isfinite(hs_a) else TA["hs"]
        TA["hs_e"] = hse_a if np.isfinite(hse_a) else TA["hs_e"]
        pd.DataFrame([dict(pop="all", density_mode=density_mode,
                           h_sigma=TA["hs"], h_sigma_err=TA["hs_e"],
                           hZ=TA["hZ"], hR=TA["hR"], group_size=TA["ng"])
                      ]).to_csv(outd / "hsigma_fit.csv", index=False)

        print("Fitting sigma_Z and sigma_RZ profiles (through-origin odd sigma_RZ fit)...")
        fa, qa = fit_bins(pa, zmx=TA["zmax"], robust=not args.no_robust_fit,
                          clip=args.fit_clip, sigz_model=args.sigz_model, pp=TA)
        fa.to_csv(outd / "linear_fits_all.csv", index=False)
        fa.to_csv(outd / "surface_fits_all.csv", index=False)
        qa.to_csv(outd / "sigz_model_selection_all.csv", index=False)
        fit_quality_table(pa, fa, outd, "all")
        print("All-star fits:")
        print(fa[["rm", "sigz_model", "sigz_rmse", "rz_b", "m", "me", "fn", "n"]].to_string(index=False, float_format="{:.2f}".format))

        # Linear-only all-stars fits (Cheng+2024 comparison)
        print("Fitting linear-only sigma_Z (for Cheng+2024 comparison)...")
        fa_lin_all, _ = fit_bins(pa, zmx=TA["zmax"], robust=not args.no_robust_fit,
                                  clip=args.fit_clip, sigz_model="linear", pp=TA)
        fa_lin_all.to_csv(outd / "linear_fits_all_linear_only.csv", index=False)

        # Fine R-bin all-stars: full analysis with surface density
        if args.r_bin_width_fine < args.r_bin_width:
            print(f"Computing fine-bin all-stars profile (rbw={args.r_bin_width_fine} kpc)...")
            profile_fine_path = outd / f"profile_all_rbw{args.r_bin_width_fine:.2f}.csv"
            if not profile_fine_path.exists():
                pa_fine = cprof(da, ng=TA["ng"], rbw=args.r_bin_width_fine,
                                nb=args.nboot, rng=rng)
                pa_fine.to_csv(profile_fine_path, index=False)
            else:
                pa_fine = pd.read_csv(profile_fine_path)
                print(f"  Using cached fine profile: {profile_fine_path}")
            fa_fine, _ = fit_bins(pa_fine, zmx=TA["zmax"], robust=not args.no_robust_fit,
                                   clip=args.fit_clip, sigz_model=args.sigz_model, pp=TA)
            fa_fine.to_csv(outd / f"linear_fits_all_fine_rbw{args.r_bin_width_fine:.2f}.csv",
                           index=False)
            print(f"  Fine-bin: {len(fa_fine)} R-bins (vs {len(fa)} standard) — sigma fits only")

        plot_fit_diagnostics_combined(pa, fa, outd, overlays=overlays)
        plot_appendix_sigrz_combined(pa, fa, outd, overlays=overlays)
        plot_sigz_model_selection(pa, fa, outd, "all", TA)
        _mgfe_tk_solar = get_solar_row(fa_mgfe_thick_lin_main) if not fa_mgfe_thick_lin_main.empty else None
        plot_fig6_combined(fa, outd, rng, overlays=overlays,
                           fa_lin=fa_lin_all,
                           thick_lin_row=thick_lin_row,
                           thin_lin_row=thin_lin_row,
                           mgfe_thick_lin_row=_mgfe_tk_solar)
        plot_fig7_combined(fa, outd, rng, overlays=overlays,
                           fa_lin=fa_lin_all,
                           fa_thin_lin=fa_thin_lin,
                           fa_thick_lin=fa_thick_lin,
                           fa_mgfe_thick_lin=fa_mgfe_thick_lin_main)
        plot_fig8_combined(fa, outd, rng, overlays=overlays)
        plot_fig9_combined(fa, outd, rng, overlays=overlays)
        if not fa_thin_lin.empty or not fa_thick_lin.empty:
            print("Generating Cheng+2024 linear-sigma_Z comparison figure...")
            plot_cheng_linear_comparison(fa_thin_lin, fa_thick_lin, outd, rng,
                                          overlays=overlays,
                                          chem_data=chem_da if chem_path and chem_path.exists() else None,
                                          fa_mgfe_thick_lin=fa_mgfe_thick_lin_main)
            print("Generating sigma_Z model surface density comparison grid...")
            plot_sigz_surface_density_grid(fa_thin_lin, fa_thick_lin, outd, rng)
            if 'chem_da' in locals():
                print("Generating bin-wise number-density diagnostics and density-fit sigma_Z grids...")
                density_points, density_params = fit_binwise_number_density_models(
                    chem_da, thin_mask, thick_mask, fa_thin_lin, fa_thick_lin,
                    outd, rbw=args.r_bin_width)
                for density_kind in ["exp_vertical", "exp_zr", "sech2_vertical"]:
                    plot_binwise_number_density_fits(
                        density_points, density_params, outd, density_kind)
                    plot_sigz_surface_density_grid_binwise_density(
                        fa_thin_lin, fa_thick_lin, density_params, outd, density_kind)
        res = plot_fig10_combined(pa, outd, overlay_dir=ov_dir,
                                   chem_da=chem_da if 'chem_da' in dir() else None)
        write_consistency_summary_combined(fa, res, outd)
        save_eq1_components_combined(fa, outd, density_result=density_result,
                                     density_mode=density_mode)
        df = save_surface_table_combined(fa, outd, rng)
        solar = df[np.isclose(df["R"], 8.5)]
        if not solar.empty:
            print("Solar-circle table:")
            print(solar.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        # ── KG89 ML-DF pipeline (Fig 8 flowchart) ────────────────────────────
        if args.kg89_ml:
            try:
                run_kg89_full_suite(
                    data_dict=da,
                    fa=fa, pp=TA,
                    outd=outd,
                    tag="combined",
                    K_lo=args.kg89_K_lo, K_hi=args.kg89_K_hi, nK=args.kg89_nK,
                    D=args.kg89_D,
                    alpha=args.kg89_alpha,
                    hR_tilt=(args.kg89_hR_tilt if args.kg89_hR_tilt is not None
                             else TA["hR"]),
                    sigma_lnz=args.kg89_sigma_lnz,
                    sigma_v=args.kg89_sigma_v,
                    R_target=args.kg89_R_target,
                    R_width=args.kg89_R_width)
            except Exception as _kg_exc:
                print(f"  [KG89 WARNING] Pipeline raised an exception: {_kg_exc}")
                import traceback; traceback.print_exc()

        print(f"All outputs saved to: {outd.resolve()}")
        return

    dn, dk = load_data(Path(args.cache))

    print("Computing velocity dispersion profiles...")
    pn = cprof(dn, ng=TN["ng"], rbw=args.r_bin_width, nb=args.nboot, rng=rng)
    pk = cprof(dk, ng=TK["ng"], rbw=args.r_bin_width, nb=args.nboot, rng=rng)
    pn.to_csv(outd / "profile_thin.csv", index=False)
    pk.to_csv(outd / "profile_thick.csv", index=False)
    print(f"Thin groups: {len(pn):,}; Thick groups: {len(pk):,}")
    plot_tilt_diagnostic(pn, pk, outd)
    plot_velocity_figures(pn, pk, outd)

    print("Fitting h_sigma...")
    hs_n, hse_n, _ = fit_hs(pn, zts=[-1, 1])
    hs_k, hse_k, _ = fit_hs(pk, zts=[-2, -1, 1, 2])
    print(f"Thin  h_sigma = {hs_n:.2f} +/- {hse_n:.2f} kpc")
    print(f"Thick h_sigma = {hs_k:.2f} +/- {hse_k:.2f} kpc")
    TN["hs"] = hs_n if np.isfinite(hs_n) else 4.53
    TK["hs"] = hs_k if np.isfinite(hs_k) else 5.03
    pd.DataFrame([
        dict(pop="thin", h_sigma=TN["hs"], h_sigma_err=hse_n),
        dict(pop="thick", h_sigma=TK["hs"], h_sigma_err=hse_k),
    ]).to_csv(outd / "hsigma_fit.csv", index=False)
    plot_hsigma(pn, pk, outd)

    print("Fitting sigma_Z and sigma_RZ profiles (through-origin odd sigma_RZ fit)...")
    fn, qn = fit_bins(pn, zmx=TN["zmax"], robust=not args.no_robust_fit,
                      clip=args.fit_clip, sigz_model=args.sigz_model, pp=TN)
    fk, qk = fit_bins(pk, zmx=TK["zmax"], robust=not args.no_robust_fit,
                      clip=args.fit_clip, sigz_model=args.sigz_model, pp=TK)
    fn.to_csv(outd / "linear_fits_thin.csv", index=False)
    fk.to_csv(outd / "linear_fits_thick.csv", index=False)
    fn.to_csv(outd / "surface_fits_thin.csv", index=False)
    fk.to_csv(outd / "surface_fits_thick.csv", index=False)
    qn.to_csv(outd / "sigz_model_selection_thin.csv", index=False)
    qk.to_csv(outd / "sigz_model_selection_thick.csv", index=False)
    fit_quality_table(pn, fn, outd, "thin")
    fit_quality_table(pk, fk, outd, "thick")
    print("Thin fits:")
    print(fn[["rm", "sigz_model", "sigz_rmse", "rz_b", "m", "me", "fn", "n"]].to_string(index=False, float_format="{:.2f}".format))
    print("Thick fits:")
    print(fk[["rm", "sigz_model", "sigz_rmse", "rz_b", "m", "me", "fn", "n"]].to_string(index=False, float_format="{:.2f}".format))

    plot_fit_diagnostics(pn, pk, fn, fk, outd)
    plot_sigz_model_selection(pn, fn, outd, "thin", TN)
    plot_sigz_model_selection(pk, fk, outd, "thick", TK)
    plot_fig6(fn, fk, outd, rng, plot_shm=not args.no_shm)
    plot_fig7(fn, fk, outd, rng)
    plot_fig8(fn, fk, outd, rng)
    plot_fig9(fn, fk, outd, rng)
    res_n, res_k = plot_fig10(pn, pk, outd)
    write_consistency_summary(fn, fk, res_n, res_k, outd)
    df = save_surface_table(fn, fk, outd, rng)

    solar = df[np.isclose(df["R"], 8.5)]
    if not solar.empty:
        print("Solar-circle table:")
        print(solar.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ── KG89 ML-DF pipeline: run on thin disc data (eq 18-21, 39-57) ────────
    if args.kg89_ml:
        for pop_lbl, pop_data, pop_fa, pop_pp in [
                ("thin",  dn, fn, TN),
                ("thick", dk, fk, TK)]:
            try:
                run_kg89_full_suite(
                    data_dict=pop_data,
                    fa=pop_fa, pp=pop_pp,
                    outd=outd,
                    tag=pop_lbl,
                    K_lo=args.kg89_K_lo, K_hi=args.kg89_K_hi, nK=args.kg89_nK,
                    D=args.kg89_D,
                    alpha=args.kg89_alpha,
                    hR_tilt=(args.kg89_hR_tilt if args.kg89_hR_tilt is not None
                             else pop_pp["hR"]),
                    sigma_lnz=args.kg89_sigma_lnz,
                    sigma_v=args.kg89_sigma_v,
                    R_target=args.kg89_R_target,
                    R_width=args.kg89_R_width)
            except Exception as _kg_exc:
                print(f"  [KG89 WARNING] {pop_lbl}: {_kg_exc}")
                import traceback; traceback.print_exc()

    print(f"All outputs saved to: {outd.resolve()}")


if __name__ == "__main__":
    main()
