#!/usr/bin/env python3
"""
Merge two isochrone models by CMD intersection (V vs V-I).

For each age (log10 6.5..10.0 step 0.1):
 - find intersection in V vs V-I (rounded to mag-decimals).
 - build merged CMD: V < V_intersection  -> take from PARSEC (file2)
                     V > V_intersection  -> take from BHAC15 (file1)
 - output per-age CSV: merged_iso_age_<age>.csv with columns: mass, V, I, VminusI, source
 - save verification PNG showing both curves, intersection, and merged curve.

This variant also stores U and B magnitudes, but ONLY from PARSEC.
BHAC15 rows will have U/B = NaN. Midpoint rows (if present) will NOT be
filled for U/B (they remain NaN). No midpoint averaging is used for U/B.

Usage:
  python merge_iso_by_cmd_intersection.py --file_bhac BHAC15_iso_UBVRI.txt \
      --file_parsec isocronas_UBVRI_idades_6.5-10.dat --outdir merged_iso --mag-decimals 1 --max-V 8.0 --n-mass 5000

Requirements:
  pip install numpy scipy pandas matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from pathlib import Path
import argparse, re, sys

# ------------------ header parsing & detection utilities ------------------
def read_isochrone_with_header(fname):
    fname = Path(fname)
    header_lines = []
    with open(fname, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip().startswith('#'):
                header_lines.append(line.rstrip('\n'))
            else:
                break
    col_line = None
    for hl in reversed(header_lines):
        s = hl.lstrip('#').strip()
        tokens = re.split(r'\s+', s)
        if len(tokens) >= 4 and any(k in s for k in ['logAge','Age','Mini','M/Ms','Mv','Vmag','Teff']):
            col_line = s
            break
    colnames = re.split(r'\s+', col_line) if col_line else None
    df = pd.read_csv(fname, comment='#', delim_whitespace=True, header=None, engine='python')
    if colnames is not None and df.shape[1] == len(colnames):
        df.columns = colnames
    else:
        df.columns = [f'c{i}' for i in range(df.shape[1])]
    return df

def detect_age_column(df):
    for name in ['logAge','logAge(yr)','Age','age','log_age']:
        if name in df.columns:
            return name
    for col in df.columns:
        arr = pd.to_numeric(df[col], errors='coerce').dropna()
        if ((arr > 4.5) & (arr < 10.5)).mean() > 0.6:
            return col
    raise RuntimeError("Could not detect age column.")

def detect_mass_column(df):
    for name in ['Mini','M/Ms','M/Msun','M','mass','m']:
        if name in df.columns:
            return name
    for col in df.columns:
        arr = pd.to_numeric(df[col], errors='coerce').dropna()
        if ((arr > 0) & (arr < 100)).mean() > 0.6:
            return col
    raise RuntimeError("Could not detect mass column.")

def detect_band_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ------------------ interpolation ------------------
def build_interpolator(df, age_col, mass_col, band_col):
    # Convert to numeric, coerce errors to NaN
    ages = pd.to_numeric(df[age_col], errors='coerce')
    masses = pd.to_numeric(df[mass_col], errors='coerce')
    band = pd.to_numeric(df[band_col], errors='coerce')

    # Keep only rows with all finite values
    mask = ages.notna() & masses.notna() & band.notna()
    ages, masses, band = ages[mask], masses[mask], band[mask]

    if len(band) < 4:
        return None  # not enough points to interpolate

    pts = np.vstack([ages.values, masses.values]).T
    vals = band.values
    return LinearNDInterpolator(pts, vals, fill_value=np.nan)

# ------------------ mass grid utils ------------------
def make_mass_grid(m_min, m_max, n_mass):
    n_log = max(10, int(n_mass*60//100))
    n_lin = max(2, n_mass - n_log)
    grid = np.unique(np.concatenate([
        np.logspace(np.log10(max(m_min,1e-6)), np.log10(m_max), n_log),
        np.linspace(m_min, m_max, n_lin)
    ]))
    return grid

def eval_on_massgrid(interp, age, mass_grid):
    if interp is None:
        return np.full_like(mass_grid, np.nan, dtype=float)
    pts = np.vstack([np.full_like(mass_grid, age, dtype=float), mass_grid]).T
    return interp(pts)

def eval_at_masses(interp, age, masses):
    """Evaluate interpolator at an arbitrary array of masses (can contain NaN).
    Returns an array of same shape with NaN where mass is NaN or interp is None."""
    masses = np.asarray(masses, dtype=float)
    out = np.full_like(masses, np.nan, dtype=float)
    if interp is None:
        return out
    mask = np.isfinite(masses)
    if not mask.any():
        return out
    pts = np.vstack([np.full(mask.sum(), age, dtype=float), masses[mask]]).T
    vals = interp(pts)
    out[mask] = vals
    return out

# ------------------ find intersection in CMD (V vs V-I) ------------------
def find_cmd_intersection(interpV1, interpI1, interpV2, interpI2, age, mass_grid, mag_decimals, max_V):
    V1 = eval_on_massgrid(interpV1, age, mass_grid)
    I1 = eval_on_massgrid(interpI1, age, mass_grid)
    V2 = eval_on_massgrid(interpV2, age, mass_grid)
    I2 = eval_on_massgrid(interpI2, age, mass_grid)

    mask1 = np.isfinite(V1) & np.isfinite(I1)
    mask2 = np.isfinite(V2) & np.isfinite(I2)
    if not (mask1.any() and mask2.any()):
        return None

    m1 = mass_grid[mask1]; V1v = V1[mask1]; I1v = I1[mask1]
    m2 = mass_grid[mask2]; V2v = V2[mask2]; I2v = I2[mask2]
    CI1 = V1v - I1v
    CI2 = V2v - I2v

    Vmin = max(np.nanmin(V1v), np.nanmin(V2v))
    Vmax = min(np.nanmax(V1v), np.nanmax(V2v))
    if not (np.isfinite(Vmin) and np.isfinite(Vmax) and Vmin < Vmax):
        return None

    rV1 = np.round(V1v, mag_decimals); rCI1 = np.round(CI1, mag_decimals)
    rV2 = np.round(V2v, mag_decimals); rCI2 = np.round(CI2, mag_decimals)

    tuples1 = set(zip(rV1.tolist(), rCI1.tolist()))
    tuples2 = set(zip(rV2.tolist(), rCI2.tolist()))
    common = tuples1.intersection(tuples2)
    if not common:
        return None

    candidates = []
    for (rV, rCI) in common:
        if rV > max_V:
            continue
        idxs1 = np.where((rV1 == rV) & (rCI1 == rCI))[0]
        idxs2 = np.where((rV2 == rV) & (rCI2 == rCI))[0]
        if idxs1.size == 0 or idxs2.size == 0:
            continue
        d1 = np.sqrt((V1v[idxs1] - rV)**2 + (CI1[idxs1] - rCI)**2)
        best1 = idxs1[int(np.argmin(d1))]
        m1_sel = float(m1[best1]); V1_sel = float(V1v[best1]); I1_sel = float(I1v[best1]); CI1_sel = float(CI1[best1])
        d2 = np.sqrt((V2v[idxs2] - rV)**2 + (CI2[idxs2] - rCI)**2)
        best2 = idxs2[int(np.argmin(d2))]
        m2_sel = float(m2[best2]); V2_sel = float(V2v[best2]); I2_sel = float(I2v[best2]); CI2_sel = float(CI2[best2])
        dist_between = np.sqrt((V1_sel - V2_sel)**2 + (CI1_sel - CI2_sel)**2)
        candidates.append({'rV': rV, 'rCI': rCI,
                           'm1': m1_sel, 'V1': V1_sel, 'I1': I1_sel, 'CI1': CI1_sel,
                           'm2': m2_sel, 'V2': V2_sel, 'I2': I2_sel, 'CI2': CI2_sel,
                           'dist': dist_between})
    if not candidates:
        return None
    best = min(candidates, key=lambda c: c['dist'])
    return best

# ------------------ build merged CMD for one age ------------------
def build_merged_for_age(interpV_bhac, interpI_bhac, interpV_parsec, interpI_parsec, interpU_parsec, interpB_parsec, age, mass_grid, intersection):
    # get arrays for both models on their mass grids
    VA = eval_on_massgrid(interpV_bhac, age, mass_grid)
    IA = eval_on_massgrid(interpI_bhac, age, mass_grid)
    VB = eval_on_massgrid(interpV_parsec, age, mass_grid)
    IB = eval_on_massgrid(interpI_parsec, age, mass_grid)

    # Build dataframes for each model with valid points
    dfA = pd.DataFrame({'mass': mass_grid, 'V': VA, 'I': IA})
    dfA = dfA[np.isfinite(dfA['V']) & np.isfinite(dfA['I'])].copy()
    dfA['VminusI'] = dfA['V'] - dfA['I']
    dfA['source'] = 'BHAC15'  # model A

    dfB = pd.DataFrame({'mass': mass_grid, 'V': VB, 'I': IB})
    dfB = dfB[np.isfinite(dfB['V']) & np.isfinite(dfB['I'])].copy()
    dfB['VminusI'] = dfB['V'] - dfB['I']
    dfB['source'] = 'PARSEC'  # model B

    # Prepare U/B columns: initialize to NaN
    dfA['U'] = np.nan; dfA['B'] = np.nan
    dfB['U'] = np.nan; dfB['B'] = np.nan

    if intersection is None:
        # No intersection found: return concatenation but mark it as no-switch (user can decide)
        merged = pd.concat([dfB.assign(origin_order=0), dfA.assign(origin_order=1)], ignore_index=True)
        merged.sort_values(['VminusI','V'], inplace=True)
        merged.reset_index(drop=True, inplace=True)
        # fill U/B for all rows that have a finite mass using PARSEC interpolators (if available)
        mask_mass = merged['mass'].notna()
        if mask_mass.any():
            if interpU_parsec is not None:
                merged.loc[mask_mass, 'U'] = eval_at_masses(interpU_parsec, age, merged.loc[mask_mass, 'mass'].values)
            if interpB_parsec is not None:
                merged.loc[mask_mass, 'B'] = eval_at_masses(interpB_parsec, age, merged.loc[mask_mass, 'mass'].values)
        return merged, None

    # when intersection found: keep V/I selection as before
    V_avg = 0.5*(intersection['V1'] + intersection['V2'])

    # choose from PARSEC (model B) those points with V <= V_avg (brighter or equal)
    dfB_sel = dfB[dfB['V'] <= V_avg].copy()
    # choose from BHAC15 (model A) those points with V >= V_avg (fainter or equal)
    dfA_sel = dfA[dfA['V'] >= V_avg].copy()

    # include the matched points explicitly (mass from best candidates) to ensure continuity
    df_matchA = pd.DataFrame([{'mass': intersection['m1'], 'V': intersection['V1'], 'I': intersection['I1'], 'VminusI': intersection['CI1'], 'source': 'BHAC15_match', 'U': np.nan, 'B': np.nan}])
    df_matchB = pd.DataFrame([{'mass': intersection['m2'], 'V': intersection['V2'], 'I': intersection['I2'], 'VminusI': intersection['CI2'], 'source': 'PARSEC_match', 'U': np.nan, 'B': np.nan}])
    # include midpoint averaged point for reference as well (U/B will NOT be midpoint-filled)
    df_mid = pd.DataFrame([{'mass': np.nan, 'V': 0.5*(intersection['V1']+intersection['V2']), 'I': 0.5*(intersection['I1']+intersection['I2']), 'VminusI': 0.5*(intersection['CI1']+intersection['CI2']), 'source': 'midpoint', 'U': np.nan, 'B': np.nan}])

    merged = pd.concat([dfB_sel, df_matchB, df_mid, df_matchA, dfA_sel], ignore_index=True)
    # drop duplicates extremely close in (V,V-I) to avoid double rows
    merged['V_round'] = np.round(merged['V'], 6)
    merged['VI_round'] = np.round(merged['VminusI'], 6)
    merged = merged.drop_duplicates(subset=['V_round','VI_round'])
    merged.drop(columns=['V_round','VI_round'], inplace=True)

    # Fill U/B for ALL rows with finite mass using PARSEC interpolators (so BHAC-derived rows get U/B from PARSEC)
    mask_mass = merged['mass'].notna()
    if mask_mass.any():
        if interpU_parsec is not None:
            merged.loc[mask_mass, 'U'] = eval_at_masses(interpU_parsec, age, merged.loc[mask_mass, 'mass'].values)
        if interpB_parsec is not None:
            merged.loc[mask_mass, 'B'] = eval_at_masses(interpB_parsec, age, merged.loc[mask_mass, 'mass'].values)

    # sort by color (V-I) then V to make a clean CMD-track
    merged.sort_values(['VminusI','V'], inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged, V_avg

# ------------------ plotting verification ------------------
def plot_merged(outpng, dfA, dfB, merged_df, intersection, age):
    plt.figure(figsize=(6,7))
    # plot PARSEC (B)
    plt.plot(dfB['VminusI'], dfB['V'], label='PARSEC', color='tab:orange')
    # plot BHAC15 (A)
    plt.plot(dfA['VminusI'], dfA['V'], label='BHAC15', color='tab:blue')
    # plot merged
    plt.plot(merged_df['VminusI'], merged_df['V'], label='merged', color='tab:green', linewidth=1.5)
    # highlight intersection
    if intersection is not None:
        plt.scatter([intersection['CI1']], [intersection['V1']], color='tab:blue', marker='x', s=50, label='BHAC match')
        plt.scatter([intersection['CI2']], [intersection['V2']], color='tab:orange', marker='x', s=50, label='PARSEC match')
        xm = 0.5*(intersection['CI1'] + intersection['CI2']); ym = 0.5*(intersection['V1'] + intersection['V2'])
        plt.scatter([xm], [ym], color='k', marker='o', s=40, label='midpoint')
        plt.text(xm, ym, f' mA={intersection["m1"]:.3f}, mB={intersection["m2"]:.3f}', fontsize=8, va='bottom', ha='center')
    plt.gca().invert_yaxis()
    plt.xlabel('V - I')
    plt.ylabel('V [mag]')
    plt.title(f'Merged isochrone at logAge={age:.2f}')
    plt.legend(loc='best', fontsize='small')
    plt.tight_layout()
    plt.savefig(outpng, dpi=200)
    plt.close()

# ------------------ main ------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file_bhac', type=Path, required=True, help='BHAC15 isochrone (used for faint end)')
    p.add_argument('--file_parsec', type=Path, required=True, help='PARSEC isochrone (used for bright end)')
    p.add_argument('--outdir', type=Path, default=Path('merged_iso'))
    p.add_argument('--mag-decimals', type=int, default=1)
    p.add_argument('--max-V', type=float, default=8.0)
    p.add_argument('--n-mass', type=int, default=5000)
    p.add_argument('--mass-range', type=float, nargs=2, default=(0.01, 10.0))
    args = p.parse_args()

    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    plotsdir = outdir / 'plots'; plotsdir.mkdir(parents=True, exist_ok=True)

    df_bhac = read_isochrone_with_header(args.file_bhac)
    df_parsec = read_isochrone_with_header(args.file_parsec)

    age_col_b = detect_age_column(df_bhac); mass_col_b = detect_mass_column(df_bhac)
    age_col_p = detect_age_column(df_parsec); mass_col_p = detect_mass_column(df_parsec)

    V_candidates = ['Vmag','V','Mv']
    I_candidates = ['Imag','I','Mi']
    U_candidates = ['Umag','U','Mu']
    B_candidates = ['Bmag','B','Mb']
    Vcol_b = detect_band_column(df_bhac, V_candidates)
    Vcol_p = detect_band_column(df_parsec, V_candidates)
    Icol_b = detect_band_column(df_bhac, I_candidates)
    Icol_p = detect_band_column(df_parsec, I_candidates)

    # detect U/B only in PARSEC (we will use U/B only from PARSEC)
    Ucol_p = detect_band_column(df_parsec, U_candidates)
    Bcol_p = detect_band_column(df_parsec, B_candidates)

    if Vcol_b is None or Vcol_p is None or Icol_b is None or Icol_p is None:
        raise RuntimeError("Both files must contain V and I bands (one of Vmag/Mv and Imag/Mi). Missing check columns.")

    interpV_b = build_interpolator(df_bhac, age_col_b, mass_col_b, Vcol_b)
    interpI_b = build_interpolator(df_bhac, age_col_b, mass_col_b, Icol_b)
    interpV_p = build_interpolator(df_parsec, age_col_p, mass_col_p, Vcol_p)
    interpI_p = build_interpolator(df_parsec, age_col_p, mass_col_p, Icol_p)

    # build U/B interpolators for PARSEC only (may be None if missing)
    interpU_p = build_interpolator(df_parsec, age_col_p, mass_col_p, Ucol_p) if Ucol_p is not None else None
    interpB_p = build_interpolator(df_parsec, age_col_p, mass_col_p, Bcol_p) if Bcol_p is not None else None

    if any(x is None for x in [interpV_b, interpI_b, interpV_p, interpI_p]):
        raise RuntimeError("Could not build interpolators for V/I in one or both files (too few valid points).")

    mass_grid = make_mass_grid(args.mass_range[0], args.mass_range[1], args.n_mass)

    ages = np.arange(6.5, 10.0001, 0.1)
    summary_rows = []
    for age in ages:
        inter = find_cmd_intersection(interpV_b, interpI_b, interpV_p, interpI_p, age, mass_grid, args.mag_decimals, args.max_V)
        if inter is None:
            print(f"[warn] age {age:.2f}: no intersection found (rounded to {args.mag_decimals} decimals) within max-V {args.max_V}. Merging will be a conservative concat.")
        merged_df, V_avg = build_merged_for_age(interpV_b, interpI_b, interpV_p, interpI_p, interpU_p, interpB_p, age, mass_grid, inter)
        # save merged csv
        fname = outdir / f'merged_iso_age_{age:.2f}.csv'
        # ensure column order includes U and B
        cols = ['mass','V','I','VminusI','U','B','source']
        # there may be other columns; select intersection with available
        cols_available = [c for c in cols if c in merged_df.columns]
        merged_df.to_csv(fname, index=False, columns=cols_available)
        # for plotting we need originals arrays to show them
        # re-evaluate full arrays (for plotting)
        VA = eval_on_massgrid(interpV_b, age, mass_grid)
        IA = eval_on_massgrid(interpI_b, age, mass_grid)
        dfA_plot = pd.DataFrame({'mass': mass_grid, 'V': VA, 'I': IA})
        dfA_plot = dfA_plot[np.isfinite(dfA_plot['V']) & np.isfinite(dfA_plot['I'])].copy()
        dfA_plot['VminusI'] = dfA_plot['V'] - dfA_plot['I']

        VB = eval_on_massgrid(interpV_p, age, mass_grid)
        IB = eval_on_massgrid(interpI_p, age, mass_grid)
        dfB_plot = pd.DataFrame({'mass': mass_grid, 'V': VB, 'I': IB})
        dfB_plot = dfB_plot[np.isfinite(dfB_plot['V']) & np.isfinite(dfB_plot['I'])].copy()
        dfB_plot['VminusI'] = dfB_plot['V'] - dfB_plot['I']

        outpng = plotsdir / f'merged_age_{age:.2f}.png'
        try:
            plot_merged(outpng, dfA_plot, dfB_plot, merged_df, inter, age)
        except Exception as e:
            print(f"[warn] plotting failed for age {age:.2f}: {e}")

        summary_rows.append({
            'age_log10': age,
            'found_intersection': (inter is not None),
            'rV': (inter['rV'] if inter is not None else np.nan),
            'rCI': (inter['rCI'] if inter is not None else np.nan),
            'm_bhac': (inter['m1'] if inter is not None else np.nan),
            'm_parsec': (inter['m2'] if inter is not None else np.nan),
            'dist': (inter['dist'] if inter is not None else np.nan),
            'V_avg_threshold': (0.5*(inter['V1']+inter['V2']) if inter is not None else np.nan)
        })
        print(f"Saved merged for age {age:.2f} -> {fname}")

    pd.DataFrame(summary_rows).to_csv(outdir / 'merge_summary.csv', index=False)
    print("All done. Merged CSVs in:", outdir, "Plots in:", plotsdir)

if __name__ == '__main__':
    main()
