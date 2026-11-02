#!/usr/bin/env python3
"""
dsp_framework.py -- reproducible calculations for
"A Portable Area-Equivalent Framework for Evaluating Enhanced FPGA DSP Blocks ..."

Core functions use only the Python standard library.  Plotting (--figs) needs matplotlib.

Usage
-----
    python dsp_framework.py              # print all tables
    python dsp_framework.py --selftest   # check results against the values printed in the paper
    python dsp_framework.py --csv DIR    # also write the tables as CSV files into DIR
    python dsp_framework.py --figs DIR   # also draw the sensitivity figure into DIR (matplotlib)

Notation (same as the paper)
----------------------------
    M_peak, M_eff : MACs per DSP per cycle (architectural peak / achieved by the mapped workload)
    U             : utilization = M_eff / M_peak                                        Eq. (1)
    G             : gain = (M_peak,enh/M_peak,std) * (U_enh/U_std)                      Eq. (2)
    D             : density = (M_eff f / A) / (M_eff,std f_std / A_std)                 Eq. (3)
    C             : area-equivalent cost = a*N_DSP + lam*N_LUT, lam = 1/K               Eq. (4)
    lam*          : break-even lam                                                      Eq. (5)
    phi           : frequency ratio f_enh / f_std
    phi_crit, a_crit, s_star, G_min, U_min : critical values (Eqs. 6-10 of the paper)

Every input below is a value published in the cited source; nothing is measured here.
"""
import argparse
import csv
import math
import os

# ----------------------------------------------------------------------------------------
# 1. LUT-to-DSP area ratio K = 1/lam (LUT-equivalents with the area of one standard DSP)
# ----------------------------------------------------------------------------------------
# Wong, Betz, Rose, FPGA 2011, Table 2 (Stratix III, routing included in tile areas):
#   LAB = 1 (0.0221 mm^2), ALUT (half ALM) = 0.05 LAB, DSP block = 11.9 LAB (0.2623 mm^2), 10 ALMs per LAB
K_HIGH = 11.9 / 0.10                  # 119 ALMs per DSP block (high estimate: block is larger than a DSP48E2)
# Ma et al. Table 2: synthesized DSP48E2 core 25,107 um^2 (TSMC 65 nm) / one ALM tile 0.10*0.0221 mm^2 = 2,210 um^2
K_LOW = 25107.0 / 2210.0              # about 11.4 (excludes the DSP routing interface -> lower bound)
K_ALUT = 2 * K_HIGH                   # 238: same block when the LUT-equivalent is an ALUT (half ALM)
K_SET = [("K_low", K_LOW), ("K_high", K_HIGH), ("K_ALUT", K_ALUT)]

# ----------------------------------------------------------------------------------------
# 2. Published inputs
# ----------------------------------------------------------------------------------------
# --- INT8-type designs: (name, M_peak, M_eff for 1x1x16 / 3x3 / 5x5, A, f [MHz]) from Ma et al. Tables 1 and 3
INT8 = {
    "Standard DSP48E2": dict(peak=2, meff=[1.75, 1.75, 1.75], A=1.00, f=667.0),
    "PIR-DSP-D0":       dict(peak=6, meff=[32 / 6, 18 / 3, 50 / 9], A=1.19, f=806.0),
    "APIR-DSP":         dict(peak=6, meff=[32 / 6, 18 / 3, 50 / 9], A=0.94, f=847.5),
    "Ma #0 (INT8)":     dict(peak=4, meff=[32 / 8, 18 / 5, 50 / 13], A=1.10, f=658.0),
    "Ma #1 (+FP->FX)":  dict(peak=4, meff=[32 / 8, 18 / 5, 50 / 13], A=1.11, f=645.0),
}
KERNELS = ["1x1x16", "3x3", "5x5"]

# --- Cost cases: counts per output (divide raw counts by the number of outputs D)
#     (label, source, N_DSP std, N_LUT std, N_DSP enh, N_LUT enh, a_enh, phi)
def per_output(n_dsp, n_lut, d):
    return n_dsp / d, n_lut / d

PHI_MA = 645.0 / 667.0                # Ma et al. design #1 vs standard DSP48E2
PHI_MEK = 1.0                         # Mekhemer et al.: 482 vs 483 MHz, taken as 1.00 (0.2 % difference)
A_MA, A_MEK = 1.11, 1.36

def _mek(fmt, std, enh):
    ds, ls = per_output(std[0], std[1], std[2])
    de, le = per_output(enh[0], enh[1], enh[2])
    return (f"[8] {fmt}", "Mekhemer et al. 2026, Table V", ds, ls, de, le, A_MEK, PHI_MEK)

COST_CASES = [
    ("[1] E3M3 core, area only",   "Ma et al. Sec. 4.3", 10, 872, 10, 320, A_MA, 1.0),
    ("[1] E3M3 core, area + f",    "Ma et al. Sec. 4.3", 10, 872, 10, 320, A_MA, PHI_MA),
    ("[1] mixed, area only",       "Ma et al. Sec. 4.4", 14, 496, 8, 160, A_MA, 1.0),
    ("[1] mixed, area + f",        "Ma et al. Sec. 4.4", 14, 496, 8, 160, A_MA, PHI_MA),
    _mek("E4M3", (16, 4841, 2), (4, 210, 2)),
    _mek("E3M2", (16, 4153, 3), (3, 96, 2)),
    _mek("E2M3", (4, 420, 2), (3, 96, 2)),
    _mek("E2M1", (4, 198, 2), (2, 39, 2)),
]
# Reported gains used for the density break-even test (G_reported)
REPORTED_G = {
    "Ma #1 (+FP->FX)": dict(G=[2.29, 2.06, 2.20], a=1.11, phi=PHI_MA),
    "Boutros 8-bit (application level)": dict(G=[1.32], a=1.12, phi=1.0),
    "Boutros 4-bit (application level)": dict(G=[1.6], a=1.12, phi=1.0),
    "Mekhemer E4M3": dict(G=[12.4 / 3.4], a=A_MEK, phi=PHI_MEK),
    "Mekhemer E3M2": dict(G=[16.5 / 6.5], a=A_MEK, phi=PHI_MEK),
    "Mekhemer E2M3": dict(G=[16.5 / 12.4], a=A_MEK, phi=PHI_MEK),
    "Mekhemer E2M1": dict(G=[24.8 / 12.4], a=A_MEK, phi=PHI_MEK),
}

# ----------------------------------------------------------------------------------------
# 3. Metric functions
# ----------------------------------------------------------------------------------------
def utilization(m_eff, m_peak):                                   # Eq. (1)
    return m_eff / m_peak

def gain_decomposition(peak_enh, peak_std, u_enh, u_std):         # Eq. (2)
    peak_ratio, u_ratio = peak_enh / peak_std, u_enh / u_std
    return peak_ratio * u_ratio, peak_ratio, u_ratio

def density(m_eff, f, A, m_eff_std, f_std, A_std=1.0):            # Eq. (3)
    return (m_eff * f / A) / (m_eff_std * f_std / A_std)

def cost(a, n_dsp, lam, n_lut):                                   # Eq. (4)
    return a * n_dsp + lam * n_lut

def cost_ratio(case, lam, phi=None):
    """C_std / (C_enh / phi): > 1 means the enhanced block is cost-effective."""
    _, _, ds, ls, de, le, a_e, phi0 = case
    phi = phi0 if phi is None else phi
    return cost(1.0, ds, lam, ls) / (cost(a_e, de, lam, le) / phi)

def lam_star(case):                                               # Eq. (5) with the frequency factor
    _, _, ds, ls, de, le, a_e, phi = case
    num, den = a_e * de / phi - ds, ls - le / phi
    if num <= 0 and den >= 0:
        return 0.0                                                # favored for every lam >= 0
    if num > 0 and den > 0:
        return num / den
    return math.inf                                               # never cost-effective

def phi_crit(case, lam):                                          # Eq. (6): cost-effective iff phi > phi_crit
    _, _, ds, ls, de, le, a_e, _ = case
    return (a_e * de + lam * le) / (ds + lam * ls)

def a_crit(case, lam, phi=None):                                  # Eq. (7): cost-effective iff a < a_crit
    _, _, ds, ls, de, le, _, phi0 = case
    phi = phi0 if phi is None else phi
    return (phi * (ds + lam * ls) - lam * le) / de

def s_star(case, lam, phi=None):                                  # Eq. (8): required fraction of the reported LUT saving
    _, _, ds, ls, de, le, a_e, phi0 = case
    phi = phi0 if phi is None else phi
    dl = ls - le
    if dl <= 0:
        return math.inf
    return max(0.0, (a_e * de + lam * ls - phi * (ds + lam * ls)) / (lam * dl))

def g_min(a, phi):                                                # Eq. (9): density break-even gain
    return a / phi

def u_min(a, phi, u_std, peak_ratio):                             # Eq. (10): density break-even utilization
    return u_std * g_min(a, phi) / peak_ratio

# ----------------------------------------------------------------------------------------
# 4. Tables
# ----------------------------------------------------------------------------------------
def table_density():
    std = INT8["Standard DSP48E2"]
    rows = []
    for name, d in INT8.items():
        u = [utilization(m, d["peak"]) for m in d["meff"]]
        dd = [density(m, d["f"], d["A"], std["meff"][i], std["f"], std["A"]) for i, m in enumerate(d["meff"])]
        rows.append((name, u, dd))
    return rows

def table_gain():
    std = INT8["Standard DSP48E2"]
    out = []
    for name in ("Ma #0 (INT8)", "Ma #1 (+FP->FX)"):
        d = INT8[name]
        for i, k in enumerate(KERNELS):
            g, pr, ur = gain_decomposition(d["peak"], std["peak"], d["meff"][i] / d["peak"], std["meff"][i] / std["peak"])
            out.append((name, k, g, pr, ur))
    return out

def table_cost():
    rows = []
    for case in COST_CASES:
        ls = lam_star(case)
        kstar = (1 / ls) if (ls not in (0.0, math.inf)) else ls
        ratios = {k: cost_ratio(case, 1 / v) for k, v in K_SET}
        rows.append((case[0], ls, kstar, ratios))
    return rows

def table_critical():
    """phi_crit, a_crit, s* at each K for every cost case (frequency / area / mapping robustness)."""
    rows = []
    for case in COST_CASES:
        r = {"case": case[0], "phi": case[7], "a": case[6]}
        for kname, k in K_SET:
            lam = 1 / k
            r[kname] = dict(phi_crit=phi_crit(case, lam), a_crit=a_crit(case, lam), s_star=s_star(case, lam))
        rows.append(r)
    return rows

def table_gmin():
    rows = []
    std = INT8["Standard DSP48E2"]
    for name, d in REPORTED_G.items():
        gm = g_min(d["a"], d["phi"])
        rows.append((name, d["a"], d["phi"], gm, d["G"]))
    # utilization threshold for Ma #1
    um = u_min(1.11, PHI_MA, std["meff"][0] / std["peak"], 4 / 2)
    return rows, um

def table_density_vs_u_phi(u_list=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), phi_list=(1.0, 0.967, 0.9, 0.8)):
    """D of Ma #1-type block (peak 4, A = 1.11, baseline 1.75 MACs/DSP) versus achieved utilization and frequency ratio."""
    out = []
    for u in u_list:
        out.append((u, [((4 * u) * (phi * 667) / 1.11) / (1.75 * 667) for phi in phi_list]))
    return out, phi_list

def grid_K_phi(case, k_values, phi_values):
    return [[cost_ratio(case, 1 / k, phi) for k in k_values] for phi in phi_values]

# ----------------------------------------------------------------------------------------
# 5. Output helpers
# ----------------------------------------------------------------------------------------
def fmt(x, nd=2):
    if x == math.inf:
        return "never"
    if x == 0.0:
        return "always"
    return f"{x:.{nd}f}"

def print_all():
    print(f"K bracket: K_low = {K_LOW:.2f}, K_high = {K_HIGH:.1f}, K_ALUT = {K_ALUT:.1f}\n")
    print("Table IV: utilization U and density D (1x1x16 | 3x3 | 5x5)")
    for name, u, d in table_density():
        print(f"  {name:18s} U = " + " | ".join(f"{100*x:5.1f}%" for x in u) + "   D = " + " | ".join(f"{x:5.2f}" for x in d))
    print("\nGain decomposition G = peak ratio x U ratio")
    for name, k, g, pr, ur in table_gain():
        print(f"  {name:16s} {k:7s} G = {g:.2f} = {pr:.2f} x {ur:.2f}")
    print("\nTable V: unified cost (ratio C_std / C_enh,f at K_low, K_high, K_ALUT) and break-even")
    for name, ls, ks, ratios in table_cost():
        kstr = fmt(ks, 0) if isinstance(ks, float) and ks not in (0.0, math.inf) else fmt(ks)
        print(f"  {name:26s} K* = {kstr:>6s}  " + "  ".join(f"{k}: {v:5.2f}" for k, v in ratios.items()))
    print("\nTable VI: critical values  (phi_crit: lowest frequency ratio, a_crit: highest DSP area ratio, s*: needed LUT-saving fraction)")
    for r in table_critical():
        line = f"  {r['case']:26s} (phi={r['phi']:.3f}, a={r['a']:.2f})"
        for kname, _ in K_SET:
            v = r[kname]
            line += f" | {kname}: phi_c={v['phi_crit']:.2f} a_c={v['a_crit']:.2f} s*={fmt(v['s_star'])}"
        print(line)
    rows, um = table_gmin()
    print("\nDensity break-even gain G_min = a/phi versus the reported gain")
    for name, a, phi, gm, g in rows:
        print(f"  {name:36s} a={a:.2f} phi={phi:.3f} G_min={gm:.3f}  reported G = {', '.join(f'{x:.2f}' for x in g)}")
    print(f"  Minimum utilization of the Ma #1 block for D >= 1: {100*um:.1f}% of its 4-MAC peak")
    t, plist = table_density_vs_u_phi()
    print("\nTable VII: D of the Ma #1-type block versus utilization U_enh (rows) and frequency ratio phi (columns " + ", ".join(map(str, plist)) + ")")
    for u, ds in t:
        print(f"  U = {u:.1f}: " + "  ".join(f"{d:5.2f}" for d in ds))

def selftest():
    """Assert the values quoted in the paper."""
    def near(a, b, tol):
        assert abs(a - b) <= tol, (a, b)
    near(K_LOW, 11.36, 0.01); near(K_HIGH, 119.0, 1e-9)
    d = {n: (u, dd) for n, u, dd in table_density()}
    near(d["Ma #1 (+FP->FX)"][1][0], 1.99, 0.005); near(d["Ma #1 (+FP->FX)"][1][1], 1.79, 0.005); near(d["Ma #1 (+FP->FX)"][1][2], 1.91, 0.005)
    near(d["PIR-DSP-D0"][1][1], 3.48, 0.005); near(d["APIR-DSP"][1][1], 4.63, 0.005)
    cc = {c[0]: c for c in COST_CASES}
    near(1 / lam_star(cc["[1] E3M3 core, area only"]), 502, 1); near(1 / lam_star(cc["[1] E3M3 core, area + f"]), 366, 1)
    near(1 / lam_star(cc["[8] E2M3"]), 4050, 5)
    assert lam_star(cc["[1] mixed, area only"]) == 0.0 and lam_star(cc["[8] E4M3"]) == 0.0
    near(cost_ratio(cc["[1] E3M3 core, area only"], 1 / K_HIGH), 1.26, 0.005)
    near(cost_ratio(cc["[1] E3M3 core, area + f"], 1 / K_HIGH), 1.22, 0.005)
    near(cost_ratio(cc["[1] E3M3 core, area + f"], 1 / K_ALUT), 1.06, 0.005)
    near(cost_ratio(cc["[8] E4M3"], 1 / K_HIGH), 7.87, 0.01)
    c1 = cc["[1] E3M3 core, area + f"]
    near(phi_crit(c1, 1 / K_HIGH), 0.80, 0.005); near(a_crit(c1, 1 / K_HIGH), 1.41, 0.005); near(s_star(c1, 1 / K_HIGH), 0.36, 0.005)
    near(phi_crit(c1, 1 / K_ALUT), 0.91, 0.005); near(a_crit(c1, 1 / K_ALUT), 1.19, 0.005); near(s_star(c1, 1 / K_ALUT), 0.67, 0.005)
    near(u_min(1.11, PHI_MA, 0.875, 2.0), 0.502, 0.001)
    print("selftest: all values match the paper")

def write_csv(directory):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "table_cost.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "lambda_star", "K_star"] + [f"ratio_{k}" for k, _ in K_SET])
        for name, ls, ks, ratios in table_cost():
            w.writerow([name, ls, ks] + [round(ratios[k], 4) for k, _ in K_SET])
    with open(os.path.join(directory, "table_critical.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["case", "K", "phi_crit", "a_crit", "s_star"])
        for r in table_critical():
            for kname, k in K_SET:
                v = r[kname]; w.writerow([r["case"], round(k, 2), round(v["phi_crit"], 4), round(v["a_crit"], 4), v["s_star"]])
    with open(os.path.join(directory, "table_density.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["design"] + [f"U_{k}" for k in KERNELS] + [f"D_{k}" for k in KERNELS])
        for name, u, dd in table_density():
            w.writerow([name] + [round(x, 4) for x in u] + [round(x, 4) for x in dd])
    print(f"CSV files written to {directory}")

def draw_figs(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    os.makedirs(directory, exist_ok=True)
    cc = {c[0]: c for c in COST_CASES}
    a_case, b_case = cc["[1] E3M3 core, area only"], cc["[8] E2M3"]
    ks = np.logspace(math.log10(5), math.log10(2000), 120)
    phis = np.linspace(0.5, 1.0, 101)
    fig = plt.figure(figsize=(11.5, 3.9))
    gs = GridSpec(1, 5, width_ratios=[1, 1, 0.05, 0.42, 1.1], wspace=0.30, figure=fig)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[4])]
    cax = fig.add_subplot(gs[2])
    for ax, case, ttl in [(axs[0], a_case, "(a) [1] E3M3 core"), (axs[1], b_case, "(b) [8] E2M3")]:
        z = np.array([[cost_ratio(case, 1 / k, p) for k in ks] for p in phis])
        im = ax.pcolormesh(ks, phis, np.log10(z), cmap="RdYlGn", vmin=-0.6, vmax=0.6, shading="auto")
        ax.contour(ks, phis, z, levels=[1.0], colors="k", linewidths=1.6)
        ax.axvspan(K_LOW, K_HIGH, facecolor="none", edgecolor="k", hatch="//", alpha=0.35, linewidth=0)
        ax.set_xscale("log"); ax.set_xlabel("K = LUT-equivalents per DSP"); ax.set_title(ttl, fontsize=10)
    axs[0].set_ylabel(r"frequency ratio $\varphi=f_{enh}/f_{std}$")
    axs[1].set_yticklabels([])
    axs[0].plot([5, 2000], [PHI_MA] * 2, "b--", lw=1)
    cb = fig.colorbar(im, cax=cax); cb.set_label(r"$\log_{10}(C_{std}/C_{enh})$", fontsize=9)
    for case, lab, phi, col in [(a_case, "[1] E3M3 core", PHI_MA, "tab:blue"), (b_case, "[8] E2M3", 1.0, "tab:red")]:
        axs[2].semilogx(ks, [min(1.0, s_star(case, 1 / k, phi)) for k in ks], color=col, label=f"{lab} ($\\varphi$ = {phi:.2f})")
    axs[2].axvspan(K_LOW, K_HIGH, color="#FFF2CC", alpha=0.9)
    axs[2].set_xlabel("K = LUT-equivalents per DSP"); axs[2].set_ylabel(r"required LUT-saving fraction $s^{*}$")
    axs[2].set_ylim(0, 1.02); axs[2].set_title("(c) mapping robustness", fontsize=10); axs[2].legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(directory, "fig_sensitivity.png"), dpi=220, bbox_inches="tight")
    print(f"figure written to {directory}")


def draw_letter_figs(directory):
    """Two column-width figures for the 3-4 page letter version."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.size": 7, "axes.linewidth": 0.6, "mathtext.fontset": "stix", "font.family": "STIXGeneral"})
    os.makedirs(directory, exist_ok=True)
    cc = {c[0]: c for c in COST_CASES}
    ks = np.logspace(math.log10(5), math.log10(5000), 300)
    # ---- Fig. 1: cost ratio versus K
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    ax.axvspan(K_LOW, K_HIGH, color="#FFF2CC", alpha=0.95, lw=0, label="K = 11-119")
    for key, lab, col, ls in [("[1] E3M3 core, area + f", "[1] E3M3 core", "#2F5597", "-"),
                              ("[1] mixed, area + f", "[1] mixed", "#548235", "-"),
                              ("[8] E4M3", "[8] E4M3", "#ED7D31", "-"),
                              ("[8] E2M3", "[8] E2M3", "#1B9AAA", "-")]:
        ax.loglog(ks, [cost_ratio(cc[key], 1 / k) for k in ks], color=col, ls=ls, lw=1.2, label=lab)
    ax.axhline(1, color="k", lw=0.7)
    ax.axvline(K_ALUT, color="0.5", lw=0.6, ls=":")
    ks_pts = [(1 / lam_star(cc["[1] E3M3 core, area + f"])), (1 / lam_star(cc["[8] E2M3"]))]
    ax.scatter(ks_pts, [1, 1], color="k", s=8, zorder=5)
    ax.text(ks_pts[0] * 1.08, 0.66, "$K^{*}$=366", fontsize=6.5); ax.text(ks_pts[1] * 0.42, 0.66, "$K^{*}$=4050", fontsize=6.5)
    ax.set_xlabel("$K$ = LUT-equivalents per DSP"); ax.set_ylabel("$C_{std}/(C_{enh}/\\varphi)$")
    ax.set_xlim(5, 5000); ax.set_ylim(0.6, 30)
    ax.set_yticks([0.6, 1, 2, 5, 10, 30]); ax.set_yticklabels(["0.6", "1", "2", "5", "10", "30"])
    ax.legend(fontsize=6, loc="upper right", frameon=False, ncol=1, handlelength=1.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(os.path.join(directory, "letter_fig1.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # ---- Fig. 2: (a) K-phi map and (b) required LUT-saving fraction
    a_case, b_case = cc["[1] E3M3 core, area only"], cc["[8] E2M3"]
    phis = np.linspace(0.5, 1.0, 101)
    kk = np.logspace(math.log10(5), math.log10(2000), 200)
    fig = plt.figure(figsize=(3.3, 4.3))
    gs = GridSpec(2, 2, height_ratios=[1.15, 1], width_ratios=[1, 0.035], hspace=0.42, wspace=0.05, figure=fig)
    ax1, cax, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    z = np.array([[cost_ratio(a_case, 1 / k, p) for k in kk] for p in phis])
    im = ax1.pcolormesh(kk, phis, np.log10(z), cmap="RdYlGn", vmin=-0.6, vmax=0.6, shading="auto")
    ax1.contour(kk, phis, z, levels=[1.0], colors="k", linewidths=1.0)
    ax1.axvspan(K_LOW, K_HIGH, facecolor="none", edgecolor="k", hatch="//", alpha=0.3, linewidth=0)
    ax1.axhline(PHI_MA, color="b", lw=0.8, ls="--")
    ax1.set_xscale("log"); ax1.set_xlabel("$K$"); ax1.set_ylabel("frequency ratio $\\varphi$")
    ax1.set_title("(a) [1] E3M3 core: $\\log_{10}$ cost ratio", fontsize=7)
    fig.colorbar(im, cax=cax)
    for case, lab, phi, col in [(cc["[1] E3M3 core, area + f"], "[1] E3M3 core ($\\varphi$=0.967)", PHI_MA, "#2F5597"), (b_case, "[8] E2M3 ($\\varphi$=1.00)", 1.0, "#C00000")]:
        ax2.semilogx(kk, [min(1.0, s_star(case, 1 / k, phi)) for k in kk], color=col, lw=1.2, label=lab)
    ax2.axvspan(K_LOW, K_HIGH, color="#FFF2CC", alpha=0.95, lw=0)
    ax2.set_xlabel("$K$ = LUT-equivalents per DSP"); ax2.set_ylabel("required fraction $s^{*}$")
    ax2.set_ylim(0, 1.02); ax2.set_title("(b) realized LUT-saving needed", fontsize=7)
    ax2.legend(fontsize=6, loc="upper left", frameon=False)
    for a in (ax1, ax2): a.spines[["top", "right"]].set_visible(False) if a is ax2 else None
    fig.savefig(os.path.join(directory, "letter_fig2.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"letter figures written to {directory}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv", metavar="DIR")
    ap.add_argument("--figs", metavar="DIR")
    args = ap.parse_args()
    print_all()
    if args.selftest:
        selftest()
    if args.csv:
        write_csv(args.csv)
    if args.figs:
        draw_figs(args.figs)
        draw_letter_figs(args.figs)
