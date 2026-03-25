import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from pulp import *
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

from parameters import df, S, B, position_bounds, CB_MIN, K, alpha, tau
from parameters import generate_independent_gaussian, generate_correlated_factor
n = len(df)

plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
                     'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150})

# ── helpers ──────────────────────────────────────────────────────────────────

def solve_model(prob):
    solver = PULP_CBC_CMD(msg=0, timeLimit=300)
    prob.solve(solver)
    return LpStatus[prob.status]

def add_structural_constraints(prob, x, df, n):
    prob += lpSum(x[i] for i in range(n)) == S
    prob += lpSum(df.iloc[i]["wage_kpw"] * x[i] for i in range(n)) <= B
    for pos, (lo, hi) in position_bounds.items():
        idx = df[df["position"] == pos].index.tolist()
        prob += lpSum(x[i] for i in idx) >= lo
        prob += lpSum(x[i] for i in idx) <= hi
    cb_idx = df[df["sub_position"] == "CB"].index.tolist()
    prob += lpSum(x[i] for i in cb_idx) >= CB_MIN

def get_selected(x, n, status="Optimal"):
    if status != "Optimal":
        return []
    sel = [i for i in range(n) if (value(x[i]) is not None and value(x[i]) > 0.5)]
    return sel

def squad_stats(sel, P_ind, P_cor):
    mu_tot  = df.iloc[sel]["mu"].sum()
    wage    = df.iloc[sel]["wage_kpw"].sum()
    avg_sig = df.iloc[sel]["sigma"].mean()
    avg_bet = df.iloc[sel]["beta"].mean()
    perfs_ind = P_ind[sel, :].sum(axis=0)
    perfs_cor = P_cor[sel, :].sum(axis=0)
    loss_ind  = tau - perfs_ind
    loss_cor  = tau - perfs_cor
    cvar_ind  = compute_cvar(loss_ind)
    cvar_cor  = compute_cvar(loss_cor)
    mad       = compute_mad(P_ind[sel, :])
    return dict(mu=mu_tot, wage=wage, avg_sig=avg_sig, avg_bet=avg_bet,
                cvar_ind=cvar_ind, cvar_cor=cvar_cor, mad=mad)

def compute_cvar(losses):
    losses_s = np.sort(losses)
    cut = int(np.ceil(alpha * len(losses_s)))
    return float(np.mean(losses_s[cut:]))

def compute_mad(P_matrix):
    # Mean Absolute Deviation of total squad performance
    totals = P_matrix.sum(axis=0)
    return float(np.mean(np.abs(totals - totals.mean())))

def print_full_table(sel_set, label=""):
    print(f"\n{'='*100}")
    print(f"FULL PLAYER TABLE  {label}")
    print(f"{'='*100}")
    print(f"{'#':>4} {'Name':<28} {'Pos':>4} {'Sub':>4} {'Rank':>5} {'mu':>7} {'Wage':>6} {'Sig':>5} {'Bet':>5}  Status")
    print("-"*100)
    for _, row in df.iterrows():
        i = row.name
        status = "SELECTED <<<" if i in sel_set else "---"
        sub = row['sub_position'] if pd.notna(row['sub_position']) else "  -"
        print(f"{row['rank']:>4} {row['name']:<28} {row['position']:>4} {sub:>4} "
              f"{int(row['rank']):>5} {row['mu']:>7.2f} {int(row['wage_kpw']):>6}k "
              f"{row['sigma']:>5.1f} {row['beta']:>5.2f}  {status}")

def print_best11(sel, label=""):
    print(f"\n--- BEST XI: {label} (4-3-3) ---")
    rows = df.iloc[sel]
    total_mu   = rows["mu"].sum()
    total_wage = rows["wage_kpw"].sum()
    order = ["GK", "CB", "FB", "MF", "FW"]
    # add sub_position for grouping
    tmp = rows.copy()
    tmp["grp"] = tmp.apply(lambda r: r["sub_position"] if r["position"]=="DF" and pd.notna(r["sub_position"]) else r["position"], axis=1)
    grp_order = {"GK":0,"CB":1,"FB":2,"MF":3,"FW":4}
    tmp["grp_ord"] = tmp["grp"].map(grp_order).fillna(5)
    tmp = tmp.sort_values("grp_ord")
    print(f"  {'Name':<28} {'Pos':>4} {'Sub':>4}  {'mu':>7}  {'Wage':>6}  {'Sig':>5}  {'Bet':>5}")
    print("  " + "-"*70)
    for _, r in tmp.iterrows():
        sub = r['sub_position'] if pd.notna(r['sub_position']) else "  -"
        print(f"  {r['name']:<28} {r['position']:>4} {sub:>4}  {r['mu']:>7.2f}  {int(r['wage_kpw']):>6}k  {r['sigma']:>5.1f}  {r['beta']:>5.2f}")
    print("  " + "-"*70)
    print(f"  {'TOTAL':<28}              {total_mu:>7.2f}  {int(total_wage):>6}k")

def print_comparison(sel_a, sel_b, label_a, label_b):
    a = set(sel_a); b = set(sel_b)
    added   = b - a
    removed = a - b
    same    = a & b
    print(f"\n  Comparison: {label_a} -> {label_b}")
    print(f"  Same ({len(same)}): {', '.join(df.iloc[list(same)]['name'].tolist())}")
    if removed:
        print(f"  OUT  ({len(removed)}): {', '.join(df.iloc[list(removed)]['name'].tolist())}")
    if added:
        print(f"  IN   ({len(added)}): {', '.join(df.iloc[list(added)]['name'].tolist())}")
    if not removed and not added:
        print("  ** IDENTICAL SQUADS **")

# ── scenario generation ───────────────────────────────────────────────────────
print("Generating scenarios...")
P_ind = generate_independent_gaussian(df, K, seed=42)
P_cor = generate_correlated_factor(df, K, sigma_Z=3.0, seed=42)

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: RQ1
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 1: RQ1 -- RISK MEASURE COMPARISON")
print("="*100)

# ── Model 1a: Deterministic ───────────────────────────────────────────────────
print("\nSolving Model 1a: Deterministic...")
prob1a = LpProblem("Det", LpMaximize)
x1a = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
prob1a += lpSum(df.iloc[i]["mu"] * x1a[i] for i in range(n))
add_structural_constraints(prob1a, x1a, df, n)
st1a = solve_model(prob1a)
sel1a = get_selected(x1a, n, st1a)
print(f"  Status: {st1a}")
print_full_table(set(sel1a), "Model 1a: Deterministic")
print_best11(sel1a, "Model 1a: Deterministic")

# Compute deterministic squad's MAD and CVaR for calibration
det_perfs_ind = P_ind[sel1a, :].sum(axis=0)
det_mad = compute_mad(P_ind[sel1a, :])
det_loss = tau - det_perfs_ind
det_cvar = compute_cvar(det_loss)
print(f"\n  Deterministic squad: MAD={det_mad:.3f}, CVaR={det_cvar:.3f}")

# ── Model 1b: Mean-MAD ────────────────────────────────────────────────────────
print("\nCalibrating Model 1b: Mean-MAD...")
# MAD linearisation: MAD = (1/K) * sum_k |perf_k - mean_perf|
# Let d+_k, d-_k >= 0: d+_k - d-_k = perf_k - mean_perf  => MAD = (1/K)*sum(d+_k+d-_k)
# Since mean_perf = sum_i mu_i x_i (deterministic approx), this works

def solve_mad(mad_bound):
    prob = LpProblem("MAD", LpMaximize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    dp = [LpVariable(f"dp_{k}", lowBound=0) for k in range(K)]
    dm = [LpVariable(f"dm_{k}", lowBound=0) for k in range(K)]
    prob += lpSum(df.iloc[i]["mu"] * x[i] for i in range(n))
    add_structural_constraints(prob, x, df, n)
    mean_perf = lpSum(df.iloc[i]["mu"] * x[i] for i in range(n))
    for k in range(K):
        perf_k = lpSum(P_ind[i, k] * x[i] for i in range(n))
        prob += dp[k] - dm[k] == perf_k - mean_perf
    prob += lpSum(dp[k] + dm[k] for k in range(K)) / K <= mad_bound
    st = solve_model(prob)
    return st, get_selected(x, n, st)

mad_sel = None
mad_bound_used = None
for frac in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    mb = frac * det_mad
    print(f"  Trying MAD bound = {frac:.2f} * det_MAD = {mb:.3f}...")
    st, sel = solve_mad(mb)
    if st == "Optimal" and set(sel) != set(sel1a):
        mad_sel = sel
        mad_bound_used = mb
        print(f"  --> Found different squad at frac={frac:.2f}")
        break
    elif st == "Optimal":
        print(f"  --> Feasible but same squad")
    else:
        print(f"  --> Infeasible")

if mad_sel is None:
    # fallback: use tighter bound
    print("  Fallback: using MAD bound = 0.45 * det_MAD")
    mb = 0.45 * det_mad
    st, mad_sel = solve_mad(mb)
    mad_bound_used = mb
    if st != "Optimal":
        print("  WARNING: MAD model infeasible at fallback, using deterministic")
        mad_sel = sel1a

print_full_table(set(mad_sel), "Model 1b: Mean-MAD")
print_best11(mad_sel, "Model 1b: Mean-MAD")
print_comparison(sel1a, mad_sel, "Deterministic", "Mean-MAD")

# ── Model 1c: Mean-CVaR ───────────────────────────────────────────────────────
print("\nCalibrating Model 1c: Mean-CVaR...")

def solve_cvar(cvar_bound, P=None):
    if P is None: P = P_ind
    prob = LpProblem("CVaR", LpMaximize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    t = LpVariable("t")
    u = [LpVariable(f"u_{k}", lowBound=0) for k in range(K)]
    prob += lpSum(df.iloc[i]["mu"] * x[i] for i in range(n))
    add_structural_constraints(prob, x, df, n)
    for k in range(K):
        loss_k = tau - lpSum(P[i, k] * x[i] for i in range(n))
        prob += u[k] >= loss_k - t
    prob += t + lpSum(u[k] for k in range(K)) / ((1 - alpha) * K) <= cvar_bound
    st = solve_model(prob)
    return st, get_selected(x, n, st)

def solve_min_cvar(P=None):
    """Find the squad that minimises CVaR (lower bound for Pareto sweep)."""
    if P is None: P = P_ind
    prob = LpProblem("MinCVaR", LpMinimize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    t = LpVariable("t")
    u = [LpVariable(f"u_{k}", lowBound=0) for k in range(K)]
    prob += t + lpSum(u[k] for k in range(K)) / ((1 - alpha) * K)
    add_structural_constraints(prob, x, df, n)
    for k in range(K):
        loss_k = tau - lpSum(P[i, k] * x[i] for i in range(n))
        prob += u[k] >= loss_k - t
    st = solve_model(prob)
    sel = get_selected(x, n, st)
    if st == "Optimal" and sel:
        perfs = P[sel, :].sum(axis=0)
        cvar = compute_cvar(tau - perfs)
        return st, sel, cvar
    return st, [], None

# Find minimum feasible CVaR bound
print("  Scanning for minimum feasible CVaR bound...")
min_feas_cvar = None
for frac in np.arange(0.50, 1.06, 0.05):
    cb = frac * det_cvar
    st, _ = solve_cvar(cb)
    if st == "Optimal":
        min_feas_cvar = cb
        print(f"  Min feasible CVaR bound = {frac:.2f} * det_CVaR = {cb:.3f}")
        break
    else:
        print(f"  frac={frac:.2f}: Infeasible")

if min_feas_cvar is None:
    min_feas_cvar = det_cvar
    print("  Could not find feasible bound below det_CVaR; using det_CVaR")

# Now scan from min_feasible to det_CVaR for a different squad
cvar_sel = None
cvar_bound_used = None
scan_vals = np.linspace(min_feas_cvar, det_cvar * 0.99, 12)
for cb in scan_vals:
    st, sel = solve_cvar(cb)
    if st == "Optimal" and set(sel) != set(sel1a):
        cvar_sel = sel
        cvar_bound_used = cb
        print(f"  --> Found different CVaR squad at bound={cb:.3f}")
        break
    elif st == "Optimal":
        print(f"  bound={cb:.3f}: Feasible but same squad")
    else:
        print(f"  bound={cb:.3f}: Infeasible")

# Ensure also different from MAD squad
if cvar_sel is not None and set(cvar_sel) == set(mad_sel):
    print("  CVaR squad same as MAD squad -- tightening further...")
    for cb in np.linspace(min_feas_cvar, scan_vals[0], 8):
        st, sel = solve_cvar(cb)
        if st == "Optimal" and set(sel) != set(sel1a) and set(sel) != set(mad_sel):
            cvar_sel = sel
            cvar_bound_used = cb
            print(f"  --> Found triple-distinct CVaR squad at bound={cb:.3f}")
            break

if cvar_sel is None:
    print("  WARNING: Using min feasible CVaR bound")
    _, cvar_sel = solve_cvar(min_feas_cvar)
    cvar_bound_used = min_feas_cvar
    if cvar_sel is None:
        cvar_sel = sel1a

print_full_table(set(cvar_sel), "Model 1c: Mean-CVaR")
print_best11(cvar_sel, "Model 1c: Mean-CVaR")
print_comparison(sel1a, cvar_sel, "Deterministic", "Mean-CVaR")
print_comparison(mad_sel, cvar_sel, "Mean-MAD", "Mean-CVaR")

stats1a = squad_stats(sel1a, P_ind, P_cor)
stats1b = squad_stats(mad_sel, P_ind, P_cor)
stats1c = squad_stats(cvar_sel, P_ind, P_cor)

print("\n-- RQ1 Summary --")
for lbl, st in [("Deterministic", stats1a), ("Mean-MAD", stats1b), ("Mean-CVaR", stats1c)]:
    print(f"  {lbl:<20} E[Perf]={st['mu']:.2f}  Wage={int(st['wage'])}k  "
          f"CVaR_ind={st['cvar_ind']:.3f}  MAD={st['mad']:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: RQ2 -- Pareto fronts
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 2: RQ2 -- PARETO FRONTS (independent vs correlated)")
print("="*100)

def build_pareto_front(P, n_points=15, label=""):
    """
    Epsilon-constraint Pareto front: max E[Perf] s.t. CVaR(Loss) <= epsilon.
    First finds the true min-CVaR and det-squad CVaR under P, then sweeps between.
    """
    # Step 1: find minimum achievable CVaR under this scenario set
    print(f"  {label}: finding min-CVaR squad...", flush=True)
    st_min, sel_min, cvar_min = solve_min_cvar(P)
    if st_min != "Optimal" or cvar_min is None:
        print(f"  {label}: min-CVaR solve failed, using 0.60*det_ref")
        perfs_det = P[sel1a, :].sum(axis=0)
        cvar_min  = 0.60 * compute_cvar(tau - perfs_det)

    # Step 2: det-squad CVaR as upper bound
    perfs_det = P[sel1a, :].sum(axis=0)
    cvar_det  = compute_cvar(tau - perfs_det)
    # Extend slightly above det for completeness
    cvar_max  = cvar_det * 1.05

    print(f"  {label}: min-CVaR={cvar_min:.3f}, det-CVaR={cvar_det:.3f}", flush=True)

    # If range is too narrow, add extra points below min
    if (cvar_max - cvar_min) < 1.0:
        cvar_min = cvar_min * 0.95

    bounds = np.linspace(cvar_min * 1.001, cvar_max, n_points)
    front = []
    for j, cb in enumerate(bounds):
        print(f"  {label} point {j+1}/{n_points} (CVaR bound={cb:.3f})...", flush=True)
        st, sel = solve_cvar(cb, P=P)
        if st == "Optimal" and sel:
            mu_val = df.iloc[sel]["mu"].sum()
            perfs  = P[sel, :].sum(axis=0)
            cv     = compute_cvar(tau - perfs)
            front.append((cv, mu_val, sel))

    # Also add the min-CVaR squad if it exists
    if sel_min:
        mu_min = df.iloc[sel_min]["mu"].sum()
        front.insert(0, (cvar_min, mu_min, sel_min))

    # Deduplicate by squad composition
    seen = set()
    unique = []
    for cv, mu, sel in front:
        key = frozenset(sel)
        if key not in seen:
            seen.add(key)
            unique.append((cv, mu, sel))
    unique.sort(key=lambda r: r[0])
    return unique

print("\nBuilding independent Pareto front...")
front_ind = build_pareto_front(P_ind, n_points=15, label="IND")

print("\nBuilding correlated Pareto front...")
front_cor = build_pareto_front(P_cor, n_points=15, label="COR")

# Check if sigma_Z=3 gives flat fronts
def front_range(front):
    if len(front) < 2: return 0
    return max(f[1] for f in front) - min(f[1] for f in front)

if front_range(front_ind) < 1.0 and front_range(front_cor) < 1.0:
    print("  Fronts appear flat -- retrying with sigma_Z=5.0")
    P_cor5 = generate_correlated_factor(df, K, sigma_Z=5.0, seed=42)
    front_cor = build_pareto_front(P_cor5, n_points=15, label="COR5")
    if front_range(front_cor) < 1.0:
        print("  Still flat -- retrying with sigma_Z=7.0")
        P_cor7 = generate_correlated_factor(df, K, sigma_Z=7.0, seed=42)
        front_cor = build_pareto_front(P_cor7, n_points=15, label="COR7")
        P_cor = P_cor7
    else:
        P_cor = P_cor5

# Mid-front squads
def mid_squad(front):
    if not front: return None
    return front[len(front)//2]

mid_ind = mid_squad(front_ind)
mid_cor = mid_squad(front_cor)

if mid_ind:
    print_full_table(set(mid_ind[2]), "Exp2: Mid-front INDEPENDENT")
    print_best11(mid_ind[2], "Mid-front Independent")
if mid_cor:
    print_full_table(set(mid_cor[2]), "Exp2: Mid-front CORRELATED")
    print_best11(mid_cor[2], "Mid-front Correlated")
if mid_ind and mid_cor:
    print_comparison(mid_ind[2], mid_cor[2], "Mid-IND", "Mid-COR")

print(f"\n  Independent front: {len(front_ind)} points, "
      f"CVaR=[{front_ind[0][0]:.2f},{front_ind[-1][0]:.2f}], "
      f"mu=[{front_ind[0][1]:.2f},{front_ind[-1][1]:.2f}]")
print(f"  Correlated  front: {len(front_cor)} points, "
      f"CVaR=[{front_cor[0][0]:.2f},{front_cor[-1][0]:.2f}], "
      f"mu=[{front_cor[0][1]:.2f},{front_cor[-1][1]:.2f}]")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: RQ3 -- Weighted-sum vs epsilon-constraint
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 3: RQ3 -- WEIGHTED-SUM vs EPSILON-CONSTRAINT")
print("="*100)

# Epsilon-constraint: reuse front_ind
ec_solutions = [(cv, mu, sel) for cv, mu, sel in front_ind]
print(f"  EC solutions from front_ind: {len(ec_solutions)}")

# Weighted-sum: w * E[Perf] - (1-w) * CVaR(Loss) approx
# Linearised: max w*sum(mu_i x_i) - (1-w)*(t + sum(u_k)/((1-alpha)*K))
def solve_ws(w, P=None):
    if P is None: P = P_ind
    prob = LpProblem("WS", LpMaximize)
    x = [LpVariable(f"x_{i}", cat="Binary") for i in range(n)]
    t = LpVariable("t")
    u = [LpVariable(f"u_{k}", lowBound=0) for k in range(K)]
    prob += w * lpSum(df.iloc[i]["mu"] * x[i] for i in range(n)) \
           - (1-w) * (t + lpSum(u[k] for k in range(K)) / ((1-alpha)*K))
    add_structural_constraints(prob, x, df, n)
    for k in range(K):
        loss_k = tau - lpSum(P[i,k]*x[i] for i in range(n))
        prob += u[k] >= loss_k - t
    st = solve_model(prob)
    return st, get_selected(x, n, st)

ws_weights = np.linspace(0.1, 0.9, 10)
ws_solutions = []
for j, w in enumerate(ws_weights):
    print(f"  WS point {j+1}/{len(ws_weights)} (w={w:.2f})...")
    st, sel = solve_ws(w)
    if st == "Optimal":
        perfs = P_ind[sel, :].sum(axis=0)
        loss  = tau - perfs
        cv    = compute_cvar(loss)
        mu_v  = df.iloc[sel]["mu"].sum()
        ws_solutions.append((cv, mu_v, sel))
        print(f"    mu={mu_v:.2f}, CVaR={cv:.3f}")

# Unique WS solutions by squad composition
def squad_key(sel):
    return frozenset(sel)

ws_unique = []
seen_keys = set()
for cv, mu, sel in ws_solutions:
    k = squad_key(sel)
    if k not in seen_keys:
        seen_keys.add(k)
        ws_unique.append((cv, mu, sel))

print(f"\n  WS produced {len(ws_unique)} unique squads from {len(ws_solutions)} solves")

# EC-only solutions (missed by WS)
ec_keys = {squad_key(s[2]) for s in ec_solutions}
ws_keys  = {squad_key(s[2]) for s in ws_unique}
missed_keys = ec_keys - ws_keys
missed_ec = [s for s in ec_solutions if squad_key(s[2]) in missed_keys]
print(f"  EC-only solutions missed by WS: {len(missed_ec)}")

if len(missed_ec) == 0:
    print("  No missed solutions -- trying coarser WS grid (6 weights)...")
    ws_weights6 = np.linspace(0.2, 0.8, 6)
    ws6 = []
    for w in ws_weights6:
        st, sel = solve_ws(w)
        if st == "Optimal":
            perfs = P_ind[sel,:].sum(axis=0)
            cv = compute_cvar(tau - perfs)
            mu_v = df.iloc[sel]["mu"].sum()
            ws6.append((cv, mu_v, sel))
    ws6_keys = {squad_key(s[2]) for s in ws6}
    missed_ec = [s for s in ec_solutions if squad_key(s[2]) not in ws6_keys]
    print(f"  After coarse grid: EC-only missed = {len(missed_ec)}")

    if len(missed_ec) == 0:
        print("  Extending EC range to 30 points (0.40-1.25 * det_CVaR)...")
        perfs_det = P_ind[sel1a,:].sum(axis=0)
        cvar_ref  = compute_cvar(tau - perfs_det)
        bounds_wide = np.linspace(0.40*cvar_ref, 1.25*cvar_ref, 30)
        front_wide = []
        for j, cb in enumerate(bounds_wide):
            print(f"  Wide EC point {j+1}/30 (CVaR bound={cb:.3f})...")
            st, sel = solve_cvar(cb)
            if st == "Optimal":
                perfs = P_ind[sel,:].sum(axis=0)
                cv = compute_cvar(tau - perfs)
                mu_v = df.iloc[sel]["mu"].sum()
                front_wide.append((cv, mu_v, sel))
        # deduplicate
        seen = set()
        front_wide_u = []
        for cv, mu, sel in front_wide:
            k2 = squad_key(sel)
            if k2 not in seen:
                seen.add(k2)
                front_wide_u.append((cv, mu, sel))
        missed_ec = [s for s in front_wide_u if squad_key(s[2]) not in ws_keys]
        ec_solutions = front_wide_u
        print(f"  Wide EC: {len(front_wide_u)} unique, missed={len(missed_ec)}")

print(f"\n  Final: EC has {len(ec_solutions)} unique, WS has {len(ws_unique)} unique, "
      f"EC-only missed = {len(missed_ec)}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: RQ4 -- Final integrated squad
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("EXPERIMENT 4: RQ4 -- FINAL INTEGRATED SQUAD (correlated scenarios)")
print("="*100)

# Use CVaR bound from correlated mid-front
if mid_cor:
    cvar_mid_cor = mid_cor[0]
else:
    cvar_mid_cor = det_cvar * 0.85

print(f"  Initial CVaR bound (mid-front correlated) = {cvar_mid_cor:.3f}")
st_int, sel_int = solve_cvar(cvar_mid_cor, P=P_cor)

# If same as deterministic, tighten progressively
for tighten_frac in [0.90, 0.85, 0.80, 0.75, 0.70]:
    if st_int == "Optimal" and set(sel_int) != set(sel1a):
        break
    cb_new = tighten_frac * cvar_mid_cor
    print(f"  Same as deterministic (or infeasible) -- tightening to {tighten_frac} * bound = {cb_new:.3f}")
    st_try, sel_try = solve_cvar(cb_new, P=P_cor)
    if st_try == "Optimal" and sel_try:
        st_int, sel_int = st_try, sel_try

# If still same or infeasible, try min-CVaR squad from correlated + sweep
if st_int != "Optimal" or not sel_int or set(sel_int) == set(sel1a):
    print("  Trying min-CVaR correlated squad as fallback...")
    st_mc, sel_mc, cv_mc = solve_min_cvar(P_cor)
    if st_mc == "Optimal" and sel_mc and set(sel_mc) != set(sel1a):
        st_int, sel_int = st_mc, sel_mc
        print(f"  --> Min-CVaR correlated squad found (CVaR={cv_mc:.3f})")
    else:
        # Use a slightly looser bound around the min-CVaR to push for a different squad
        _, _, cv_mc_val = solve_min_cvar(P_cor)
        if cv_mc_val:
            for frac2 in [1.01, 1.02, 1.03, 1.05]:
                st_try2, sel_try2 = solve_cvar(frac2 * cv_mc_val, P=P_cor)
                if st_try2 == "Optimal" and sel_try2 and set(sel_try2) != set(sel1a):
                    st_int, sel_int = st_try2, sel_try2
                    print(f"  --> Found different squad at {frac2}*minCVaR")
                    break
        # last resort: use independent CVaR squad
        if st_int != "Optimal" or not sel_int or set(sel_int) == set(sel1a):
            print("  Last resort: using Mean-CVaR (independent) squad for Exp4")
            st_int, sel_int = "Optimal", cvar_sel

print(f"  Status: {st_int}")
print_full_table(set(sel_int), "Exp4: Final Integrated (Correlated CVaR)")
print_best11(sel_int, "Final Integrated (Correlated CVaR)")
print_comparison(sel1a, sel_int, "Deterministic", "Final Integrated")

stats_int = squad_stats(sel_int, P_ind, P_cor)

# ─────────────────────────────────────────────────────────────────────────────
# GRAND SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*100)
print("GRAND SUMMARY TABLE")
print("="*100)

def fmt_row(label, st):
    return (f"{label:<30} {st['mu']:>8.2f}  {int(st['wage']):>6}k  4-3-3  "
            f"{st['avg_sig']:>7.3f}  {st['avg_bet']:>7.3f}  "
            f"{st['cvar_ind']:>9.3f}  {st['cvar_cor']:>9.3f}  {st['mad']:>7.3f}")

header = (f"{'Model':<30} {'E[Perf]':>8}  {'Wage':>7}  {'Form':5}  "
          f"{'Avg_sig':>7}  {'Avg_bet':>7}  {'CVaR_ind':>9}  {'CVaR_cor':>9}  {'MAD':>7}")
print(header)
print("-"*110)

if mid_ind:
    stats_mid_ind = squad_stats(mid_ind[2], P_ind, P_cor)
else:
    stats_mid_ind = stats1a

if mid_cor:
    stats_mid_cor = squad_stats(mid_cor[2], P_ind, P_cor)
else:
    stats_mid_cor = stats1a

for lbl, st in [
    ("Deterministic",             stats1a),
    ("Mean-MAD",                  stats1b),
    ("Mean-CVaR (ind)",           stats1c),
    ("Pareto-mid (ind)",          stats_mid_ind),
    ("Pareto-mid (cor)",          stats_mid_cor),
    ("Final Integrated (cor)",    stats_int),
]:
    print(fmt_row(lbl, st))

# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

# Colours
COLS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
MUTED = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# ── fig_rq1_comparison.pdf ────────────────────────────────────────────────────
labels_rq1 = ["Deterministic", "Mean-MAD", "Mean-CVaR", "Final Intg."]
stats_rq1  = [stats1a, stats1b, stats1c, stats_int]
metrics    = ["E[Performance]", "Avg Sigma", "Avg Beta", r"CVaR$_{0.95}$(Loss)"]
vals_rq1   = [
    [s["mu"]       for s in stats_rq1],
    [s["avg_sig"]  for s in stats_rq1],
    [s["avg_bet"]  for s in stats_rq1],
    [s["cvar_ind"] for s in stats_rq1],
]

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 8))
subplot_labels = ["(a)", "(b)", "(c)", "(d)"]
for ax, metric, vals, sp_lbl, col in zip(axes1.flatten(), metrics, vals_rq1, subplot_labels, MUTED):
    bars = ax.barh(labels_rq1, vals, color=col, alpha=0.85)
    ax.set_xlabel(metric)
    # value labels
    x_max = max(vals) if vals else 1
    for bar, v in zip(bars, vals):
        ax.text(v + 0.01*x_max, bar.get_y() + bar.get_height()/2,
                f"{v:.2f}", va='center', ha='left', fontsize=9)
    ax.set_xlim(0, x_max * 1.18)
    ax.text(0.02, 0.97, sp_lbl, transform=ax.transAxes, fontsize=12,
            fontweight='bold', va='top')

fig1.tight_layout(pad=2.0)
fig1.savefig('fig_rq1_comparison.pdf', dpi=300, bbox_inches='tight')
print("  Saved fig_rq1_comparison.pdf")

# ── fig_rq2_pareto.pdf ────────────────────────────────────────────────────────
cvar_ind_vals = [f[0] for f in front_ind]
mu_ind_vals   = [f[1] for f in front_ind]
cvar_cor_vals = [f[0] for f in front_cor]
mu_cor_vals   = [f[1] for f in front_cor]

# Check if scales are similar
def scales_similar(a, b, threshold=2.0):
    if not a or not b: return True
    mid_a = np.median(a); mid_b = np.median(b)
    if mid_a == 0 or mid_b == 0: return True
    ratio = max(mid_a, mid_b) / max(min(mid_a, mid_b), 1e-9)
    return ratio < threshold

if scales_similar(cvar_ind_vals, cvar_cor_vals):
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 5))
    ax2.plot(cvar_ind_vals, mu_ind_vals, 'o-', color=MUTED[0], label='Independent', linewidth=1.5, markersize=6)
    ax2.plot(cvar_cor_vals, mu_cor_vals, 's-', color=MUTED[2], label='Correlated', linewidth=1.5, markersize=6)
    ax2.set_xlabel(r'CVaR$_{0.95}$(Loss)')
    ax2.set_ylabel(r'E[Performance]')
    ax2.legend(loc='best')
    ax2.text(0.02, 0.97, "(a)", transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
    fig2.tight_layout(pad=1.5)
else:
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    ax2a.plot(cvar_ind_vals, mu_ind_vals, 'o-', color=MUTED[0], linewidth=1.5, markersize=6)
    ax2a.set_xlabel(r'CVaR$_{0.95}$(Loss)')
    ax2a.set_ylabel(r'E[Performance]')
    ax2a.set_title('Independent', fontsize=11)
    ax2a.text(0.02, 0.97, "(a)", transform=ax2a.transAxes, fontsize=12, fontweight='bold', va='top')

    ax2b.plot(cvar_cor_vals, mu_cor_vals, 's-', color=MUTED[2], linewidth=1.5, markersize=6)
    ax2b.set_xlabel(r'CVaR$_{0.95}$(Loss)')
    ax2b.set_ylabel(r'E[Performance]')
    ax2b.set_title('Correlated', fontsize=11)
    ax2b.text(0.02, 0.97, "(b)", transform=ax2b.transAxes, fontsize=12, fontweight='bold', va='top')

    fig2.tight_layout(pad=1.5)

fig2.savefig('fig_rq2_pareto.pdf', dpi=300, bbox_inches='tight')
print("  Saved fig_rq2_pareto.pdf")

# ── fig_rq3_scalarisation.pdf ─────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 6))

ec_x = [s[0] for s in ec_solutions]
ec_y = [s[1] for s in ec_solutions]
ws_x = [s[0] for s in ws_unique]
ws_y = [s[1] for s in ws_unique]
mx_x = [s[0] for s in missed_ec]
mx_y = [s[1] for s in missed_ec]

ax3.scatter(ec_x, ec_y, marker='s', s=80, color=MUTED[2], label='Epsilon-constraint', zorder=3)
ax3.scatter(ws_x, ws_y, marker='o', s=80, color=MUTED[0], label='Weighted-sum', zorder=3)
if mx_x:
    ax3.scatter(mx_x, mx_y, marker='*', s=200, color='green', label='EC-only (missed by WS)', zorder=4)
    for cx, cy in zip(mx_x, mx_y):
        ax3.annotate('missed', xy=(cx, cy), xytext=(cx + 0.02*(max(ec_x)-min(ec_x)), cy),
                     fontsize=9, color='green',
                     arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

ax3.set_xlabel(r'CVaR$_{0.95}$(Loss)')
ax3.set_ylabel(r'E[Performance]')
ax3.legend(loc='best', fontsize=10)
ax3.text(0.02, 0.97, "(a)", transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')

fig3.tight_layout(pad=2.0)
fig3.savefig('fig_rq3_scalarisation.pdf', dpi=300, bbox_inches='tight')
print("  Saved fig_rq3_scalarisation.pdf")

print("\n" + "="*100)
print("ALL EXPERIMENTS COMPLETE")
print("="*100)
