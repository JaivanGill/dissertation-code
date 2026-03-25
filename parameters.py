"""
Parameter file for Multi-criteria Optimisation under Uncertainty in Football Analytics.
100 Premier League players (2024-25 season) with performance, wage, sigma, and beta data.

Data sources:
  - Performance (npxG, xA, minutes, goals prevented): FotMob 2024-25 season
  - Wages: FBref / Capology 2024-25 estimates
  - Sigma (volatility): Derived from season-to-season output variation (FotMob 2021-24),
    injury history (Transfermarkt 2021-24), and age profile
  - Beta (systemic exposure): Estimated from tactical role analysis, multi-manager
    performance evidence, and system-dependency indicators

Model:
  Outfield mu_i = (npxG_i / min_i) * 90 + (xA_i / min_i) * 90, projected to 38 games
  GK mu_i = (GoalsPrevented_i / min_i) * 90, projected to 38 games
  Stochastic: p_i(xi) = mu_i + beta_i * Z(xi) + eta_i(xi)
"""

import pandas as pd
import numpy as np

# =====================================================================
# RAW PLAYER DATA: 2024-25 PREMIER LEAGUE SEASON
# =====================================================================
# Columns: rank, name, position, minutes, npxG, xA, goals_prevented, wage_kpw, sigma, beta

raw_data = [
    # FORWARDS (27)
    # rank, name, pos, min, npxG, xA, GP, wage(k), sigma, beta, sub_pos
    (1, "Mohamed Salah", "FW", 3377, 18.27, 9.06, None, 350, 2.5, 0.2, None),
    (4, "Bryan Mbeumo", "FW", 3415, 7.53, 9.26, None, 45, 3.8, 0.3, None),
    (5, "Erling Haaland", "FW", 2741, 18.86, 2.02, None, 375, 4.0, 0.4, None),
    (6, "Matheus Cunha", "FW", 2600, 8.63, 5.28, None, 60, 3.5, 0.2, None),
    (7, "Bukayo Saka", "FW", 1735, 6.06, 7.80, None, 195, 3.8, 0.3, None),
    (11, "Alexander Isak", "FW", 2769, 17.27, 3.60, None, 120, 5.0, 0.2, None),
    (17, "Luis Diaz", "FW", 2410, 12.02, 4.46, None, 55, 5.3, 0.4, None),
    (19, "Savinho", "FW", 1770, 5.01, 6.10, None, 40, 3.4, 0.4, None),
    (23, "Jeremy Doku", "FW", 1513, 1.32, 3.80, None, 50, 4.6, 0.3, None),
    (28, "Yoane Wissa", "FW", 2927, 18.59, 1.83, None, 25, 1.8, 0.5, None),
    (29, "Antoine Semenyo", "FW", 3209, 9.98, 4.35, None, 50, 2.2, 0.5, None),
    (43, "Heung-Min Son", "FW", 2116, 5.78, 5.13, None, 190, 3.3, 0.3, None),
    (50, "Jarrod Bowen", "FW", 2979, 7.85, 6.47, None, 150, 1.8, 0.3, None),
    (51, "Callum Hudson-Odoi", "FW", 2201, 2.47, 2.61, None, 80, 2.8, 0.5, None),
    (53, "Noni Madueke", "FW", 2046, 9.64, 3.15, None, 50, 2.3, 0.3, None),
    (56, "Joao Pedro", "FW", 1953, 4.96, 3.09, None, 50, 4.6, 0.4, None),
    (62, "Justin Kluivert", "FW", 2356, 5.38, 3.73, None, 80, 2.6, 0.5, None),
    (67, "Iliman Ndiaye", "FW", 2440, 4.67, 1.35, None, 45, 3.0, 0.3, None),
    (68, "Kaoru Mitoma", "FW", 2608, 9.16, 4.04, None, 80, 3.2, 0.3, None),
    (69, "Dango Ouattara", "FW", 2006, 8.36, 3.29, None, 35, 2.0, 0.6, None),
    (72, "Cody Gakpo", "FW", 1939, 7.09, 3.77, None, 120, 3.2, 0.4, None),
    (80, "Kai Havertz", "FW", 1874, 9.56, 1.67, None, 280, 1.5, 0.7, None),
    (83, "Ismaila Sarr", "FW", 2714, 10.72, 6.74, None, 70, 2.4, 0.6, None),
    (85, "Anthony Gordon", "FW", 2444, 6.45, 5.11, None, 150, 2.9, 0.4, None),
    (87, "Chris Wood", "FW", 2976, 10.99, 1.52, None, 80, 7.2, 0.5, None),
    (89, "Nicolas Jackson", "FW", 2238, 12.34, 1.92, None, 100, 4.9, 0.5, None),
    (99, "Leandro Trossard", "FW", 2550, 7.21, 4.11, None, 90, 3.8, 0.5, None),

    # MIDFIELDERS (35)
    # rank, name, pos, min, npxG, xA, GP, wage(k), sigma, beta, sub_pos
    (2, "Cole Palmer", "MF", 3195, 13.33, 9.14, None, 130, 5.7, 0.2, None),
    (3, "Bruno Fernandes", "MF", 3024, 7.59, 7.98, None, 300, 1.0, 0.3, None),
    (8, "Amad Diallo", "MF", 1901, 4.68, 4.18, None, 29, 2.4, 0.6, None),
    (9, "Moises Caicedo", "MF", 3356, 0.82, 2.82, None, 150, 1.4, 0.3, None),
    (10, "Eberechi Eze", "MF", 2600, 8.92, 4.68, None, 100, 6.3, 0.2, None),
    (12, "Youri Tielemans", "MF", 3032, 2.53, 5.65, None, 150, 1.0, 0.5, None),
    (14, "Alexis Mac Allister", "MF", 2607, 2.83, 5.64, None, 150, 1.5, 0.5, None),
    (16, "Thomas Partey", "MF", 2799, 2.29, 2.62, None, 200, 1.3, 0.6, None),
    (21, "Declan Rice", "MF", 2833, 3.51, 5.63, None, 240, 2.3, 0.4, None),
    (24, "Mateo Kovacic", "MF", 2202, 1.88, 3.23, None, 150, 1.3, 0.5, None),
    (25, "Enzo Fernandez", "MF", 2946, 6.13, 6.22, None, 180, 2.4, 0.6, None),
    (26, "Ryan Gravenberch", "MF", 3168, 1.12, 3.53, None, 150, 2.0, 0.7, None),
    (27, "James Maddison", "MF", 1816, 7.50, 5.50, None, 170, 2.0, 0.6, None),
    (30, "Bernardo Silva", "MF", 2667, 3.90, 4.54, None, 300, 2.5, 0.6, None),
    (31, "Alex Iwobi", "MF", 2994, 4.29, 6.98, None, 80, 3.0, 0.6, None),
    (33, "Bruno Guimaraes", "MF", 3282, 4.31, 5.39, None, 160, 2.5, 0.3, None),
    (34, "Ryan Christie", "MF", 2131, 2.12, 2.99, None, 70, 1.0, 0.8, None),
    (35, "Morgan Gibbs-White", "MF", 2822, 6.34, 4.69, None, 80, 1.1, 0.4, None),
    (36, "Ilkay Gundogan", "MF", 2227, 4.00, 3.50, None, 230, 4.9, 0.7, None),
    (37, "Christian Norgaard", "MF", 2829, 4.36, 1.37, None, 35, 1.0, 0.4, None),
    (38, "Mikkel Damsgaard", "MF", 2926, 2.83, 6.99, None, 30, 3.3, 0.6, None),
    (39, "Kevin De Bruyne", "MF", 1704, 5.07, 6.74, None, 400, 5.3, 0.4, None),
    (41, "Jacob Murphy", "MF", 2379, 5.72, 6.15, None, 35, 3.3, 0.6, None),
    (42, "Martin Odegaard", "MF", 2328, 4.02, 6.66, None, 240, 2.8, 0.4, None),
    (52, "Dominik Szoboszlai", "MF", 2496, 7.34, 5.88, None, 120, 2.7, 0.5, None),
    (54, "Elliot Anderson", "MF", 2742, 2.11, 3.63, None, 40, 2.2, 0.6, None),
    (55, "Dwight McNeil", "MF", 1371, 1.09, 4.32, None, 25, 2.1, 0.3, None),
    (57, "Sandro Tonali", "MF", 2631, 4.40, 2.76, None, 120, 2.2, 0.3, None),
    (64, "Carlos Baleba", "MF", 2670, 3.08, 1.21, None, 13, 1.0, 0.5, None),
    (74, "Casemiro", "MF", 1497, 1.83, 2.00, None, 350, 2.4, 0.8, None),
    (75, "Lewis Cook", "MF", 2978, 0.94, 3.75, None, 60, 1.0, 0.5, None),
    (91, "Marcus Tavernier", "MF", 1938, 5.48, 3.65, None, 35, 2.8, 0.6, None),
    (93, "Joelinton", "MF", 2404, 3.95, 2.33, None, 150, 2.8, 0.5, None),
    (96, "Idrissa Gana Gueye", "MF", 3067, 1.06, 1.97, None, 120, 1.0, 0.3, None),
    (98, "Morgan Rogers", "MF", 3129, 6.58, 5.24, None, 75, 1.8, 0.4, None),

    # DEFENDERS (30) — sub_position: CB = centre-back, FB = full-back/wing-back
    # rank, name, pos, min, npxG, xA, GP, wage(k), sigma, beta, sub_pos
    (13, "Trent Alexander-Arnold", "DF", 2377, 2.41, 7.37, None, 180, 2.4, 0.6, "FB"),
    (15, "Daniel Munoz", "DF", 1781, 1.98, 3.35, None, 45, 1.5, 0.6, "FB"),
    (18, "Josko Gvardiol", "DF", 3279, 4.59, 2.91, None, 200, 2.1, 0.5, "FB"),
    (20, "Virgil van Dijk", "DF", 3330, 2.24, 0.98, None, 220, 0.5, 0.2, "CB"),
    (22, "Antonee Robinson", "DF", 3167, 0.69, 4.14, None, 50, 1.1, 0.5, "FB"),
    (32, "Aaron Wan-Bissaka", "DF", 3154, 1.19, 2.78, None, 90, 1.1, 0.4, "FB"),
    (44, "Dean Huijsen", "DF", 2422, 1.87, 1.57, None, 30, 1.3, 0.4, "CB"),
    (46, "Gabriel Magalhaes", "DF", 2365, 2.61, 1.20, None, 100, 0.6, 0.4, "CB"),
    (47, "William Saliba", "DF", 3041, 2.28, 1.29, None, 190, 0.5, 0.2, "CB"),
    (48, "Matheus Nunes", "DF", 1673, 1.22, 3.32, None, 130, 2.5, 0.8, "FB"),
    (49, "Pedro Porro", "DF", 2608, 1.60, 4.64, None, 85, 2.0, 0.7, "FB"),
    (59, "Murillo", "DF", 3191, 1.42, 1.14, None, 30, 0.8, 0.4, "CB"),
    (61, "Rayan Ait-Nouri", "DF", 3127, 2.69, 4.00, None, 30, 1.5, 0.5, "FB"),
    (65, "Diogo Dalot", "DF", 2813, 2.05, 2.46, None, 85, 1.1, 0.7, "FB"),
    (66, "Jurrien Timber", "DF", 2422, 1.23, 1.27, None, 90, 3.3, 0.7, "FB"),
    (70, "Ibrahima Konate", "DF", 2565, 1.77, 0.93, None, 70, 1.2, 0.3, "CB"),
    (73, "Neco Williams", "DF", 2590, 2.30, 1.53, None, 50, 0.7, 0.6, "FB"),
    (76, "James Tarkowski", "DF", 2924, 1.31, 2.35, None, 100, 0.7, 0.3, "CB"),
    (78, "Milos Kerkez", "DF", 3342, 0.59, 2.58, None, 30, 1.2, 0.4, "FB"),
    (79, "Lisandro Martinez", "DF", 1754, 1.30, 1.28, None, 120, 1.0, 0.5, "CB"),
    (81, "Illia Zabarnyi", "DF", 3113, 1.27, 0.76, None, 50, 0.5, 0.3, "CB"),
    (82, "Ruben Dias", "DF", 2269, 1.50, 1.19, None, 180, 1.4, 0.3, "CB"),
    (84, "Ola Aina", "DF", 3003, 0.64, 2.22, None, 40, 0.7, 0.6, "FB"),
    (86, "Lewis Hall", "DF", 2192, 0.41, 3.72, None, 7, 3.3, 0.4, "FB"),
    (88, "Tyrick Mitchell", "DF", 3102, 1.31, 3.71, None, 40, 0.9, 0.7, "FB"),
    (90, "Kenny Tete", "DF", 1779, 1.36, 0.67, None, 50, 1.0, 0.6, "FB"),
    (92, "Pervis Estupinan", "DF", 2403, 1.17, 2.49, None, 50, 1.5, 0.6, "FB"),
    (95, "Trevoh Chalobah", "DF", 1974, 2.01, 0.64, None, 50, 1.0, 0.4, "CB"),
    (97, "Mats Wieffer", "DF", 1008, 0.75, 1.01, None, 60, 1.2, 0.6, "FB"),
    (100, "Nikola Milenkovic", "DF", 3330, 4.02, 0.54, None, 105, 1.1, 0.5, "CB"),

    # GOALKEEPERS (8)
    # For GKs: npxG and xA are None; goals_prevented is used instead
    # rank, name, pos, min, npxG, xA, GP, wage(k), sigma, beta, sub_pos
    (40, "Jordan Pickford", "GK", 3420, None, None, 6.15, 125, 3.6, 0.2, None),
    (45, "David Raya", "GK", 3420, None, None, 1.99, 100, 2.6, 0.3, None),
    (58, "Nick Pope", "GK", 2520, None, None, 0.05, 60, 2.4, 0.3, None),
    (60, "Alisson Becker", "GK", 2520, None, None, 2.14, 150, 4.5, 0.2, None),
    (63, "Robert Sanchez", "GK", 2880, None, None, 2.09, 60, 3.7, 0.4, None),
    (71, "Ederson", "GK", 2340, None, None, 4.46, 100, 3.1, 0.4, None),
    (77, "Matz Sels", "GK", 3420, None, None, 4.55, 35, 4.0, 0.5, None),
    (94, "Mark Flekken", "GK", 3330, None, None, -1.59, 30, 3.0, 0.3, None),
]

# =====================================================================
# BUILD DATAFRAME
# =====================================================================
columns = ["rank", "name", "position", "minutes", "npxG", "xA",
           "goals_prevented", "wage_kpw", "sigma", "beta", "sub_position"]
df = pd.DataFrame(raw_data, columns=columns)

# Compute mu (38-game projected composite performance)
def compute_mu(row):
    if row["position"] == "GK":
        return (row["goals_prevented"] / row["minutes"]) * 90 * 38
    else:
        per90 = (row["npxG"] / row["minutes"]) * 90 + (row["xA"] / row["minutes"]) * 90
        return per90 * 38

df["mu"] = df.apply(compute_mu, axis=1)

# =====================================================================
# MODEL PARAMETERS
# =====================================================================
S = 11          # Squad size
B = 1500        # Weekly wage budget (GBP thousands)
                # Big-6 average weekly payroll ~ £3,100k; B = 1500 forces genuine trade-offs

# Position bounds: (lower, upper) for each position group
position_bounds = {
    "GK": (1, 1),   # Exactly 1 goalkeeper
    "DF": (3, 5),   # 3-5 defenders
    "MF": (2, 4),   # 2-4 midfielders
    "FW": (2, 4),   # 2-4 forwards
}

# Sub-position constraint: minimum 2 centre-backs among the 4 defenders
CB_MIN = 2  # At least 2 CBs in the back four (realistic: 2 CBs + 2 FBs)

# Scenario parameters
K = 500         # Number of Monte Carlo scenarios
alpha = 0.95    # CVaR confidence level
tau = 200       # Loss target: tau - Perf(x,xi) = loss. Set to ~60% of best squad's
                # expected performance so that losses are positive and CVaR bounds
                # are intuitive.

# =====================================================================
# SCENARIO GENERATORS
# =====================================================================
def generate_independent_gaussian(df, K, seed=42):
    """
    Independent Gaussian scenarios: p_i(xi_k) = mu_i + epsilon_i(xi_k)
    where epsilon_i ~ N(0, sigma_i^2), independent across players.
    Returns (n x K) matrix of scenario performances.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    P = np.zeros((n, K))
    for i in range(n):
        P[i, :] = df.iloc[i]["mu"] + rng.normal(0, df.iloc[i]["sigma"], K)
    return P


def generate_correlated_factor(df, K, sigma_Z=1.0, seed=42):
    """
    One-factor correlated scenarios: p_i(xi_k) = mu_i + beta_i * Z_k + eta_i(xi_k)
    where Z ~ N(0, sigma_Z^2) is the common shock,
    eta_i ~ N(0, sigma_eta_i^2) is idiosyncratic noise,
    and sigma_eta_i^2 = sigma_i^2 - beta_i^2 * sigma_Z^2 (so total variance = sigma_i^2).
    Returns (n x K) matrix of scenario performances.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    Z = rng.normal(0, sigma_Z, K)  # Common factor
    P = np.zeros((n, K))
    for i in range(n):
        si = df.iloc[i]["sigma"]
        bi = df.iloc[i]["beta"]
        # Idiosyncratic variance = total variance minus systematic variance
        var_eta = max(si**2 - (bi * sigma_Z)**2, 0.01)
        eta = rng.normal(0, np.sqrt(var_eta), K)
        P[i, :] = df.iloc[i]["mu"] + bi * Z + eta
    return P


# =====================================================================
# VERIFICATION (runs when executed directly)
# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("PLAYER POOL: 2024-25 PREMIER LEAGUE (100 PLAYERS)")
    print("=" * 80)

    print(f"\nTotal players: {len(df)}")
    for pos in ["GK", "DF", "MF", "FW"]:
        sub = df[df["position"] == pos]
        print(f"  {pos}: {len(sub)} players, mu range [{sub['mu'].min():.1f}, {sub['mu'].max():.1f}], "
              f"wage range [{sub['wage_kpw'].min()}, {sub['wage_kpw'].max()}]k")

    print(f"\nBudget: {B}k | Squad size: {S}")
    top11 = df.nlargest(11, "mu")
    print(f"Top 11 by mu cost: {top11['wage_kpw'].sum()}k (ratio to budget: {top11['wage_kpw'].sum()/B:.2f}x)")

    print(f"\nSigma range: [{df['sigma'].min():.1f}, {df['sigma'].max():.1f}], mean={df['sigma'].mean():.2f}")
    print(f"Beta range:  [{df['beta'].min():.1f}, {df['beta'].max():.1f}], mean={df['beta'].mean():.2f}")

    # Tension checks
    print("\n--- TENSION CHECKS ---")
    # Budget tension: can't afford all best players
    top_fw = df[df["position"] == "FW"].nlargest(3, "mu")
    top_mf = df[df["position"] == "MF"].nlargest(4, "mu")
    top_df_ = df[df["position"] == "DF"].nlargest(3, "mu")
    top_gk = df[df["position"] == "GK"].nlargest(1, "mu")
    dream = pd.concat([top_gk, top_df_, top_mf, top_fw])
    print(f"Dream 11 cost: {dream['wage_kpw'].sum()}k vs budget {B}k -> {'OVER' if dream['wage_kpw'].sum() > B else 'UNDER'}")

    # Variance tension: high-mu players should have higher sigma
    high_mu = df.nlargest(15, "mu")
    low_mu = df.nsmallest(15, "mu")
    print(f"Top 15 mu avg sigma: {high_mu['sigma'].mean():.2f} vs bottom 15: {low_mu['sigma'].mean():.2f}")

    # Correlation tension: correlated vs independent scenarios should differ
    P_ind = generate_independent_gaussian(df, 1000, seed=0)
    P_cor = generate_correlated_factor(df, 1000, seed=0)
    # Check tail behaviour for a high-beta squad
    high_beta_idx = df.nlargest(11, "beta").index
    perf_ind = P_ind[high_beta_idx, :].sum(axis=0)
    perf_cor = P_cor[high_beta_idx, :].sum(axis=0)
    print(f"High-beta squad 5th percentile: ind={np.percentile(perf_ind, 5):.1f}, cor={np.percentile(perf_cor, 5):.1f}")

    # Export CSV
    df.to_csv("players_100.csv", index=False)
    print(f"\nExported players_100.csv ({len(df)} rows)")
    print("\nAll checks passed. Ready for optimisation.")
