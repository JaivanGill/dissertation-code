 Multi-criteria Optimisation under Uncertainty in Football Analytics

Jaivan Gill · University of Birmingham · Student ID: 2600501  
3RSM Research Skills in Mathematics Dissertation, 2025–26


 Overview

This repository contains the Python implementation for the numerical experiments in the dissertation *"Multi-criteria Optimisation under Uncertainty in Football Analytics."* The project formulates squad selection as a binary integer programme with multiple objectives, expected performance and Conditional Value at Risk (CVaR) and solves it using ε-constraint scalarisation over finite scenarios.
 Repository Structure

| File | Description |
|---|---|
| `parameters.py` | Player pool data (100 players), wages, volatility parameters, systemic exposure coefficients, and structural constraints |
| `optimisation.py` | MILP formulations, scenario generation, ε-constraint Pareto front construction (Algorithm 1 in the dissertation), and weighted-sum scalarisation |

 Data Sources and Parameter Estimation

 Player Pool

The pool consists of n = 100 players drawn from the highest-rated players by [FotMob](https://www.fotmob.com/) average match rating in the 2024–25 Premier League season:

- 27 forwards, 35 midfielders, 30 defenders (12 centre-backs, 18 full-backs), 8 goalkeepers

 Expected Contribution (μᵢ)

For each outfield player *i*, the baseline expected contribution is computed as:

```
μᵢ = (npxGᵢ + xAᵢ) × (90 / minᵢ) × 38
```

where `npxGᵢ` is non penalty expected goals per 90 minutes, `xAᵢ` is expected assists per 90, and `minᵢ` is total minutes played. This projects per 90 attacking output to a full 38 match season. For goalkeepers, `μᵢ` is derived analogously from goals prevented per 90 (post shot xG minus goals conceded, per 90, scaled to 38 matches).

Sources: FotMob (2024–25 season data), FBref (supplementary per-90 statistics).

 Wage Data (cᵢ)

Weekly wage estimates in thousands of pounds (£k/week), sourced from [Capology](https://www.capology.com/) and [FBref](https://fbref.com/) estimates for the 2024–25 season.

 Volatility Parameters (σᵢ)

Player specific standard deviations capturing season-to-season output variation. Estimated from:

1. Historical output variation across 2021–22, 2022–23, and 2023–24 seasons using FotMob data (npxG + xA per 90, scaled to 38 matches).
2. Injury adjustment: Players with significant injury history (source: [Transfermarkt](https://www.transfermarkt.co.uk/) injury records) received upward σᵢ adjustments of 15–30%, reflecting the increased uncertainty in their availability and output.
3. Age adjustment: Players aged ≥ 30 received a 10–20% upward adjustment to reflect the increased variability associated with physical decline, while players aged ≤ 22 received a 5–15% upward adjustment for developmental volatility.

Where fewer than three historical seasons were available (e.g. newly promoted players), σᵢ was estimated using positional averages supplemented by scouting level qualitative assessment.

 Systemic Exposure Parameters (βᵢ)

The factor loading βᵢ ≥ 0 measures player *i*'s sensitivity to the common shock Z(ξ) in the one factor model (Equation 12 in the dissertation). These were estimated from:

1. Tactical role analysis: Players in rigid tactical systems (e.g. wing-backs in a 3-4-3) received higher βᵢ values (0.4–0.6) due to greater dependence on team structure. Players in more autonomous roles (e.g. free roaming attacking midfielders, penalty box strikers) received lower values (0.1–0.3).
2. Multi manager evidence: Players who maintained output across multiple managers were assigned lower βᵢ (suggesting intrinsic quality less dependent on system), while those whose output dropped significantly under a new manager received higher values.
3. System dependency indicators: Players whose xG output is heavily driven by team created chances (high xG per shot, low shot volume) received higher βᵢ than players who generate their own shooting opportunities.


 Experimental Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Squad size *S* | 11 | Starting XI selection |
| Wage budget *B* | £1,500k/week | Below Big Six average (~£3,100k), forcing genuine trade-offs |
| Scenario count *K* | 500 | Balances computational cost with SAA convergence (O(K⁻¹ᐟ²) error rate) |
| CVaR level α | 0.95 | Standard tail risk threshold; focuses on worst 5% of outcomes |
| Loss target τ | 200 | Approximate expected performance of a competitive squad |
| Pareto grid size *M* | 15 | ε-constraint grid points for front construction |
| Solver time limit | 5 minutes per MILP | CBC branch and cut solver |

 Positional Bounds

| Position | Lower | Upper |
|---|---|---|
| Goalkeeper (GK) | 1 | 1 |
| Defender (DF) | 3 | 5 |
| Midfielder (MF) | 2 | 4 |
| Forward (FW) | 2 | 4 |

An additional sub positional constraint requires at least 2 centre backs among the selected defenders.

 Solver

All MILPs are solved using the CBC (COIN OR Branch and Cut) solver via PuLP, with a 5 minute time limit per problem instance. CBC is an open source mixed integer linear programming solver suitable for the problem sizes encountered here (100 binary variables, 500 scenario constraints).

Reproducing Results

```bash
pip install pulp numpy scipy
python optimisation.py
```

Output includes:
- Deterministic, Mean MAD, and Mean CVaR squad selections (RQ1)
- Independent and correlated Pareto fronts (RQ2)
- ε-constraint vs weighted-sum comparison (RQ3)
- All figures used in the dissertation

