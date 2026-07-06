import pymc as pm
import numpy as np
import pandas as pd
import arviz as az
from pybaseball import pitching_stats_bref

df = pd.read_csv("mlb_batting_2021_2025.csv")
df = df.dropna(subset=["wOBA", "PA"])
df = df[df["PA"] > 10].reset_index(drop=True)

# Drop pitcher-hitting seasons (pre-2022 NL pitchers batting, occasional pinch-hitting) so the
# small-PA end of the talent distribution isn't dragged down by pitchers, who are bad hitters.
# Two-way players (real hitting workload, PA >= 200) are kept regardless.
pitcher_seasons = set()
for year in range(2021, 2026):
    p = pitching_stats_bref(year)
    real_pitchers = p[p["IP"] >= 10]
    pitcher_seasons.update(zip(real_pitchers["mlbID"].astype(int), [year] * len(real_pitchers)))

df["is_pitcher_season"] = df.apply(lambda r: (r["player_id"], r["Season"]) in pitcher_seasons, axis=1)
df = df[(~df["is_pitcher_season"]) | (df["PA"] >= 200)].drop(columns=["is_pitcher_season"]).reset_index(drop=True)

woba = df["wOBA"].values
pa = df["PA"].values
n_players = len(df)

# Per-PA observation noise, estimated empirically via variance decomposition:
# binned Var(wOBA) regressed on 1/PA gives intercept = talent variance,
# slope = per-PA observation variance. See notebook for the derivation.
# talent_sd ~= 0.025 (informs sigma_league prior below), obs_sd ~= 0.474
obs_sd = 0.474

with pm.Model() as hierarchical_model:
    # Hyperpriors: league-wide distribution of "true talent"
    mu_league = pm.Normal("mu_league", mu=0.320, sigma=0.030)
    sigma_league = pm.HalfNormal("sigma_league", sigma=0.030)

    # Each player's true talent drawn from the league distribution
    true_talent = pm.Normal("true_talent", mu=mu_league, sigma=sigma_league, shape=n_players)

    # Observation noise scales down with more PA
    obs_sigma = obs_sd / np.sqrt(pa)

    observed = pm.Normal("observed", mu=true_talent, sigma=obs_sigma, observed=woba)

    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=True)

# Save results
df["shrunk_estimate"] = trace.posterior["true_talent"].mean(dim=["chain", "draw"]).values
df["shrunk_sd"] = trace.posterior["true_talent"].std(dim=["chain", "draw"]).values

df.to_csv("shrinkage_results.csv", index=False)
print(df[["Name", "Season", "PA", "wOBA", "shrunk_estimate", "shrunk_sd"]].sort_values("PA").head(20))
