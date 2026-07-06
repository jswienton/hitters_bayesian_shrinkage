# hitters_bayesian_shrinkage
# Estimating a Hitter's True Talent from Small-Sample wOBA

A hierarchical Bayesian (PyMC) model that estimates MLB hitters' true-talent wOBA from
small-sample plate appearances. Uses empirical variance decomposition to properly weight
observation noise, filters out pitcher at-bats, and validates against next-season
performance — 67% MSE improvement over raw stats for sub-50 PA rookie call-ups.

## The problem

A rookie goes 4-for-11 in his first callup. Is he a .360 hitter, or did he just get lucky?
Front offices face this constantly with call-ups and midseason debuts — a handful of plate
appearances is mostly noise, but there's still real signal buried in it. Regressing everyone
to the league mean throws that signal away; trusting the raw small-sample rate overreacts to
noise. This project builds an estimator that does neither.

## Approach

Each hitter's true talent is treated as being drawn from a league-wide talent distribution.
What we *observe* is that talent plus sampling noise that shrinks as `1/sqrt(PA)`. The
posterior for each player is a precision-weighted blend of his own observed rate and the
league prior — the partial-pooling / shrinkage estimator that front-office analysts use, in
spirit, for early-season and small-sample evaluation.

## Data

2021–2025 player-seasons of `PA` and `wOBA`, pulled via `pybaseball` from Baseball Savant
(`statcast_batter_expected_stats`, which reports actual wOBA down to a single plate
appearance).

**Data quality fix:** the initial pull included pitcher at-bats — before 2022, NL pitchers
batted for themselves, and pitchers still occasionally pinch-hit or hit in extra-inning
double-switches. These are almost all tiny-PA, poor-wOBA seasons, clustering in exactly the
low-PA range this model is built to characterize for real position players, which would bias
the small-sample end of the talent distribution downward. Pitcher-seasons are identified via
`pitching_stats_bref` (matched on the shared MLBAM ID), `IP >= 10`, and excluded — with
two-way players (`PA >= 200`) explicitly kept. 113 of 3,227 player-seasons were removed.
Re-running the full pipeline post-fix changed the core estimates only marginally
(`obs_sd`: 0.474 → 0.446), which increased confidence that the original finding was real and
not an artifact of the contaminated data.

## Methodology

1. **Variance decomposition** — rather than assuming a per-PA noise constant, it's
   estimated directly: players are binned by PA, the variance of observed wOBA within each
   bin is computed, and that variance is regressed against `1/PA`. The intercept recovers
   the true talent variance; the slope recovers the per-PA observation variance. This
   matters — a naive guess based on league-wide wOBA spread understates per-PA noise by
   roughly 10x, since it conflates season-average spread (already averaged over hundreds of
   PA) with single-PA outcome variance (mostly outs worth 0, occasionally a home run worth
   ~2.1).
2. **Hierarchical Bayesian model** — each player's true talent `θ_i ~ Normal(mu_league,
   sigma_league)`; each observed wOBA `~ Normal(θ_i, obs_sd / sqrt(PA_i))`. `mu_league` and
   `sigma_league` are estimated jointly across ~3,100 player-seasons via PyMC's NUTS sampler
   — full partial pooling.
3. **Out-of-sample validation** — the real test of a "true talent" estimator: does the
   shrunk estimate (fit on Year N) predict Year N+1's *actual* wOBA better than just using
   the raw Year N rate?

## Results

| PA range   | n    | Raw MSE | Shrunk MSE | Improvement |
|------------|------|---------|------------|-------------|
| [11, 50)   | 166  | 0.01297 | 0.00431    | **66.8%**   |
| [50, 150)  | 312  | 0.00592 | 0.00450    | 23.9%       |
| [150, 300) | 417  | 0.00380 | 0.00301    | 20.8%       |
| [300, 800) | 1068 | 0.00201 | 0.00176    | 12.7%       |

The improvement is largest exactly where sample sizes are smallest, and shrinks as PA grows
— the pattern you'd expect from a correctly-calibrated shrinkage estimator, not an artifact
of overfitting.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook shrinkage_analysis.ipynb
```

Running the notebook top to bottom pulls fresh data via `pybaseball`, fits the model, and
regenerates every result and plot above.

## Applying the model to new players

Once fit, scoring a new player doesn't require re-running MCMC — the shrinkage estimator has
a closed-form solution given the fitted `mu_league`, `sigma_league`, and `obs_sd`:

```python
from shrinkage_model import shrink

estimate, uncertainty = shrink(pa=15, woba=0.410)
```

This is used to score current-season, small-sample call-ups in near real time as the model's
actual intended application — flagging which hot (or cold) small-sample starts are likely
signal versus noise.

## Next steps

- **Hierarchical priors by player type** — pool call-ups toward a "prospect" sub-distribution
  (informed by minor-league performance or prospect rankings) rather than the whole-league
  mean, since a well-regarded prospect's true talent prior shouldn't be the league-average
  hitter.
- **Component-level shrinkage** — apply the same model to BB%, K%, and ISO separately, each
  of which has a different signal-to-noise ratio and stabilizes at a different PA threshold,
  then recompose into wOBA rather than shrinking the aggregate metric directly.
- **Time-varying talent** — allow true talent to drift within a season via a random-walk or
  age-curve prior, since a rookie can genuinely improve as he adjusts to MLB pitching.
