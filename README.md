# hitters_bayesian_shrinkage
Hierarchical Bayesian (PyMC) model that estimates MLB hitters' true-talent wOBA from small-sample plate appearances. Uses empirical variance decomposition to properly weight observation noise, filters out pitcher at-bats, and validates against next-season performance — 67% MSE improvement over raw stats for sub-50 PA rookie call-ups.
