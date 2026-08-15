"""Evaluate: backtest the forecast.py WAR model against real historical outcomes.

forecast.py trains on the full 2014-2025 history and predicts on the unseen
2026 projections -- which means, on its own, there's no way to know whether
those predictions are actually any good. "The top of the list is Ohtani and
Judge" is not evidence; a model that just rewards more plate appearances
would produce the same top-of-list.

This script gets real evidence by holding out recent *seasons* whose actual
WAR we already have on file (war_actuals), training only on the seasons
before them, and scoring predicted-vs-actual WAR on the held-out seasons.
Reuses forecast.py's exact feature sets and training data loaders, so this
evaluates the model that actually ships, not a lookalike.

Reports, separately for bat and pitch:
    - MAE / RMSE / R^2 for the Ridge model
    - the same metrics for two naive baselines -- if the model can't beat
      these, it isn't adding value over doing nothing
    - an ablation on the fielding feature (prior_def_runs): does removing
      it measurably hurt batting predictions, or was it just reshuffling
      rankings with no real accuracy gain?

Run:
    python -m src.evaluate
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.forecast import (  # noqa: E402
    BAT_FEATURES,
    DEF_FEATURE,
    PITCH_FEATURES,
    _attach_prior_defense,
    _load_defense_history,
    _load_training_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("evaluate")

# Held out entirely from training -- real WAR for these seasons is already
# known, so predictions on them can be scored directly instead of eyeballed.
TEST_SEASONS = [2024, 2025]


# ------------------------------------------------------------------ split ----

def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[~df["season"].isin(TEST_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)]
    return train, test


def _prepare(full: pd.DataFrame, def_hist: pd.DataFrame) -> pd.DataFrame:
    """Attach the lagged fielding feature to bat rows (mirrors forecast.py's
    _train_models). Pitch rows pass through unchanged -- PITCH_FEATURES
    never references it."""
    bat = _attach_prior_defense(full[full["role"] == "bat"].copy(), "season", def_hist)
    pit = full[full["role"] == "pitch"].copy()
    return pd.concat([bat, pit], ignore_index=True)


# ------------------------------------------------------------- predictors ----

def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    model.fit(train[features], train["war"])
    return model.predict(test[features])


def _naive_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: predict every test row as the train-set average WAR."""
    return np.full(len(test), train["war"].mean())


def _naive_last_season(role_df: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Baseline: predict each row as that player's PRIOR season's actual WAR
    (0 if they have none on record, e.g. a rookie debut). This is a fair
    comparison, not leakage -- by the time you'd predict a player's 2025
    season, their 2024 season has already been played and its WAR is known.

    Summed by (mlbam_id, season) because a mid-season trade means
    war_actuals has one row per stint -- summing gives the real full-season
    WAR instead of an arbitrary single stint's partial number.
    """
    lookup = role_df.groupby(["mlbam_id", "season"])["war"].sum()
    preds = []
    for row in test.itertuples(index=False):
        preds.append(lookup.get((row.mlbam_id, row.season - 1), 0.0))
    return np.array(preds)


# ------------------------------------------------------------------ score ----

def _score(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    log.info("  %-30s MAE=%.3f  RMSE=%.3f  R^2=%.3f", name, mae, rmse, r2)
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def evaluate_role(prepared: pd.DataFrame, role: str, features: list[str]) -> list[dict]:
    role_df = prepared[prepared["role"] == role]
    train, test = _split(role_df)
    log.info(
        "[%s] train=%d rows (seasons before %d), test=%d rows (seasons %s)",
        role, len(train), min(TEST_SEASONS), len(test), TEST_SEASONS,
    )
    return [
        _score(f"{role}: naive_mean", test["war"], _naive_mean(train, test)),
        _score(f"{role}: naive_last_season", test["war"], _naive_last_season(role_df, test)),
        _score(f"{role}: ridge (current model)", test["war"], _fit_predict(train, test, features)),
    ]


def evaluate_defense_ablation(prepared: pd.DataFrame) -> list[dict]:
    """Train the SAME bat model with and without prior_def_runs, score both
    on the identical held-out seasons. Isolates whether the fielding
    feature actually reduces prediction error, vs. just moving rankings
    around with no real accuracy gain."""
    bat = prepared[prepared["role"] == "bat"]
    train, test = _split(bat)
    without_features = [f for f in BAT_FEATURES if f != DEF_FEATURE]

    log.info("[defense ablation] bat model, with vs without %s:", DEF_FEATURE)
    return [
        _score("bat: WITHOUT prior_def_runs", test["war"], _fit_predict(train, test, without_features)),
        _score("bat: WITH prior_def_runs", test["war"], _fit_predict(train, test, BAT_FEATURES)),
    ]


# `w`/`l` are driven mostly by run support and bullpen, not the pitcher's own
# skill; era/whip mix in defense, sequencing, and BABIP luck. so9 (already a
# PITCH_FEATURES column) is the one existing feature that's already
# skill-driven -- bb9/hr9 are the same idea applied to command and the long
# ball, the same logic behind why FIP/xFIP exist as alternatives to ERA.
PITCH_FEATURES_SKILL = ["ip", "so9", "bb9", "hr9", "gs", "sv"]


def _add_pitch_rate_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["bb9"] = out["bb"] / out["ip"] * 9
    out["hr9"] = out["hr"] / out["ip"] * 9
    return out.replace([np.inf, -np.inf], np.nan)


def evaluate_pitch_feature_swap(prepared: pd.DataFrame) -> list[dict]:
    """Does swapping the results/context-driven features (era, whip, w, l)
    for skill-driven rate stats (bb9, hr9, alongside the already-skill-driven
    so9) improve pitcher predictions, or hurt them?"""
    pit = _add_pitch_rate_stats(prepared[prepared["role"] == "pitch"])
    train, test = _split(pit)

    log.info("[pitch feature swap] era/whip/w/l (current) vs so9/bb9/hr9 (skill-driven):")
    return [
        _score("pitch: era/whip/w/l (current)", test["war"], _fit_predict(train, test, PITCH_FEATURES)),
        _score("pitch: so9/bb9/hr9 (skill-driven)", test["war"], _fit_predict(train, test, PITCH_FEATURES_SKILL)),
    ]


# ------------------------------------------------------------------ main ----

def main() -> None:
    full = _load_training_data().replace([np.inf, -np.inf], np.nan)
    def_hist = _load_defense_history()
    prepared = _prepare(full, def_hist)

    log.info("=== pitch model ===")
    evaluate_role(prepared, "pitch", PITCH_FEATURES)

    log.info("=== bat model ===")
    evaluate_role(prepared, "bat", BAT_FEATURES)

    log.info("=== defense feature ablation (bat only) ===")
    evaluate_defense_ablation(prepared)

    log.info("=== pitch feature swap ablation ===")
    evaluate_pitch_feature_swap(prepared)


if __name__ == "__main__":
    main()
