"""Evaluate (true forecast): does season N's stats predict season N+1's WAR?

evaluate.py answers an easier question: given a completed season's REAL stat
line, can we reconstruct THAT SAME season's WAR? WAR is partly *defined* by
those same stats, so that's closer to solving a known equation than
forecasting -- and it's not the pipeline's actual job. The real job is:
given what a player has already done, predict what they'll do NEXT season
(that's exactly what forecast.py does with the 2026 cheat-sheet projections).

This script tests that harder, more honest version: train on season N's
stats -> season N+1's real WAR, for many (N, N+1) pairs, holding out the
most recent ones as a test set. This is also the test that can fairly
settle the era/whip/w/l vs so9/bb9/hr9 pitcher-feature question raised in
evaluate.py -- component/skill stats are supposed to be MORE predictive of a
pitcher's FUTURE performance, even though they lost the same-season
reconstruction test there (see that file's ablation for why).

Run:
    python -m src.evaluate_forecast
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBRegressor

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.db import engine  # noqa: E402
from src.evaluate import PITCH_FEATURES_SKILL, _add_pitch_rate_stats  # noqa: E402
from src.forecast import (  # noqa: E402
    BAT_FEATURES,
    DEF_FEATURE,
    PITCH_FEATURES,
    _attach_prior_defense,
    _load_defense_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("evaluate_forecast")

# Held out: predicting INTO these seasons is the test set (e.g. using 2024
# stats to predict 2025 WAR). Everything forecasting into an earlier season
# is training data.
TEST_TARGET_SEASONS = [2024, 2025]


# ------------------------------------------------------------------ data ----

def _load_lag1_pairs() -> pd.DataFrame:
    """One row per player-role who played in consecutive seasons N and N+1:
    season N's stats as features, season N+1's real WAR as the label.
    Also carries season N's own WAR (war_this_season), for the "assume they
    repeat last year" baseline.

    Players who didn't play the following season -- retired, injured, or a
    rookie's debut with no "next" year on record yet -- are naturally
    excluded by the join. This only forecasts RETURNING players, same as
    the real 2026 use case (the cheat sheet only covers rostered players).
    """
    query = """
        WITH war_by_season AS (
            SELECT mlbam_id, season, role, SUM(war) AS war
            FROM war_actuals
            WHERE war IS NOT NULL
            GROUP BY mlbam_id, season, role
        )
        SELECT
            ss.mlbam_id,
            ss.season AS stat_season,
            ss.season + 1 AS target_season,
            ss.role,
            ss.pa, ss.h, ss.hr, ss.bb, ss.so, ss.sb, ss.obp, ss.slg,
            ss.ip, ss.era, ss.whip, ss.so9, ss.gs, ss.w, ss.l, ss.sv,
            this_year.war AS war_this_season,
            next_year.war AS war
        FROM season_stats ss
        JOIN war_by_season this_year
            ON ss.mlbam_id = this_year.mlbam_id
            AND ss.season = this_year.season
            AND ss.role = this_year.role
        JOIN war_by_season next_year
            ON ss.mlbam_id = next_year.mlbam_id
            AND ss.season + 1 = next_year.season
            AND ss.role = next_year.role
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    return df.replace([np.inf, -np.inf], np.nan)


def _prepare(raw: pd.DataFrame, def_hist: pd.DataFrame) -> pd.DataFrame:
    """Attach the fielding feature for bat rows. Pass target_season (not
    stat_season) as the anchor: _attach_prior_defense finds the most recent
    def_hist season strictly before it, which resolves to stat_season's own
    real fielding value here -- the player's most recent known defensive
    performance at forecast time. That's NOT leakage: the target is season
    N+1's WAR, and stat_season's fielding is fully known before N+1 starts.
    """
    bat = _attach_prior_defense(raw[raw["role"] == "bat"].copy(), "target_season", def_hist)
    pit = raw[raw["role"] == "pitch"].copy()
    return pd.concat([bat, pit], ignore_index=True)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[~df["target_season"].isin(TEST_TARGET_SEASONS)]
    test = df[df["target_season"].isin(TEST_TARGET_SEASONS)]
    return train, test


# --------------------------------------------------------- predict/score ----

def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    model.fit(train[features], train["war"])
    return model.predict(test[features])


def _fit_predict_xgb(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Same imputation as _fit_predict (an apples-to-apples comparison of
    linear vs. tree-based, not also a comparison of missing-data handling).
    No StandardScaler -- tree splits don't care about feature scale.

    Shallow trees (max_depth=3) and a low learning rate on purpose: ~6-9k
    training rows and 6-9 features is a small, noisy tabular dataset (see
    evaluate.py/evaluate_forecast.py's own naive-baseline results -- WAR,
    especially pitching WAR, has a lot of real year-to-year volatility that
    isn't signal). A deep, high-capacity model is more likely to fit that
    noise than find real structure Ridge is missing.
    """
    model = make_pipeline(
        SimpleImputer(),
        XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
    )
    model.fit(train[features], train["war"])
    return model.predict(test[features])


def _fit_predict_elasticnet(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Still linear like Ridge, but ElasticNet can zero out a feature
    entirely (L1) instead of only shrinking it (Ridge's L2) -- checks
    whether some of the current features are dead weight. CV picks the
    regularization strength on the train fold only, same as tuning Ridge's
    alpha would be, so it's not an unfair handicap vs. the fixed-alpha Ridge.
    """
    model = make_pipeline(SimpleImputer(), StandardScaler(), ElasticNetCV(cv=5, random_state=0))
    model.fit(train[features], train["war"])
    return model.predict(test[features])


def _fit_predict_rf(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """A different tree-based approach than XGBoost: averages many
    independently-built trees instead of building them sequentially to
    correct each other's errors. Checks whether the pitch-model gain XGBoost
    found is a "trees vs. linear" story or specific to boosting."""
    model = make_pipeline(
        SimpleImputer(),
        RandomForestRegressor(n_estimators=300, max_depth=5, random_state=0),
    )
    model.fit(train[features], train["war"])
    return model.predict(test[features])


def _fit_predict_blend(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Average of the Ridge and XGBoost predictions. A linear model and a
    tree model tend to make different kinds of errors, so blending often
    beats either alone -- essentially free since both are already trained."""
    ridge_pred = _fit_predict(train, test, features)
    xgb_pred = _fit_predict_xgb(train, test, features)
    return (ridge_pred + xgb_pred) / 2


def _score(name: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    log.info("  %-34s MAE=%.3f  RMSE=%.3f  R^2=%.3f", name, mae, rmse, r2)
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def evaluate_role(prepared: pd.DataFrame, role: str, features: list[str]) -> list[dict]:
    role_df = prepared[prepared["role"] == role]
    train, test = _split(role_df)
    log.info(
        "[%s] train=%d rows (target seasons before %d), test=%d rows (target seasons %s)",
        role, len(train), min(TEST_TARGET_SEASONS), len(test), TEST_TARGET_SEASONS,
    )
    return [
        _score(f"{role}: naive_mean", test["war"], np.full(len(test), train["war"].mean())),
        _score(f"{role}: naive_repeat_last_season", test["war"], test["war_this_season"].to_numpy()),
        _score(f"{role}: ridge (true forecast)", test["war"], _fit_predict(train, test, features)),
    ]


def evaluate_defense_ablation(prepared: pd.DataFrame) -> list[dict]:
    bat = prepared[prepared["role"] == "bat"]
    train, test = _split(bat)
    without = [f for f in BAT_FEATURES if f != DEF_FEATURE]
    log.info("[defense ablation, true forecast] with vs without %s:", DEF_FEATURE)
    return [
        _score("bat: WITHOUT prior_def_runs", test["war"], _fit_predict(train, test, without)),
        _score("bat: WITH prior_def_runs", test["war"], _fit_predict(train, test, BAT_FEATURES)),
    ]


def evaluate_pitch_feature_swap(prepared: pd.DataFrame) -> list[dict]:
    pit = _add_pitch_rate_stats(prepared[prepared["role"] == "pitch"])
    train, test = _split(pit)
    log.info("[pitch feature swap, true forecast] era/whip/w/l vs so9/bb9/hr9:")
    return [
        _score("pitch: era/whip/w/l (current)", test["war"], _fit_predict(train, test, PITCH_FEATURES)),
        _score("pitch: so9/bb9/hr9 (skill-driven)", test["war"], _fit_predict(train, test, PITCH_FEATURES_SKILL)),
    ]


def evaluate_model_ablation(prepared: pd.DataFrame, role: str, features: list[str]) -> list[dict]:
    """Same features, same train/test split, five candidate models -- does
    anything beat plain Ridge, or is linear regression already capturing
    what real signal exists in this data?"""
    role_df = prepared[prepared["role"] == role]
    train, test = _split(role_df)
    log.info("[model ablation, %s] Ridge / ElasticNet / RandomForest / XGBoost / blend:", role)
    return [
        _score(f"{role}: Ridge (linear)", test["war"], _fit_predict(train, test, features)),
        _score(f"{role}: ElasticNet (linear, auto-tuned)", test["war"], _fit_predict_elasticnet(train, test, features)),
        _score(f"{role}: RandomForest (tree-based)", test["war"], _fit_predict_rf(train, test, features)),
        _score(f"{role}: XGBoost (tree-based)", test["war"], _fit_predict_xgb(train, test, features)),
        _score(f"{role}: Ridge+XGBoost blend", test["war"], _fit_predict_blend(train, test, features)),
    ]


# ------------------------------------------------------------------ main ----

def main() -> None:
    raw = _load_lag1_pairs()
    def_hist = _load_defense_history()
    prepared = _prepare(raw, def_hist)

    log.info("=== pitch model (season N stats -> season N+1 WAR) ===")
    evaluate_role(prepared, "pitch", PITCH_FEATURES)

    log.info("=== bat model (season N stats -> season N+1 WAR) ===")
    evaluate_role(prepared, "bat", BAT_FEATURES)

    log.info("=== defense feature ablation ===")
    evaluate_defense_ablation(prepared)

    log.info("=== pitch feature swap ablation ===")
    evaluate_pitch_feature_swap(prepared)

    log.info("=== model ablation: linear vs tree-based ===")
    evaluate_model_ablation(prepared, "bat", BAT_FEATURES)
    evaluate_model_ablation(prepared, "pitch", PITCH_FEATURES)


if __name__ == "__main__":
    main()
