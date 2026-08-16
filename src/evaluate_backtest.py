"""Evaluate the backtest: does the surplus-value pipeline actually pick out
good, cheap players, and does recommend.py's "fair trade" claim hold up
against what really happened?

evaluate_forecast.py already proves the WAR model itself has real predictive
power (season N stats -> season N+1 WAR, held out on 2024-2025). This script
carries that one step further, through the exact same valuation math
production uses (src/valuation.py's aging curve + $/WAR conversion +
real salary), and checks the two things that actually matter for the
product rather than the model in isolation:

  1. Surplus-value accuracy -- rank players by PREDICTED surplus value,
     computed using only data available before TARGET_SEASON, then
     correlate (Spearman) against what they were REALLY worth that season
     (actual realized WAR at actual real salary). Compared against a naive
     "assume they repeat last season" baseline run through the identical
     formula, so any lift is attributable to the model, not the valuation
     math itself.

  2. Fair-trade pairing accuracy -- recommend.py's specific promise is that
     players it pairs as "close surplus value" are a realistic, roughly-even
     swap. This runs the real find_fair_trades() on PREDICTED surplus
     values, then checks whether those pairs' REALIZED values also stayed
     close -- versus random pairs, as a baseline.

Both use TARGET_SEASON=2023 by default so all three years of the real
production surplus formula's horizon (2023/2024/2025 salary -- guaranteed
money, legitimately knowable in advance, unlike performance) have real data
to draw from. Training only ever sees stat_season -> target_season pairs
with target_season < TARGET_SEASON.

Known simplifications, stated rather than hidden:
  - Only year 1 (the season the model actually predicts) is scored against
    reality. Years 2-3 of the real surplus formula are carried forward by
    the aging curve, a stated non-learned assumption (see README) -- there's
    nothing here that legitimately validates a hand-picked curve against
    future-realized outcomes without conflating it with the model's own
    accuracy.
  - Historical season_stats/war_actuals don't carry a player's position
    (only the current-season cheat sheet recommend.py normally reads does),
    so the fair-trade pairing check runs with roster-fit filtering OFF
    (require_position_fit=False). It's evaluating the surplus-closeness
    claim specifically, not the position-fit heuristic, which has no
    historical data to test against here.

Run:
    python -m src.evaluate_backtest
    python -m src.evaluate_backtest --target-season 2022
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DOLLARS_PER_WAR  # noqa: E402
from src.db import engine  # noqa: E402
from src.evaluate_forecast import _fit_predict, _fit_predict_xgb, _load_lag1_pairs, _prepare  # noqa: E402
from src.forecast import BAT_FEATURES, PITCH_FEATURES, _load_defense_history  # noqa: E402
from src.recommend import find_fair_trades  # noqa: E402
from src.valuation import (  # noqa: E402
    LEAGUE_MINIMUM_SALARY,
    _load_year0_salaries,
    _project_war_path,
    compute_surplus_value,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("evaluate_backtest")

HORIZON_YEARS = 3
TOP_K = 20            # for precision@K and the fair-trade pairing check
RANDOM_SEED = 0


# --------------------------------------------------------------- loaders ----

def _split_by_target_season(df: pd.DataFrame, target_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["target_season"] < target_season]
    test = df[df["target_season"] == target_season]
    return train, test


def _load_names_and_teams(target_season: int) -> pd.DataFrame:
    """mlbam_id, name, team for players active in target_season. team_id in
    season_stats is a comma-joined list of full team names for players
    traded mid-season (e.g. "Boston,Colorado") -- fine here, since
    find_fair_trades only checks team_a != team_b, not a specific code.
    """
    query = """
        SELECT DISTINCT ss.mlbam_id, pl.first_name, pl.last_name, ss.team_id AS team
        FROM season_stats ss
        JOIN players pl ON pl.mlbam_id = ss.mlbam_id
        WHERE ss.season = :season
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params={"season": target_season})
    df["name"] = df["first_name"] + " " + df["last_name"]
    return df.drop(columns=["first_name", "last_name"]).drop_duplicates(subset=["mlbam_id"])


# --------------------------------------------------------- surplus value ----

def _salary_for(salaries_by_season: dict[int, pd.DataFrame], mlbam_id: int, season: int) -> float:
    """Real historical salary for (mlbam_id, season), falling back to league
    minimum -- same policy valuation.py uses for a missing year. No
    "estimated" flag here (unlike valuation.py): this script only ever
    scores year-1, so a fallback in a not-scored out-year can't leak into
    the correlation, it would only ever soften the predicted-vs-naive gap
    on the identical formula both sides share.
    """
    df = salaries_by_season.get(season)
    if df is None:
        return LEAGUE_MINIMUM_SALARY
    match = df[df["mlbam_id"] == mlbam_id]
    if match.empty or pd.isna(match["salary_usd"].iloc[0]):
        return LEAGUE_MINIMUM_SALARY
    return float(match["salary_usd"].iloc[0])


def _surplus_value(war_year1: float, age_at_target: int, mlbam_id: int, target_season: int,
                    salaries_by_season: dict[int, pd.DataFrame]) -> float:
    war_path = _project_war_path(war_year1, age_at_target)
    salaries = [
        _salary_for(salaries_by_season, mlbam_id, target_season + y)
        for y in range(HORIZON_YEARS)
    ]
    return compute_surplus_value(war_path, salaries)


# ------------------------------------------------------------------ main ----

def build_evaluation_frame(target_season: int) -> pd.DataFrame:
    raw = _load_lag1_pairs()
    def_hist = _load_defense_history()
    prepared = _prepare(raw, def_hist)

    frames = []
    for role, features, fit_fn in [
        ("bat", BAT_FEATURES, _fit_predict),
        ("pitch", PITCH_FEATURES, _fit_predict_xgb),
    ]:
        role_df = prepared[prepared["role"] == role]
        train, test = _split_by_target_season(role_df, target_season)
        if test.empty:
            log.warning("no %s rows for target_season=%d -- skipping", role, target_season)
            continue
        log.info("[%s] train=%d rows (target_season < %d), test=%d rows (target_season = %d)",
                  role, len(train), target_season, len(test), target_season)
        test = test.copy()
        test["predicted_war"] = fit_fn(train, test, features)
        frames.append(test)

    evaluated = pd.concat(frames, ignore_index=True)
    evaluated["age_at_target"] = evaluated["age"] + 1

    salaries_by_season = {
        y: _load_year0_salaries(y) for y in range(target_season, target_season + HORIZON_YEARS)
    }

    evaluated["predicted_surplus"] = [
        _surplus_value(row.predicted_war, row.age_at_target, row.mlbam_id, target_season, salaries_by_season)
        for row in evaluated.itertuples(index=False)
    ]
    evaluated["naive_surplus"] = [
        _surplus_value(row.war_this_season, row.age_at_target, row.mlbam_id, target_season, salaries_by_season)
        for row in evaluated.itertuples(index=False)
    ]

    year0_salary = salaries_by_season[target_season]
    evaluated = evaluated.merge(
        year0_salary.rename(columns={"salary_usd": "_salary_year0"}), on="mlbam_id", how="left",
    )
    evaluated["_salary_year0"] = evaluated["_salary_year0"].fillna(LEAGUE_MINIMUM_SALARY)
    evaluated["realized_value"] = evaluated["war"] * DOLLARS_PER_WAR - evaluated["_salary_year0"]

    names_teams = _load_names_and_teams(target_season)
    evaluated = evaluated.merge(names_teams, on="mlbam_id", how="left")
    evaluated = evaluated.dropna(subset=["name", "team"])

    # Two-way players (or a position player who pitched an inning in a
    # blowout) get one row per role -- same "keep the bigger role" policy
    # forecast.py's _write() uses for the live 2026 pipeline, so mlbam_id is
    # unique here same as it is downstream (find_fair_trades assumes one row
    # per player).
    return evaluated.sort_values("predicted_war", ascending=False).drop_duplicates(
        subset=["mlbam_id"], keep="first"
    )


# -------------------------------------------------- 1. surplus accuracy ----

def evaluate_surplus_accuracy(df: pd.DataFrame) -> None:
    log.info("=== 1. surplus-value accuracy (predicted pre-season vs. realized) ===")

    model_corr, _ = spearmanr(df["predicted_surplus"], df["realized_value"])
    naive_corr, _ = spearmanr(df["naive_surplus"], df["realized_value"])
    log.info("  Spearman(predicted_surplus, realized_value) = %.3f", model_corr)
    log.info("  Spearman(naive_surplus,     realized_value) = %.3f  (repeat-last-season baseline)", naive_corr)

    for label, col in [("model", "predicted_surplus"), ("naive", "naive_surplus")]:
        top_k = df.nlargest(TOP_K, col)
        hit_rate = (top_k["realized_value"] > 0).mean()
        log.info("  precision@%d (%s): %.0f%% of top-%d picks had positive realized value",
                  TOP_K, label, hit_rate * 100, TOP_K)
    league_rate = (df["realized_value"] > 0).mean()
    log.info("  league-wide base rate: %.0f%% of all %d players had positive realized value",
              league_rate * 100, len(df))


# ---------------------------------------------- 2. fair-trade pairing ----

def evaluate_fair_trade_pairing(df: pd.DataFrame) -> None:
    log.info("=== 2. fair-trade pairing accuracy (predicted-fair pairs vs. realized gap) ===")

    valuations = df.rename(columns={"predicted_surplus": "surplus_value"}).copy()
    valuations["primary_pos"] = None
    valuations["other_pos"] = None

    pairs = find_fair_trades(valuations, top_n=TOP_K, require_position_fit=False)
    realized = df.set_index("mlbam_id")["realized_value"]

    realized_gaps = [
        abs(realized.loc[row.mlbam_id_a] - realized.loc[row.mlbam_id_b])
        for row in pairs.itertuples(index=False)
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    all_ids = df["mlbam_id"].to_numpy()
    random_gaps = [
        abs(realized.loc[a] - realized.loc[b])
        for a, b in zip(rng.choice(all_ids, size=500), rng.choice(all_ids, size=500))
        if a != b
    ]

    log.info("  %d predicted-fair pairs -- median realized value gap: $%s",
              len(realized_gaps), f"{np.median(realized_gaps):,.0f}")
    log.info("  500 random pairs        -- median realized value gap: $%s",
              f"{np.median(random_gaps):,.0f}")


def main(target_season: int) -> None:
    df = build_evaluation_frame(target_season)
    log.info("evaluation frame: %d players, target_season=%d", len(df), target_season)

    evaluate_surplus_accuracy(df)
    evaluate_fair_trade_pairing(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-season", type=int, default=2023)
    args = parser.parse_args()
    main(target_season=args.target_season)
