from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from xgboost import XGBRegressor

from src.db import engine

# Only columns available in BOTH season_stats (training) and projections_raw (inference)
DEF_FEATURE = "prior_def_runs"
SC_BAT_FEATURES = ["sc_xwoba", "sc_barrel_pct", "sc_exit_velo"]
SC_PITCH_FEATURES = ["sc_xwoba", "sc_barrel_pct"]
BAT_FEATURES = [
    "pa", "h", "hr", "bb", "so", "sb", "obp", "slg", DEF_FEATURE,
    "age", "age_sq", "r", "rbi", "doubles", "triples", "cs", "prior_war",
    *SC_BAT_FEATURES,
]
PITCH_FEATURES = ["ip", "era", "whip", "gs", "w", "l", "sv", "so9", "prior_war", *SC_PITCH_FEATURES]


def _load_training_data() -> pd.DataFrame:
    # war_actuals has one row per STINT, so a player traded mid-season has
    # multiple rows for the same (mlbam_id, season, role). season_stats is
    # already one row per (mlbam_id, season, role) (see _clean_bref), so
    # joining straight to war_actuals would fan out and duplicate that row
    # once per stint. Sum war per season first so the join stays 1:1.
    query = """
        WITH war_by_season AS (
            SELECT mlbam_id, season, role, SUM(war) AS war
            FROM war_actuals
            WHERE war IS NOT NULL
            GROUP BY mlbam_id, season, role
        )
        SELECT
            ss.mlbam_id,
            ss.season,
            ss.role,
            ss.pa,
            ss.h,
            ss.hr,
            ss.bb,
            ss.so,
            ss.sb,
            ss.obp,
            ss.slg,
            ss.ip,
            ss.era,
            ss.whip,
            ss.so9,
            ss.gs,
            ss.w,
            ss.l,
            ss.sv,
            ss.age,
            ss.r,
            ss.rbi,
            ss.doubles,
            ss.triples,
            ss.cs,
            wa.war
        FROM season_stats ss
        JOIN war_by_season wa
            ON ss.mlbam_id = wa.mlbam_id
            AND ss.season  = wa.season
            AND ss.role    = wa.role
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _load_defense_history() -> pd.DataFrame:
    """One row per (mlbam_id, season): summed fielding runs above average
    across stints. Only role='bat' has this (see _BWAR_BAT_ONLY in ingest.py)."""
    query = """
        SELECT mlbam_id, season, SUM(def_runs) AS def_runs
        FROM war_actuals
        WHERE role = 'bat' AND def_runs IS NOT NULL
        GROUP BY mlbam_id, season
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _attach_prior_defense(df: pd.DataFrame, season_col: str, def_hist: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its player's most recent def_runs from a season
    strictly BEFORE season_col. def_runs is itself a component of the WAR
    target, so using the same season's value would leak the answer; using the
    most recent prior season also matches what's actually available at
    inference time, since the 2026 cheat-sheet CSVs don't project fielding.
    Players with no prior season (rookies) get NaN, filled by SimpleImputer.
    """
    left = df.copy()
    left["mlbam_id"] = left["mlbam_id"].astype("int64")
    left[season_col] = left[season_col].astype("int64")

    right = def_hist.copy()
    right["mlbam_id"] = right["mlbam_id"].astype("int64")
    right = right.rename(columns={"season": "_def_season", "def_runs": DEF_FEATURE})
    right["_def_season"] = right["_def_season"].astype("int64")

    merged = pd.merge_asof(
        left.sort_values(season_col),
        right.sort_values("_def_season"),
        left_on=season_col, right_on="_def_season",
        by="mlbam_id", direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns="_def_season")


def _load_prior_war() -> pd.DataFrame:
    """Total WAR per (mlbam_id, season, role) from actuals — used as a lag feature."""
    query = """
        SELECT mlbam_id, season, role, SUM(war) AS war
        FROM war_actuals
        WHERE war IS NOT NULL
        GROUP BY mlbam_id, season, role
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _attach_prior_war(df: pd.DataFrame, season_col: str, war_hist: pd.DataFrame) -> pd.DataFrame:
    """Join each row to its player-role's most recent WAR from a season
    strictly BEFORE season_col. Rookies or players with no prior season get
    NaN, filled by SimpleImputer."""
    left = df.copy()
    left["mlbam_id"] = left["mlbam_id"].astype("int64")
    left[season_col] = left[season_col].astype("int64")

    right = war_hist.copy()
    right["mlbam_id"] = right["mlbam_id"].astype("int64")
    right = right.rename(columns={"season": "_war_season", "war": "prior_war"})
    right["_war_season"] = right["_war_season"].astype("int64")

    merged = pd.merge_asof(
        left.sort_values(season_col),
        right.sort_values("_war_season"),
        left_on=season_col, right_on="_war_season",
        by=["mlbam_id", "role"], direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns="_war_season")


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_sq"] = df["age"] ** 2
    return df


def _load_statcast_history() -> pd.DataFrame:
    query = """
        SELECT mlbam_id, season, role, xwoba, barrel_pct, exit_velo_avg
        FROM statcast_stats
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _attach_statcast(df: pd.DataFrame, season_col: str, sc_hist: pd.DataFrame,
                     role: str, exact: bool) -> pd.DataFrame:
    """Join Statcast features for the given role onto df.

    exact=True: matches the same season (training — season N Statcast -> season N+1 WAR).
    exact=False: most recent season strictly before season_col (inference — 2026 gets 2025).
    """
    left = df.copy()
    left[season_col] = left[season_col].astype("int64")
    left["mlbam_id"] = left["mlbam_id"].astype("int64")

    right = sc_hist[sc_hist["role"] == role].drop(columns="role").copy()
    right["mlbam_id"] = right["mlbam_id"].astype("int64")
    right["season"] = right["season"].astype("int64")
    right = right.rename(columns={
        "season": "_sc_season",
        "xwoba": "sc_xwoba",
        "barrel_pct": "sc_barrel_pct",
        "exit_velo_avg": "sc_exit_velo",
    })

    merged = pd.merge_asof(
        left.sort_values(season_col),
        right.sort_values("_sc_season"),
        left_on=season_col, right_on="_sc_season",
        by="mlbam_id", direction="backward",
        allow_exact_matches=exact,
    )
    return merged.drop(columns="_sc_season")


def _train_models(df: pd.DataFrame, def_hist: pd.DataFrame, war_hist: pd.DataFrame,
                  sc_hist: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = _add_derived_features(df)
    bat_df = df[df["role"] == "bat"]
    bat_df = _attach_prior_defense(bat_df, "season", def_hist)
    bat_df = _attach_prior_war(bat_df, "season", war_hist)
    bat_df = _attach_statcast(bat_df, "season", sc_hist, "bat", exact=True)
    pit_df = df[df["role"] == "pitch"]
    pit_df = _attach_prior_war(pit_df, "season", war_hist)
    pit_df = _attach_statcast(pit_df, "season", sc_hist, "pitch", exact=True)

    # Batting: linear (Ridge) beat every alternative tested (ElasticNet,
    # RandomForest, XGBoost, a Ridge+XGBoost blend) -- see
    # EVALUATION_WALKTHROUGH.md's model-comparison ablation.
    bat_model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    bat_model.fit(bat_df[BAT_FEATURES], bat_df["war"])

    # Pitching: two independent tree-based models (RandomForest and
    # XGBoost) both beat Ridge by the same margin in a true N->N+1 forecast
    # (R^2 0.164 -> 0.219) -- strong evidence of real non-linear structure
    # a straight line can't capture, not a quirk of one algorithm. No
    # StandardScaler: tree splits don't care about feature scale. Shallow
    # trees (max_depth=3) + a low learning rate on purpose -- ~6-9k training
    # rows is a small, noisy dataset, so a high-capacity model is more
    # likely to fit noise than find structure Ridge is missing.
    pitch_model = make_pipeline(
        SimpleImputer(),
        XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=0),
    )
    pitch_model.fit(pit_df[PITCH_FEATURES], pit_df["war"])

    return bat_model, pitch_model


def _load_projections() -> pd.DataFrame:
    query = """
        SELECT
            pr.id,
            pr.player_name,
            pr.role,
            pr.team,
            pr.season,
            res.mlbam_id,
            pr.pa,
            pr.h,
            pr.hr,
            pr.bb,
            pr.so,
            pr.sb,
            pr.obp,
            pr.slg,
            pr.ip,
            pr.era,
            pr.whip,
            pr.k_per_9  AS so9,
            pr.gs,
            pr.w,
            pr.l,
            pr.sv,
            pr.age,
            pr.r,
            pr.rbi,
            pr.doubles,
            pr.triples,
            pr.cs
        FROM projections_raw pr
        JOIN resolutions res ON res.projection_raw_id = pr.id
        WHERE res.mlbam_id IS NOT NULL
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _predict(df: pd.DataFrame, bat_model, pitch_model, def_hist: pd.DataFrame,
             war_hist: pd.DataFrame, sc_hist: pd.DataFrame) -> pd.DataFrame:
    df = _add_derived_features(df)
    bat = df[df["role"] == "bat"].copy()
    bat = _attach_prior_defense(bat, "season", def_hist)
    bat = _attach_prior_war(bat, "season", war_hist)
    bat = _attach_statcast(bat, "season", sc_hist, "bat", exact=False)
    pit = df[df["role"] == "pitch"].copy()
    pit = _attach_prior_war(pit, "season", war_hist)
    pit = _attach_statcast(pit, "season", sc_hist, "pitch", exact=False)

    bat["projected_war"] = bat_model.predict(bat[BAT_FEATURES])
    pit["projected_war"] = pitch_model.predict(pit[PITCH_FEATURES])

    return pd.concat([bat, pit], ignore_index=True)


def _write(df: pd.DataFrame) -> None:
    out = df.copy()
    out["season"] = 2026
    out["system"] = "mr_cheat_sheet"
    out["as_of_date"] = date.today()
    out["projected_pa"] = out["pa"]
    out["projected_ip"] = out["ip"]

    keep = ["mlbam_id", "season", "system", "as_of_date",
            "projected_war", "projected_pa", "projected_ip"]
    out = out[keep]
    # two-way players appear in both bat and pitch CSVs; keep the higher-WAR row
    out = out.sort_values("projected_war", ascending=False).drop_duplicates(
        subset=["mlbam_id", "season", "system", "as_of_date"], keep="first"
    )

    with engine.begin() as conn:
        conn.execute(text("DELETE FROM projections WHERE season = 2026 AND system = 'mr_cheat_sheet'"))
    out.to_sql("projections", engine, if_exists="append", index=False)


def main() -> None:
    training_data = _load_training_data()
    def_hist = _load_defense_history()
    war_hist = _load_prior_war()
    sc_hist = _load_statcast_history()
    bat_model, pitch_model = _train_models(training_data, def_hist, war_hist, sc_hist)
    projections = _load_projections()
    predictions = _predict(projections, bat_model, pitch_model, def_hist, war_hist, sc_hist)
    _write(predictions)


if __name__ == "__main__":
    main()
