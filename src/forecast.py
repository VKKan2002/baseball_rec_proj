from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

from src.db import engine

# Only columns available in BOTH season_stats (training) and projections_raw (inference)
DEF_FEATURE = "prior_def_runs"
BAT_FEATURES = ["pa", "h", "hr", "bb", "so", "sb", "obp", "slg", DEF_FEATURE]
PITCH_FEATURES = ["ip", "era", "whip", "gs", "w", "l", "sv", "so9"]


def _load_training_data() -> pd.DataFrame:
    query = """
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
            wa.war
        FROM season_stats ss
        JOIN war_actuals wa
            ON ss.mlbam_id = wa.mlbam_id
            AND ss.season  = wa.season
            AND ss.role    = wa.role
        WHERE wa.war IS NOT NULL
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

    merged = pd.merge_asof(
        left.sort_values(season_col),
        right.sort_values("_def_season"),
        left_on=season_col, right_on="_def_season",
        by="mlbam_id", direction="backward", allow_exact_matches=False,
    )
    return merged.drop(columns="_def_season")


def _train_models(df: pd.DataFrame, def_hist: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    bat_df = df[df["role"] == "bat"]
    bat_df = _attach_prior_defense(bat_df, "season", def_hist)
    pit_df = df[df["role"] == "pitch"]

    bat_model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
    bat_model.fit(bat_df[BAT_FEATURES], bat_df["war"])

    pitch_model = make_pipeline(SimpleImputer(), StandardScaler(), Ridge())
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
            r.mlbam_id,
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
            pr.sv
        FROM projections_raw pr
        JOIN resolutions r ON r.projection_raw_id = pr.id
        WHERE r.mlbam_id IS NOT NULL
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _predict(df: pd.DataFrame, bat_model, pitch_model, def_hist: pd.DataFrame) -> pd.DataFrame:
    bat = df[df["role"] == "bat"].copy()
    bat = _attach_prior_defense(bat, "season", def_hist)
    pit = df[df["role"] == "pitch"].copy()

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
    bat_model, pitch_model = _train_models(training_data, def_hist)
    projections = _load_projections()
    predictions = _predict(projections, bat_model, pitch_model, def_hist)
    _write(predictions)


if __name__ == "__main__":
    main()
