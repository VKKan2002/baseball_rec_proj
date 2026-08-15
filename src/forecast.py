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
BAT_FEATURES = ["pa", "h", "hr", "bb", "so", "sb", "obp", "slg"]
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


def _train_models(df: pd.DataFrame):
    df = df.replace([np.inf, -np.inf], np.nan)
    bat_df = df[df["role"] == "bat"]
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


def _predict(df: pd.DataFrame, bat_model, pitch_model) -> pd.DataFrame:
    bat = df[df["role"] == "bat"].copy()
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
    bat_model, pitch_model = _train_models(training_data)
    projections = _load_projections()
    predictions = _predict(projections, bat_model, pitch_model)
    _write(predictions)


if __name__ == "__main__":
    main()
