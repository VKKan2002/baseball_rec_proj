"""Ingest layer: pull raw sources -> parquet snapshots -> Postgres.

Sources
-------
- Chadwick Bureau register (via pybaseball)      -> players
- Baseball-Reference bWAR bat + pitch (pybaseball) -> war_actuals
- Mr Cheat Sheet 2026 CSVs (data/raw/)           -> projections_raw

Every fact table carries ``as_of_date`` so historical snapshots are preserved
and re-running for the same date is idempotent (delete-by-date then insert).

Run:
    python -m src.ingest
"""
from __future__ import annotations

import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pybaseball as pyb
from sqlalchemy import text

# allow `python src/ingest.py` in addition to `python -m src.ingest`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import RAW_DIR, SNAPSHOTS_DIR  # noqa: E402
from src.db import engine  # noqa: E402
from src.models import Base, Player, ProjectionRaw, SeasonStats, WarActual  # noqa: E402

# bref will 429/temporarily-block a burst of requests with no delay between
# them (this is what killed 2020-2025 previously). Cache successful pulls to
# disk so a rerun after a mid-loop failure doesn't re-hit years we already
# have, and pace requests + retry with backoff on the ones that do fail.
pyb.cache.enable()

_REQUEST_DELAY_SEC = 5.0
_MAX_RETRIES = 4
_BACKOFF_BASE_SEC = 20.0

TRAIN_SEASONS = range(2014, 2026)  # 2014..2025 inclusive, ~12 years for the WAR estimator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("ingest")


# ---------------------------------------------------------------- helpers ----

def _snapshot(df: pd.DataFrame, name: str, as_of: date) -> Path:
    out = SNAPSHOTS_DIR / f"{name}__{as_of.isoformat()}.parquet"
    df.to_parquet(out, index=False)
    log.info("snapshot %-18s rows=%-7d -> %s", name, len(df), out.name)
    return out


def _replace_for_date(table_name: str, as_of: date, date_col: str = "as_of_date") -> None:
    """Delete rows for this as_of_date so re-runs are idempotent."""
    with engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {table_name} WHERE {date_col} = :d"),
            {"d": as_of},
        )


def _with_retry(fn, *args, label: str, **kwargs):
    """Call fn(*args, **kwargs), retrying on failure with exponential backoff.

    bref throttles/blocks bursts of requests rather than returning a clean
    429, so failures here are treated as rate-limit-shaped regardless of the
    exact exception type.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt == _MAX_RETRIES:
                break
            wait = _BACKOFF_BASE_SEC * (2 ** (attempt - 1))
            log.warning(
                "%s failed (attempt %d/%d): %s -- backing off %.0fs",
                label, attempt, _MAX_RETRIES, exc, wait,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def _write(df: pd.DataFrame, table_name: str, as_of: date | None) -> None:
    if df.empty:
        log.warning("no rows for %s", table_name)
        return
    if as_of is not None:
        _replace_for_date(table_name, as_of)
    df.to_sql(table_name, engine, if_exists="append", index=False, chunksize=5000)
    log.info("wrote     %-18s rows=%-7d as_of=%s", table_name, len(df), as_of)


# --------------------------------------------------------------- players -----

def fetch_players() -> pd.DataFrame:
    """Chadwick Bureau -> Player rows. Full replace on every run."""
    log.info("fetching Chadwick register...")
    raw = pyb.chadwick_register()
    df = raw.rename(columns={
        "key_mlbam":   "mlbam_id",
        "name_first":  "first_name",
        "name_last":   "last_name",
    })
    df = df[df["mlbam_id"].notna() & df["first_name"].notna() & df["last_name"].notna()].copy()
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    # Chadwick uses -1 as a sentinel for "no MLBAM id" (minor leaguers etc.);
    # drop them plus any duplicates so the PK holds.
    df = df[df["mlbam_id"] > 0].drop_duplicates(subset=["mlbam_id"])
    # birth_year / primary_position are nullable in the schema and not present
    # in Chadwick -- omit them from the insert so Postgres uses NULL defaults.
    return df[["mlbam_id", "first_name", "last_name"]]


def load_players() -> None:
    df = fetch_players()
    _snapshot(df, "players", date.today())
    # CASCADE because `resolutions` (and future contracts/rosters/projections)
    # hold FKs to players.mlbam_id. Those tables are all fully derivable, so
    # a full rebuild of players + downstream is safe. Re-run `python -m src.resolve`
    # afterwards to repopulate resolutions.
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE players CASCADE"))
    _write(df, Player.__tablename__, as_of=None)


# ------------------------------------------------------------ war actuals ----

_BWAR_COMMON = [
    "mlb_ID", "year_ID", "stint_ID",
    "name_common", "player_ID", "team_ID", "lg_ID",
    "G", "salary", "WAR", "WAR_rep", "WAA",
]

# fielding runs above average -- bwar_bat() only, bwar_pitch() has no
# equivalent column (fielding isn't broken out separately for pitchers here).
_BWAR_BAT_ONLY = ["runs_above_avg_def"]

_BWAR_RENAME = {
    "mlb_ID":              "mlbam_id",
    "year_ID":             "season",
    "stint_ID":            "stint",
    "player_ID":           "bbref_id",
    "team_ID":             "team_id",
    "lg_ID":               "league_id",
    "G":                   "games",
    "salary":              "salary_usd",
    "WAR":                 "war",
    "WAR_rep":             "war_rep",
    "WAA":                 "waa",
    "runs_above_avg_def":  "def_runs",
}


def fetch_war_actuals(as_of: date) -> pd.DataFrame:
    """Baseball-Reference bWAR bat + pitch, unioned with a `role` discriminator."""
    log.info("fetching bwar_bat...")
    bat = pyb.bwar_bat()[_BWAR_COMMON + _BWAR_BAT_ONLY].copy()
    bat["role"] = "bat"

    log.info("fetching bwar_pitch...")
    pit = pyb.bwar_pitch()[_BWAR_COMMON].copy()
    pit["runs_above_avg_def"] = float("nan")
    pit["role"] = "pitch"

    df = pd.concat([bat, pit], ignore_index=True)
    df = df.rename(columns=_BWAR_RENAME)

    df = df[df["mlbam_id"].notna()].copy()
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    df["season"] = df["season"].astype(int)
    df["stint"] = df["stint"].fillna(1).astype(int)
    df["as_of_date"] = as_of
    return df


def load_war_actuals(as_of: date) -> None:
    df = fetch_war_actuals(as_of)
    _snapshot(df, "war_actuals", as_of)
    _write(df, WarActual.__tablename__, as_of=as_of)


# --------------------------------------------------------- season stats ----

_BREF_BAT_RENAME = {
    "Age": "age", "Tm": "team_id", "G": "games", "PA": "pa", "AB": "ab",
    "H": "h", "HR": "hr", "R": "r", "RBI": "rbi", "BB": "bb", "SO": "so",
    "HBP": "hbp", "SB": "sb", "CS": "cs", "2B": "doubles", "3B": "triples",
    "BA": "ba", "OBP": "obp", "SLG": "slg", "OPS": "ops", "mlbID": "mlbam_id",
}

_BREF_PITCH_RENAME = {
    "Age": "age", "Tm": "team_id", "G": "games", "GS": "gs",
    "W": "w", "L": "l", "SV": "sv", "IP": "ip",
    "H": "h", "R": "r", "ER": "er", "BB": "bb", "SO": "so", "HR": "hr",
    "HBP": "hbp", "ERA": "era", "WHIP": "whip", "SO9": "so9", "SO/W": "so_bb",
    "BF": "bf", "mlbID": "mlbam_id",
}


def _clean_bref(df: pd.DataFrame, rename: dict[str, str], dedupe_by: str) -> pd.DataFrame:
    """Filter to majors, rename, drop bad mlbID rows, keep highest-volume row per player."""
    df = df[df["Lev"].fillna("").str.startswith("Maj")].copy()
    df = df.rename(columns=rename)
    df["mlbam_id"] = pd.to_numeric(df["mlbam_id"], errors="coerce")
    df = df.dropna(subset=["mlbam_id"]).copy()
    df["mlbam_id"] = df["mlbam_id"].astype(int)
    # Some players have multiple rows (mid-season trade). Keep the row with the
    # most PA (bat) or IP (pitch); bref sometimes provides a TOT row which
    # naturally wins the sort.
    if dedupe_by in df.columns:
        df = df.sort_values(dedupe_by, ascending=False, na_position="last")
    return df.drop_duplicates(subset=["mlbam_id"], keep="first")


def _fetch_year_bat(year: int) -> pd.DataFrame:
    df = _clean_bref(pyb.batting_stats_bref(year), _BREF_BAT_RENAME, "pa")
    df["season"] = year
    df["role"] = "bat"
    return df


def _fetch_year_pitch(year: int) -> pd.DataFrame:
    df = _clean_bref(pyb.pitching_stats_bref(year), _BREF_PITCH_RENAME, "ip")
    df["season"] = year
    df["role"] = "pitch"
    return df


def fetch_season_stats(as_of: date) -> pd.DataFrame:
    """Pull batting_stats_bref + pitching_stats_bref for TRAIN_SEASONS."""
    frames: list[pd.DataFrame] = []
    failed_years: list[int] = []
    for year in TRAIN_SEASONS:
        log.info("fetching bref bat + pitch for %d...", year)
        try:
            frames.append(_with_retry(_fetch_year_bat, year, label=f"batting_stats_bref({year})"))
            time.sleep(_REQUEST_DELAY_SEC)
            frames.append(_with_retry(_fetch_year_pitch, year, label=f"pitching_stats_bref({year})"))
        except Exception as exc:  # noqa: BLE001
            log.error("giving up on %d after retries: %s", year, exc)
            failed_years.append(year)
        time.sleep(_REQUEST_DELAY_SEC)

    if failed_years:
        log.error("season_stats: failed years after retries: %s", failed_years)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["as_of_date"] = as_of
    valid = {c.name for c in SeasonStats.__table__.columns}
    return df[[c for c in df.columns if c in valid]]


def load_season_stats(as_of: date) -> None:
    df = fetch_season_stats(as_of)
    if df.empty:
        log.warning("no season_stats rows fetched")
        return
    _snapshot(df, "season_stats", as_of)
    _write(df, SeasonStats.__tablename__, as_of=as_of)


# ------------------------------------------------------- projections raw ----

_HIT_RENAME = {
    "PLAYER": "player_name", "AGE": "age", "TM": "team",
    "MAIN POS": "primary_pos", "OTHER POS": "other_pos",
    "PA": "pa", "AB": "ab", "H": "h", "HR": "hr", "BB": "bb",
    "K": "so", "SB": "sb",
    "AVG": "avg", "OBP": "obp", "SLG": "slg", "OPS": "ops",
}

_PIT_RENAME = {
    "PLAYER": "player_name", "AGE": "age", "TM": "team",
    "MAIN POS": "primary_pos", "OTHER POS": "other_pos",
    "W": "w", "L": "l", "IP": "ip", "ER": "er", "SV": "sv",
    "ERA": "era", "WHIP": "whip", "K/9": "k_per_9", "GS": "gs",
}


def _load_cheatsheet(path: Path, rename: dict[str, str], role: str) -> pd.DataFrame:
    df = pd.read_csv(path).rename(columns=rename)
    df["role"] = role
    keep = list(rename.values()) + ["role"]
    return df[keep]


def fetch_projections(as_of: date, season: int = 2026) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    hitters = RAW_DIR / f"{season}-hitters.csv"
    pitchers = RAW_DIR / f"{season}-pitchers.csv"

    if hitters.exists():
        frames.append(_load_cheatsheet(hitters, _HIT_RENAME, "bat"))
    else:
        log.warning("missing %s", hitters)
    if pitchers.exists():
        frames.append(_load_cheatsheet(pitchers, _PIT_RENAME, "pitch"))
    else:
        log.warning("missing %s", pitchers)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["source"] = "mr_cheat_sheet"
    df["season"] = season
    df["as_of_date"] = as_of

    valid = {c.name for c in ProjectionRaw.__table__.columns if c.name != "id"}
    return df[[c for c in df.columns if c in valid]]


def load_projections(as_of: date, season: int = 2026) -> None:
    df = fetch_projections(as_of, season=season)
    if df.empty:
        return
    _snapshot(df, "projections_raw", as_of)
    _write(df, ProjectionRaw.__tablename__, as_of=as_of)


# ------------------------------------------------------------------ main ----

def _ensure_schema() -> None:
    """create_all only creates missing tables, not columns added to existing
    ones -- no Alembic in this project, so patch new columns in by hand."""
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE war_actuals ADD COLUMN IF NOT EXISTS def_runs DOUBLE PRECISION"
        ))


def main(as_of: date | None = None) -> None:
    as_of = as_of or date.today()
    log.info("ingest starting as_of=%s", as_of)

    _ensure_schema()

    load_players()
    load_war_actuals(as_of)
    load_season_stats(as_of)
    load_projections(as_of)

    log.info("ingest complete")


if __name__ == "__main__":
    main()
