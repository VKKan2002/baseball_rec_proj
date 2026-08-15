"""Valuation: turn a year-1 WAR projection into a multi-year surplus value.

Surplus value answers: "if we acquired this player, how much value do we get
over the horizon, net of what we pay them?" It's what ranks *acquisition
targets*, not talent -- Mike Trout is the best player in baseball and a
terrible trade target, because his salary already captures his value.

    surplus = sum over the horizon of:
        (projected_WAR_in_year_y * DOLLARS_PER_WAR / (1 + DISCOUNT_RATE)^y)
        - salary_in_year_y

Where:
    - year 0 = the year-1 WAR already sitting in `projections`
      (forecast.py's Ridge output -- it's already a single-season estimate,
      DO NOT run it through the aging curve, only years 1+ get aged)
    - years 1+ = year-0 WAR carried forward by an aging curve
    - salary_y = the player's expected salary that season, from
      data/manual_contracts.csv (no real contract data source exists yet --
      same manual-curation pattern as data/manual_overrides.csv)

Run:
    python -m src.valuation
"""
from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import DOLLARS_PER_WAR, DISCOUNT_RATE  # noqa: E402
from src.db import engine  # noqa: E402
from src.models import Base, Valuation  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("valuation")

# 3-year horizon: projection accuracy degrades past ~3 years (CONTEXT_PRIMER.md).
HORIZON_YEARS = 3

CONTRACTS_PATH = _ROOT / "data" / "manual_contracts.csv"

# Fallback for a horizon-year with no known salary (no war_actuals row, no
# manual_contracts.csv entry -- e.g. a pre-arb player's actual future salary
# isn't public). Using MLB's league minimum as a floor is a stated
# assumption, not a real number -- rows that hit this are flagged via
# `salary_estimated` in the output rather than silently blended in.
LEAGUE_MINIMUM_SALARY = 760_000

PEAK_AGE = 27


# ---------------------------------------------------------------- loaders ----

def _load_year1_projections() -> pd.DataFrame:
    """mlbam_id, projected_war (year 1 = 2026), age -- age comes from the
    cheat sheet (projections_raw.age) via resolutions, since players.birth_year
    is never populated by ingest.py (Chadwick doesn't have it).

    GROUP BY collapses two-way players (e.g. Ohtani have both a bat and a
    pitch row in projections_raw -> two resolutions rows -> would otherwise
    join into duplicate rows here even though `projections` already kept
    only their higher-WAR row).
    """
    query = """
        SELECT p.mlbam_id, p.projected_war, MAX(pr.age) AS age
        FROM projections p
        JOIN resolutions r ON r.mlbam_id = p.mlbam_id
        JOIN projections_raw pr ON pr.id = r.projection_raw_id
        WHERE p.season = 2026 AND p.system = 'mr_cheat_sheet'
        GROUP BY p.mlbam_id, p.projected_war
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn)


def _load_year0_salaries(season: int = 2026) -> pd.DataFrame:
    """mlbam_id, season, salary_usd for the *current* season, straight from
    war_actuals.salary_usd (bwar_bat/bwar_pitch already ship it -- no extra
    ingest needed).

    bref repeats the season's salary across every stint/role row for a
    player, and (on a mid-season trade) only fills it in on one of the
    stints -- MAX() per (mlbam_id, season) picks up the one real number and
    ignores the NaNs/duplicates safely. This only ever covers the CURRENT
    season: bref reports realized salary for a season, not the future
    guaranteed years of a multi-year deal, so years 1+ of the horizon still
    have to come from _load_contracts()/manual_contracts.csv.
    """
    query = """
        SELECT mlbam_id, :season AS season, MAX(salary_usd) AS salary_usd
        FROM war_actuals
        WHERE season = :season AND salary_usd IS NOT NULL
        GROUP BY mlbam_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(query), conn, params={"season": season})


def _load_contracts(season: int = 2026) -> pd.DataFrame:
    """mlbam_id, season, salary_usd for the full horizon.

    Year 0 (the current season) is filled in automatically from
    war_actuals via _load_year0_salaries(). Years 1+ (2027, 2028, ...) --
    the future guaranteed years of a multi-year deal -- aren't in bref's
    per-season salary field, so those still come from manually curated
    data/manual_contracts.csv (columns: mlbam_id, season, salary_usd).

    If a player appears in both (e.g. you want to override the auto 2026
    number), the manual_contracts.csv row wins.

    TODO (yours): populate data/manual_contracts.csv with the 2027/2028
    salary for whichever players you're testing trades on (Spotrac / Cot's
    Baseball Contracts are the usual sources), and decide the fallback for
    players with no entry at all (skip them from valuation_all? assume
    league minimum?).
    """
    auto = _load_year0_salaries(season)

    if CONTRACTS_PATH.exists():
        manual = pd.read_csv(CONTRACTS_PATH)
    else:
        log.warning("no manual contracts file at %s -- years 1+ will have no salary", CONTRACTS_PATH)
        manual = pd.DataFrame(columns=["mlbam_id", "season", "salary_usd"])

    if manual.empty:
        return auto
    combined = pd.concat([auto, manual], ignore_index=True)
    return combined.drop_duplicates(subset=["mlbam_id", "season"], keep="last")


# ------------------------------------------------------------ aging curve ----

def _decline_rate(age: int) -> float:
    """Fraction of WAR lost for a player at this age, relative to the year
    before. 0 at/before PEAK_AGE (flat -- pre-peak growth isn't modeled,
    a stated simplification). Past peak, decline rate rises linearly with
    distance from peak, calibrated to reproduce CONTEXT_PRIMER.md's worked
    example exactly:
        age 30 (3 yrs past peak) -> 7%  decline (rate 0.93)
        age 31 (4 yrs past peak) -> 10% decline (rate 0.90)
    Solving rate = m*distance + b for those two points gives m=0.03, b=-0.02.
    Capped at 40% so very advanced ages don't blow up implausibly.
    """
    if age <= PEAK_AGE:
        return 0.0
    distance = age - PEAK_AGE
    return min(0.03 * distance - 0.02, 0.40)


def _project_war_path(war_year1: float, age: int) -> list[float]:
    """Carry war_year1 forward across HORIZON_YEARS using a simple aging curve.

    From CONTEXT_PRIMER.md: "simple polynomial (peak ~27, decline
    afterward), NOT learned from data. Honest limitation." Worked example
    given there for a 29-year-old:
        year1 (age 29) = 4.0 WAR         <- untouched, this is war_year1
        year2 (age 30) = 4.0 * 0.93 = 3.72
        year3 (age 31) = 3.72 * 0.90 = 3.35
    _decline_rate() reproduces those exact multipliers.

    Year 1 (index 0) is never aged -- it's already a single-season estimate
    from forecast.py. Only years 2+ (index 1+) get the curve applied,
    compounding off the *previous* year's (already-aged) WAR.
    """
    path = [war_year1]
    war = war_year1
    for y in range(1, HORIZON_YEARS):
        war = war * (1 - _decline_rate(age + y))
        path.append(war)
    return path


# --------------------------------------------------------------- surplus ----

def compute_surplus_value(war_path: list[float], salaries: list[float]) -> float:
    """Discounted surplus value over HORIZON_YEARS.

        surplus = sum_y [ war_path[y] * DOLLARS_PER_WAR / (1 + DISCOUNT_RATE)**y
                           - salaries[y] ]

    y=0 uses no discount (1 + DISCOUNT_RATE)**0 == 1 -- it's the current season.
    """
    return sum(
        war * DOLLARS_PER_WAR / (1 + DISCOUNT_RATE) ** y - salary
        for y, (war, salary) in enumerate(zip(war_path, salaries))
    )


# ------------------------------------------------------------------ main ----

def _salary_for(contracts: pd.DataFrame, mlbam_id: int, season: int) -> tuple[float, bool]:
    """(salary, was_estimated). Missing years fall back to LEAGUE_MINIMUM_SALARY
    and are flagged rather than silently blended into a real number."""
    match = contracts[(contracts["mlbam_id"] == mlbam_id) & (contracts["season"] == season)]
    if match.empty or pd.isna(match["salary_usd"].iloc[0]):
        return LEAGUE_MINIMUM_SALARY, True
    return float(match["salary_usd"].iloc[0]), False


def valuation_all() -> pd.DataFrame:
    projections = _load_year1_projections()
    contracts = _load_contracts()

    results = []
    for row in projections.itertuples(index=False):
        if pd.isna(row.age):
            log.warning("skipping mlbam_id=%s: no age (can't apply aging curve)", row.mlbam_id)
            continue
        age = int(row.age)

        war_path = _project_war_path(row.projected_war, age)

        salaries: list[float] = []
        estimated = False
        for y in range(HORIZON_YEARS):
            salary, was_estimated = _salary_for(contracts, row.mlbam_id, 2026 + y)
            salaries.append(salary)
            estimated = estimated or was_estimated

        surplus = compute_surplus_value(war_path, salaries)
        results.append({
            "mlbam_id": row.mlbam_id,
            "age": age,
            "war_year1": row.projected_war,
            "surplus_value": surplus,
            "salary_estimated": estimated,
        })

    return pd.DataFrame(results)


def _write(df: pd.DataFrame, as_of: date) -> None:
    if df.empty:
        log.warning("no valuations to write")
        return
    out = df.copy()
    out["as_of_date"] = as_of
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM valuations WHERE as_of_date = :d"), {"d": as_of})
    out.to_sql(Valuation.__tablename__, engine, if_exists="append", index=False)
    log.info("wrote     valuations         rows=%-7d as_of=%s", len(out), as_of)


def main() -> None:
    Base.metadata.create_all(engine)

    df = valuation_all()
    log.info("computed surplus value for %d players", len(df))

    top = df.sort_values("surplus_value", ascending=False).head(15)
    for row in top.itertuples(index=False):
        flag = " (salary estimated)" if row.salary_estimated else ""
        log.info(
            "  mlbam=%-8s age=%-3d war_y1=%5.2f surplus=$%s%s",
            row.mlbam_id, row.age, row.war_year1, f"{row.surplus_value:,.0f}", flag,
        )

    _write(df, date.today())


if __name__ == "__main__":
    main()
