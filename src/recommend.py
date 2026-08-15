"""Recommend: find plausible 1-for-1 trade candidates from `valuations`.

A player's surplus value (valuation.py) is a UNIVERSAL number -- it doesn't
depend on which team currently holds the player. That has an easy-to-miss
consequence: under one shared value, a 1-for-1 trade can never make BOTH
sides better off except by exact coincidence (surplus_a == surplus_b). Any
real gap between two players' surplus value is zero-sum by construction --
whatever Team A gains by receiving a $60M-surplus player and giving up a
$40M-surplus player, Team B loses exactly that. That's not a modeling bug,
it's what "one universal value number" necessarily implies.

So this doesn't hunt for a "both sides gain" trade -- the surplus value
model, as built (no positional need, no contention-window modeling), can't
honestly produce one. Instead it finds the closest achievable thing: FAIR
trades -- 1-for-1 swaps between two different teams whose players have
close surplus values. That's still a genuinely useful answer to "who's a
realistic trade partner for player X," and it's honest about the ceiling of
what a value-only model can tell you.

Team affiliation comes from projections_raw.team (the 2026 cheat sheet) --
the `rosters` table is unpopulated, but the cheat sheet already carries each
player's current team, which is what we actually need here.

Run:
    python -m src.recommend
    python -m src.recommend --team NYM
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.db import engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("recommend")


# ---------------------------------------------------------------- loaders ----

def _load_valuations() -> pd.DataFrame:
    """One row per player (latest as_of_date): name, team, position, surplus
    value. DISTINCT ON collapses two-way players (e.g. Ohtani has both a bat
    and a pitch row in projections_raw -> two resolutions rows) down to one,
    same reasoning as valuation.py's own loader.
    """
    query = """
        SELECT DISTINCT ON (v.mlbam_id)
            v.mlbam_id,
            pl.first_name,
            pl.last_name,
            pr.team,
            pr.primary_pos,
            v.age,
            v.war_year1,
            v.surplus_value,
            v.salary_estimated
        FROM valuations v
        JOIN players pl ON pl.mlbam_id = v.mlbam_id
        JOIN resolutions res ON res.mlbam_id = v.mlbam_id
        JOIN projections_raw pr ON pr.id = res.projection_raw_id
        WHERE v.as_of_date = (SELECT MAX(as_of_date) FROM valuations)
        ORDER BY v.mlbam_id, pr.role
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    df["name"] = df["first_name"] + " " + df["last_name"]
    return df.drop(columns=["first_name", "last_name"])


# ------------------------------------------------------------- recommend ----

def find_fair_trades(
    valuations: pd.DataFrame,
    team: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """Every distinct-team player pair, ranked by how close their surplus
    values are (smallest gap first) -- the most realistic, roughly-even
    1-for-1 swaps. If `team` is given, only pairs involving that team are
    returned, and that team's player is always the "_a" side for readability.
    """
    pairs = valuations.merge(valuations, how="cross", suffixes=("_a", "_b"))
    pairs = pairs[
        (pairs["team_a"] != pairs["team_b"])
        & (pairs["mlbam_id_a"] < pairs["mlbam_id_b"])
    ].copy()

    if team is not None:
        involved = (pairs["team_a"] == team) | (pairs["team_b"] == team)
        pairs = pairs[involved].copy()
        # orient so the queried team is always on the "_a" side
        needs_swap = pairs["team_b"] == team
        a_cols = [c for c in pairs.columns if c.endswith("_a")]
        b_cols = [c for c in pairs.columns if c.endswith("_b")]
        swapped = pairs.loc[needs_swap, b_cols + a_cols]
        swapped.columns = a_cols + b_cols
        pairs = pd.concat([pairs[~needs_swap], swapped], ignore_index=True)

    pairs["surplus_gap"] = (pairs["surplus_value_a"] - pairs["surplus_value_b"]).abs()

    keep = [
        "team_a", "name_a", "primary_pos_a", "surplus_value_a",
        "team_b", "name_b", "primary_pos_b", "surplus_value_b",
        "surplus_gap",
    ]
    return pairs[keep].sort_values("surplus_gap").head(top_n).reset_index(drop=True)


# ------------------------------------------------------------------ main ----

def main(team: str | None = None, top_n: int = 15) -> None:
    valuations = _load_valuations()
    log.info("loaded %d players across %d teams", len(valuations), valuations["team"].nunique())

    trades = find_fair_trades(valuations, team=team, top_n=top_n)
    label = f"fairest trade candidates involving {team}" if team else "fairest trade candidates, league-wide"
    log.info("%s:", label)
    for row in trades.itertuples(index=False):
        log.info(
            "  %-4s %-22s (%-3s $%s)  <->  %-4s %-22s (%-3s $%s)  gap=$%s",
            row.team_a, row.name_a, row.primary_pos_a or "?", f"{row.surplus_value_a:,.0f}",
            row.team_b, row.name_b, row.primary_pos_b or "?", f"{row.surplus_value_b:,.0f}",
            f"{row.surplus_gap:,.0f}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=None, help="restrict to trades involving this team code (e.g. NYM)")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()
    main(team=args.team, top_n=args.top_n)
