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

Roster fit: a pair is only a plausible swap if the two players' eligible
positions overlap at all -- a reliever and a catcher share no position, so
they're filtered out even if their surplus values happen to land close
together. Eligibility is primary_pos plus every comma-separated entry in
other_pos (projections_raw), which already captures multi-position and
two-way players (e.g. Ohtani: DH + SP). This is still a coarse heuristic --
it says "could plausibly occupy the same roster spot," not "a real GM would
make this specific trade" -- but it removes the obviously-wrong matches a
value-only ranking produces.

Run:
    python -m src.recommend
    python -m src.recommend --team NYM
    python -m src.recommend --any-position   # disable the roster-fit filter
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
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

def _load_valuations(as_of_date: date | None = None, mlbam_id: int | None = None) -> pd.DataFrame:
    """One row per player: name, team, position, surplus value -- as of the
    most recent valuations run available BY `as_of_date` (default: the true
    latest, i.e. today's view). Pass `mlbam_id` to fetch a single player
    instead of the whole league (used by the API's single-player endpoint).

    This is the point-in-time read path: it never does a bare
    `MAX(as_of_date)` -- the subquery is always `MAX(as_of_date) WHERE
    as_of_date <= :cutoff`, so passing a past cutoff can only ever see rows
    that existed as of that date, never anything written later. See
    `src/backtest.py` and `tests/test_backtest_pit.py`, which exercise this
    exact guarantee.

    DISTINCT ON collapses two-way players (e.g. Ohtani has both a bat and a
    pitch row in projections_raw -> two resolutions rows) down to one, same
    reasoning as valuation.py's own loader.
    """
    query = """
        SELECT DISTINCT ON (v.mlbam_id)
            v.mlbam_id,
            v.as_of_date,
            pl.first_name,
            pl.last_name,
            pr.team,
            pr.primary_pos,
            pr.other_pos,
            v.age,
            v.war_year1,
            v.surplus_value,
            v.salary_estimated
        FROM valuations v
        JOIN players pl ON pl.mlbam_id = v.mlbam_id
        JOIN resolutions res ON res.mlbam_id = v.mlbam_id
        JOIN projections_raw pr ON pr.id = res.projection_raw_id
        WHERE v.as_of_date = (
            SELECT MAX(as_of_date) FROM valuations
            WHERE as_of_date <= :cutoff
        )
        {mlbam_filter}
        ORDER BY v.mlbam_id, pr.role
    """.format(mlbam_filter="AND v.mlbam_id = :mlbam_id" if mlbam_id is not None else "")
    cutoff = as_of_date if as_of_date is not None else date.max
    params: dict = {"cutoff": cutoff}
    if mlbam_id is not None:
        params["mlbam_id"] = mlbam_id
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    df["name"] = df["first_name"] + " " + df["last_name"]
    return df.drop(columns=["first_name", "last_name"])


# ------------------------------------------------------------- recommend ----

def _eligible_positions(primary_pos: str | None, other_pos: str | None) -> frozenset[str]:
    """Every position a player could plausibly fill: primary_pos plus each
    comma-separated entry in other_pos (e.g. "3B,OF,1B" -> {3B, OF, 1B}).
    Two-way players (Ohtani: primary DH, other SP) end up eligible at both.
    """
    positions = set()
    if primary_pos:
        positions.add(primary_pos)
    if other_pos:
        positions.update(p for p in other_pos.split(",") if p)
    return frozenset(positions)


def find_fair_trades(
    valuations: pd.DataFrame,
    team: str | None = None,
    top_n: int = 20,
    require_position_fit: bool = True,
) -> pd.DataFrame:
    """Every distinct-team player pair, ranked by how close their surplus
    values are (smallest gap first) -- the most realistic, roughly-even
    1-for-1 swaps. If `team` is given, only pairs involving that team are
    returned, and that team's player is always the "_a" side for readability.

    If `require_position_fit` (default), pairs whose eligible positions don't
    overlap at all (see `_eligible_positions`) are dropped -- otherwise the
    ranking is surplus-value-only and can pair, say, a reliever with a
    catcher purely because the dollar figures happen to be close.
    """
    valuations = valuations.copy()
    valuations["eligible_pos"] = [
        _eligible_positions(pp, op)
        for pp, op in zip(valuations["primary_pos"], valuations["other_pos"])
    ]

    pairs = valuations.merge(valuations, how="cross", suffixes=("_a", "_b"))
    pairs = pairs[
        (pairs["team_a"] != pairs["team_b"])
        & (pairs["mlbam_id_a"] < pairs["mlbam_id_b"])
    ].copy()

    if require_position_fit:
        # Built as a pd.Series (not a bare list) so this stays unambiguous
        # boolean row-filtering even when `pairs` is empty -- pandas treats
        # `df[[]]` (an empty *list*) as "select zero columns", silently
        # collapsing the whole frame instead of just filtering zero rows.
        fits = pd.Series(
            [bool(pos_a & pos_b) for pos_a, pos_b in zip(pairs["eligible_pos_a"], pairs["eligible_pos_b"])],
            index=pairs.index,
            dtype=bool,
        )
        pairs = pairs[fits].copy()

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
        "mlbam_id_a", "team_a", "name_a", "primary_pos_a", "surplus_value_a",
        "mlbam_id_b", "team_b", "name_b", "primary_pos_b", "surplus_value_b",
        "surplus_gap",
    ]
    return pairs[keep].sort_values("surplus_gap").head(top_n).reset_index(drop=True)


# ------------------------------------------------------------------ main ----

def main(
    team: str | None = None,
    top_n: int = 15,
    require_position_fit: bool = True,
    as_of_date: date | None = None,
) -> None:
    valuations = _load_valuations(as_of_date=as_of_date)
    log.info("loaded %d players across %d teams", len(valuations), valuations["team"].nunique())

    trades = find_fair_trades(valuations, team=team, top_n=top_n, require_position_fit=require_position_fit)
    label = f"fairest trade candidates involving {team}" if team else "fairest trade candidates, league-wide"
    if require_position_fit:
        label += " (roster-fit filtered)"
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
    parser.add_argument(
        "--any-position", action="store_true",
        help="disable roster-fit filtering; rank by surplus-value closeness only",
    )
    args = parser.parse_args()
    main(team=args.team, top_n=args.top_n, require_position_fit=not args.any_position)
