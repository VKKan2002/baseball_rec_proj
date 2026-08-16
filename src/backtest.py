"""Backtest: run the recommendation pipeline as of a past decision date.

The honest claim this project can make about point-in-time correctness is
narrower than "we replayed a past season and it worked" -- there is currently
only one populated `as_of_date` in `valuations` (this pipeline has been run
once, today, against the 2026 cheat sheet). A true multi-season replay would
need historical *projection* snapshots -- what a system like Steamer said
about a player before each past season started -- which this project never
collected; `season_stats` / `war_actuals` only hold realized, after-the-fact
performance, not point-in-time projections.

What CAN be proven today, and what this module + its regression test
(`tests/test_backtest_pit.py`) actually prove: the read path that decides
"what did we know as of date X" never leaks a row written after X. That's
the specific failure mode a real backtest has to rule out -- a query that
silently reads `MAX(as_of_date)` with no cutoff would happily hand a 2024
backtest a valuation computed from 2026 data, and the mistake would be
invisible in the output (the numbers just look normal, only more accurate
than they should be).

`_load_valuations()` in recommend.py is written so `as_of_date=None` (the
CLI default) is the only case where it behaves like a bare "latest" read;
every other cutoff is `MAX(as_of_date) WHERE as_of_date <= :cutoff`, so
"replay" here means "run the exact same recommend.py query path, pinned to
a cutoff, and prove nothing later than that cutoff can enter the answer."
As this project accumulates more `valuations` snapshots over time (one per
day it's actually run), `--as-of` starts replaying real history instead of
just re-deriving today's answer with extra steps.

Run:
    python -m src.backtest --as-of 2026-08-01
    python -m src.backtest --as-of 2026-08-01 --team NYM
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recommend import main as recommend_main  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("backtest")


def run(as_of: date, team: str | None = None, top_n: int = 15) -> None:
    log.info("backtest as_of=%s -- replaying recommend.py pinned to this cutoff", as_of)
    recommend_main(team=team, top_n=top_n, as_of_date=as_of)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of", required=True, type=date.fromisoformat,
        help="decision date (YYYY-MM-DD); only valuations with as_of_date <= this are visible",
    )
    parser.add_argument("--team", default=None, help="restrict to trades involving this team code (e.g. NYM)")
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()
    run(as_of=args.as_of, team=args.team, top_n=args.top_n)
