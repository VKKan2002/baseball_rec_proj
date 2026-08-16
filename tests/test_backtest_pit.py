"""Proves the point-in-time claim in src/recommend.py._load_valuations(): a
cutoff can only ever see valuations rows written on or before it, never a
row dated later -- regardless of when the test happens to run or what real
data is already sitting in the table.

Method: insert two valuations rows for one throwaway player, one dated in
the past (2020-01-01) and one dated far in the future (2099-01-01) relative
to today. Query with a cutoff sitting between them and assert we get the
past row's numbers, not the future one's -- that's the exact failure mode a
bare `MAX(as_of_date)` (no cutoff) would be blind to: it would happily hand
a historical backtest data that didn't exist yet at the decision date being
replayed.

Requires a live Postgres (docker compose up -d) -- this hits the real
`valuations` / `players` / `resolutions` / `projections_raw` tables, same as
the pipeline itself, rather than mocking the query.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.db import engine  # noqa: E402
from src.models import Base  # noqa: E402
from src.recommend import _load_valuations, find_fair_trades  # noqa: E402

FAKE_MLBAM_ID = -1
FAKE_SOURCE = "__test_backtest_pit__"

PAST = date(2020, 1, 1)
FUTURE = date(2099, 1, 1)
CUTOFF_BETWEEN = date(2020, 6, 1)   # sees PAST, not FUTURE
CUTOFF_AFTER_BOTH = date(2099, 6, 1)

PAST_SURPLUS = 100.0
FUTURE_SURPLUS = 999_999.0


def _cleanup(conn) -> None:
    conn.execute(text("DELETE FROM valuations WHERE mlbam_id = :id"), {"id": FAKE_MLBAM_ID})
    conn.execute(text("DELETE FROM resolutions WHERE mlbam_id = :id"), {"id": FAKE_MLBAM_ID})
    conn.execute(text("DELETE FROM projections_raw WHERE source = :s"), {"s": FAKE_SOURCE})
    conn.execute(text("DELETE FROM players WHERE mlbam_id = :id"), {"id": FAKE_MLBAM_ID})


@pytest.fixture
def fake_player_with_two_snapshots():
    """A throwaway player with valuations rows at PAST and FUTURE as_of_dates,
    plus the players/resolutions/projections_raw rows _load_valuations()
    joins against. Cleaned up before AND after, so a previous failed run
    can't leave stale rows behind either.
    """
    Base.metadata.create_all(engine)

    with engine.begin() as conn:
        _cleanup(conn)

        conn.execute(
            text(
                "INSERT INTO players (mlbam_id, first_name, last_name) "
                "VALUES (:id, 'Test', 'Player')"
            ),
            {"id": FAKE_MLBAM_ID},
        )
        proj_raw_id = conn.execute(
            text(
                "INSERT INTO projections_raw "
                "(source, as_of_date, season, role, player_name, team, primary_pos) "
                "VALUES (:source, :as_of, 2026, 'bat', 'Test Player', 'ZZZ', '1B') "
                "RETURNING id"
            ),
            {"source": FAKE_SOURCE, "as_of": PAST},
        ).scalar_one()
        conn.execute(
            text(
                "INSERT INTO resolutions "
                "(projection_raw_id, mlbam_id, method, score, candidate_count, resolved_at) "
                "VALUES (:praw_id, :id, 'exact', 100, 1, :today)"
            ),
            {"praw_id": proj_raw_id, "id": FAKE_MLBAM_ID, "today": date.today()},
        )
        for as_of, surplus in [(PAST, PAST_SURPLUS), (FUTURE, FUTURE_SURPLUS)]:
            conn.execute(
                text(
                    "INSERT INTO valuations "
                    "(mlbam_id, as_of_date, age, war_year1, surplus_value, salary_estimated) "
                    "VALUES (:id, :as_of, 30, 1.0, :surplus, false)"
                ),
                {"id": FAKE_MLBAM_ID, "as_of": as_of, "surplus": surplus},
            )

    yield

    with engine.begin() as conn:
        _cleanup(conn)


def _surplus_for_fake_player(as_of_date: date | None) -> float:
    df = _load_valuations(as_of_date=as_of_date)
    row = df[df["mlbam_id"] == FAKE_MLBAM_ID]
    assert len(row) == 1, f"expected exactly one row for the fake player, got {len(row)}"
    return float(row["surplus_value"].iloc[0])


def test_cutoff_between_snapshots_sees_only_the_past_row(fake_player_with_two_snapshots):
    assert _surplus_for_fake_player(CUTOFF_BETWEEN) == PAST_SURPLUS


def test_cutoff_after_both_snapshots_sees_the_later_row(fake_player_with_two_snapshots):
    assert _surplus_for_fake_player(CUTOFF_AFTER_BOTH) == FUTURE_SURPLUS


def test_cutoff_before_any_snapshot_returns_no_row(fake_player_with_two_snapshots):
    df = _load_valuations(as_of_date=date(2019, 1, 1))
    assert df[df["mlbam_id"] == FAKE_MLBAM_ID].empty


def test_find_fair_trades_handles_zero_players_without_crashing():
    """A backtest asking about a decision date before any data existed gets
    a legitimate zero-player valuations frame from _load_valuations() --
    find_fair_trades() must return an empty result, not raise. (Regression:
    filtering with a bare Python list of 0 booleans, pairs[[]], is
    ambiguous to pandas and was silently read as "select zero columns"
    instead of "keep zero rows", producing a column-less frame downstream.)
    """
    empty = _load_valuations(as_of_date=date(2019, 1, 1))
    assert empty.empty

    result = find_fair_trades(empty, top_n=5)
    assert result.empty
    assert list(result.columns) == [
        "mlbam_id_a", "team_a", "name_a", "primary_pos_a", "surplus_value_a",
        "mlbam_id_b", "team_b", "name_b", "primary_pos_b", "surplus_value_b",
        "surplus_gap",
    ]
