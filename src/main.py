"""API: expose the pipeline's outputs (valuation, recommend, backtest) over HTTP.

This is a thin read layer -- every endpoint here calls straight into the same
functions the CLI scripts use (`_load_valuations`, `find_fair_trades`), so
there's exactly one implementation of "what's a player's surplus value" and
"what's a fair trade," not a duplicate copy re-derived for the API. No auth,
no writes, no frontend -- this is a local demo: `docker compose up -d` plus
this service's `/docs` page is enough to say "containerized and reproducible"
(see README's "explicitly cut from scope").

Run:
    uvicorn src.main:app --reload
    # then open http://localhost:8000/docs
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.recommend import _load_valuations, find_fair_trades  # noqa: E402

app = FastAPI(
    title="Baseball Trade Recommendation API",
    description="Surplus-value player valuations and fair 1-for-1 trade candidates.",
    version="1.0.0",
)


# ---------------------------------------------------------------- models ----

class ValuationOut(BaseModel):
    mlbam_id: int
    name: str
    team: str | None
    position: str | None
    age: int
    war_year1: float
    surplus_value: float
    salary_estimated: bool
    as_of_date: date


class TradeCandidateOut(BaseModel):
    team_a: str | None
    name_a: str
    position_a: str | None
    surplus_value_a: float
    team_b: str | None
    name_b: str
    position_b: str | None
    surplus_value_b: float
    surplus_gap: float


def _trade_rows_to_models(df) -> list[TradeCandidateOut]:
    return [
        TradeCandidateOut(
            team_a=row.team_a, name_a=row.name_a, position_a=row.primary_pos_a,
            surplus_value_a=row.surplus_value_a,
            team_b=row.team_b, name_b=row.name_b, position_b=row.primary_pos_b,
            surplus_value_b=row.surplus_value_b,
            surplus_gap=row.surplus_gap,
        )
        for row in df.itertuples(index=False)
    ]


# -------------------------------------------------------------- endpoints ----

@app.get("/")
def root() -> dict:
    return {"status": "ok", "docs": "/docs"}


@app.get("/players/{mlbam_id}/valuation", response_model=ValuationOut)
def get_player_valuation(mlbam_id: int) -> ValuationOut:
    """A single player's latest surplus value -- same numbers valuation.py
    computed and recommend.py ranks by, just scoped to one player."""
    df = _load_valuations(mlbam_id=mlbam_id)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"no valuation found for mlbam_id={mlbam_id}")
    row = df.iloc[0]
    return ValuationOut(
        mlbam_id=int(row.mlbam_id), name=row["name"], team=row.team, position=row.primary_pos,
        age=int(row.age), war_year1=float(row.war_year1), surplus_value=float(row.surplus_value),
        salary_estimated=bool(row.salary_estimated), as_of_date=row.as_of_date,
    )


@app.get("/recommend", response_model=list[TradeCandidateOut])
def get_recommendations(
    team: str | None = Query(default=None, description="restrict to trades involving this team code, e.g. NYM"),
    top_n: int = Query(default=15, ge=1, le=100),
    any_position: bool = Query(default=False, description="disable roster-fit filtering"),
) -> list[TradeCandidateOut]:
    """Fairest 1-for-1 trade candidates today -- same as `python -m src.recommend`."""
    valuations = _load_valuations()
    trades = find_fair_trades(valuations, team=team, top_n=top_n, require_position_fit=not any_position)
    return _trade_rows_to_models(trades)


@app.get("/backtest", response_model=list[TradeCandidateOut])
def get_backtest(
    as_of: date = Query(..., description="decision date (YYYY-MM-DD); only valuations as_of_date <= this are visible"),
    team: str | None = Query(default=None, description="restrict to trades involving this team code, e.g. NYM"),
    top_n: int = Query(default=15, ge=1, le=100),
    any_position: bool = Query(default=False, description="disable roster-fit filtering"),
) -> list[TradeCandidateOut]:
    """Fairest 1-for-1 trade candidates as of a past decision date -- same
    point-in-time guarantee as `python -m src.backtest --as-of`, see README's
    'Backtesting and the point-in-time guarantee' section."""
    valuations = _load_valuations(as_of_date=as_of)
    trades = find_fair_trades(valuations, team=team, top_n=top_n, require_position_fit=not any_position)
    return _trade_rows_to_models(trades)
