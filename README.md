# Baseball Trade Recommendation Platform

A data pipeline that estimates how much surplus value an MLB player represents as a
trade target — combining historical performance, a fielding-aware WAR projection
model, and salary data into a single dollar figure teams could use to evaluate
whether a player is worth acquiring.

Surplus value answers a specific question: *not* "who's the best player," but
"who's worth more than they're being paid?" A superstar already earning superstar
money is a bad trade target even though they're a great player; a young, cheap,
high-performing player is where the actual trade value lives.

## How it works

Four stages, each a standalone script, each reading from and writing to a shared
Postgres database:

```mermaid
flowchart LR
    A["Ingest"] --> B["Resolve"]
    B --> C["Forecast"]
    C --> D["Valuation"]
    A -.-> DB[(Postgres)]
    B -.-> DB
    C -.-> DB
    D -.-> DB
```

| Stage | Script | What it does |
|---|---|---|
| **Ingest** | `src/ingest.py` | Pulls the player ID registry, 12 years of historical stats + actual WAR + salary, and this year's projection CSVs |
| **Resolve** | `src/resolve.py` | Matches each projection's player name to a canonical MLB player ID (handles nicknames, misspellings, and same-name collisions) |
| **Forecast** | `src/forecast.py` | Trains a regression model on historical stats → realized WAR, then predicts this season's WAR for every projected player |
| **Valuation** | `src/valuation.py` | Projects WAR forward across a multi-year horizon with an aging curve, converts it to dollars, and subtracts salary to get surplus value |

## Data sources

- **[Chadwick Bureau Register](https://github.com/chadwickbureau/register)** (via `pybaseball`) — the canonical crosswalk of player IDs across data sources
- **Baseball-Reference bWAR** (via `pybaseball`) — historical actual WAR, salary, and fielding value, batting and pitching
- **Baseball-Reference season stats** (via `pybaseball`) — 12 seasons (2014–present) of box-score stats used to train the WAR model
- **Mr. Cheat Sheet** — a fantasy-baseball projection CSV for the upcoming season (`data/raw/*.csv`), the input the model turns into a WAR estimate

## Design decisions

These are deliberate, explainable choices rather than arbitrary constants:

- **$8M / WAR** (`DOLLARS_PER_WAR`) — the going market rate for a win, sourced from
  free-agent market analysis (Fangraphs / MLB Trade Rumors). Literature range is
  roughly $8M–$10M.
- **8% discount rate** (`DISCOUNT_RATE`) — future WAR is discounted for projection
  uncertainty and time value.
- **3-year valuation horizon** — projection accuracy degrades meaningfully past
  ~3 years, so surplus value is only computed that far out.
- **Aging curve** — a simple, explainable formula (flat through peak at age 27, then
  linearly increasing decline) rather than a learned model. This is a stated
  simplification, not a claim of precision — it's transparent about what it is.
- **Point-in-time data** — every ingested fact carries an `as_of_date`, so historical
  snapshots are preserved rather than overwritten. This is what makes it possible to
  eventually backtest a recommendation against what was actually knowable at the time,
  without leaking future information into a past decision.

## Model evaluation

The honest question for any prediction model is "how do you know it's any good?" —
not "do the top names look like real stars," which is weak evidence at best. Two
scripts answer this with real held-out testing against actual historical outcomes:

- **`src/evaluate.py`** — trains on 2014–2023, tests on 2024–2025, scoring
  predicted vs. actual WAR for players in those held-out seasons. Also runs two
  ablations: does the fielding feature actually help, and does swapping in more
  skill-driven pitcher stats (strikeout/walk/home-run rates instead of ERA/WHIP/W-L)
  help.
- **`src/evaluate_forecast.py`** — the harder, more honest version. The test above
  reconstructs a season's WAR from that *same* season's real stats, which is close
  to solving a known equation (WAR is partly defined by those stats) rather than
  forecasting. This script instead trains on season *N*'s stats predicting season
  *N+1*'s WAR — the same shape of problem the 2026 projections actually pose — and
  is the number that should be trusted as the real accuracy ceiling.

Both compare the model against two naive baselines (predict the league average;
predict the player repeats last season) — a model that can't beat those isn't
adding value.

| | Same-season reconstruction | **True forecast (N → N+1)** |
|---|---|---|
| Batting R² | 0.82 | **0.36** |
| Pitching R² | 0.56 | **0.16** |

**The true-forecast numbers are the ones that matter** for what this pipeline
actually does. Both still clearly beat the naive baselines (batting: naive best is
R²=0.22; pitching: naive best is R²=-0.21, since pitcher performance is genuinely
volatile year to year even before a model is involved) — so the model is adding
real signal, just less than the same-season number would suggest on its own.

**Ablation findings:**
- The fielding feature (`prior_def_runs`) holds up under the harder test too —
  batting R² 0.30 → 0.36 with it included. Not an artifact of the easier evaluation.
- Swapping ERA/WHIP/W-L for skill-driven pitcher rate stats is a wash in the true
  forecast (R² 0.164 vs 0.162) despite losing clearly in same-season reconstruction
  (0.56 vs 0.47) — the current feature set stays as-is; there's no evidence a swap
  would help.
- **Model comparison** (Ridge vs. ElasticNet vs. RandomForest vs. XGBoost vs. a
  Ridge+XGBoost blend, same features, same split): for batting, Ridge ties or beats
  every alternative (R²=0.358) — no evidence a fancier model helps. For pitching,
  two independent tree-based models (RandomForest and XGBoost) both landed on the
  same improvement over Ridge (R²=0.164 → 0.219), which is stronger evidence than
  either alone that pitching has real non-linear structure a straight-line model
  misses. **Not yet applied to `forecast.py`** — the evidence supports switching the
  pitch model to a tree-based one while keeping Ridge for batting, but that change is
  still pending.

Run either with `python -m src.evaluate` / `python -m src.evaluate_forecast`. Full
narrative writeup, including why the numbers dropping between the two tests is
expected (not a bug) and how to defend a lower-than-professional-systems number in an
interview: see **`EVALUATION_WALKTHROUGH.md`**. For the data pipeline itself
(ingest → resolve → forecast → valuation), see **`PIPELINE_WALKTHROUGH.md`**.

## Getting started

**Requirements:** Docker Desktop, Python 3.12+

```bash
# 1. start Postgres
cp .env.example .env
docker compose up -d

# 2. set up the environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. run the pipeline, in order
python -m src.ingest
python -m src.resolve
python -m src.forecast
python -m src.valuation
```

Each stage logs a summary as it runs — row counts, match rates, and (for valuation)
the top players by surplus value.

## Manually curated data

Two files fill gaps that no public API covers:

- **`data/manual_overrides.csv`** — player-name matches the automated resolver can't
  make confidently on its own (nickname mismatches, two players sharing a name).
  Rebuilds automatically the moment a name is added here — just rerun
  `python -m src.resolve`.
- **`data/manual_contracts.csv`** — future-year salary (beyond the current season)
  for specific players, since no public source provides forward-looking guaranteed
  contract money in a queryable format. Current-season salary is pulled automatically;
  this file only needs entries for players you're actively evaluating trades for.

## Project structure

```
src/
  config.py     # env vars, paths, the $/WAR and discount-rate constants
  db.py         # SQLAlchemy engine
  models.py     # ORM table definitions
  ingest.py     # stage 1
  resolve.py    # stage 2
  forecast.py   # stage 3
  valuation.py  # stage 4
  evaluate.py            # same-season backtest + ablations
  evaluate_forecast.py   # true N -> N+1 forecast backtest (the honest number)
  main.py       # API layer (in progress, see below)
data/
  raw/          # source CSVs (season projections)
  snapshots/    # parquet snapshots of every ingest run, by date
  manual_overrides.csv
  manual_contracts.csv
```

## Currently in progress

- **API layer** (`src/main.py`) — a FastAPI service to expose valuations and trade
  recommendations over HTTP; not yet built.
- **Trade recommendation logic** — matching valuations across two rosters to surface
  candidate 1-for-1 trades where both sides gain surplus value; not yet built.
- **Backtest harness** — replaying a past season using only point-in-time data to
  validate that recommendations would have held up, using the `as_of_date` fields
  already in place for this purpose.
- **`data/manual_contracts.csv`** — currently empty; needs real 2027/2028 salary
  figures for whichever players get evaluated in actual trade scenarios.
- A small number of just-drafted prospects are tracked in `data/manual_overrides.csv`
  with no player ID yet, since they haven't appeared in the public registry — these
  resolve automatically once they debut and gain one.

## Known limitations

- The aging curve is an explainable heuristic, not learned from data.
- Multi-player trades (2-for-1 and larger) are out of scope — the search space grows
  combinatorially; this focuses on cleanly demonstrating 1-for-1 valuation instead.
- No auth, frontend, or cloud deployment — this is a local, containerized demo
  (Docker + Postgres + the pipeline scripts).
