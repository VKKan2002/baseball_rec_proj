# Context for a new AI assistant — Baseball Trade Rec Project

## Who I am

Venkata Krishna Kandalai. MS Health Informatics (Hood College, Dec 2025, 3.98 GPA). BS Biology/Bioinformatics (George Mason, Dec 2023). Currently AI/ML Intern at MedStar–Georgetown AI CoLab (May 2025 – Jul 2026, part-time) and Solutions Architect Intern at Vivid Solutions (Apr 2026 – Present, part-time). Based in Herndon, VA.

## The interview situation

- IBM AI Engineer role interview, expected next week (Mon–Fri window, not yet scheduled).
- **Real resume** goes to IBM (not the mentor-written "AI Engineer" version). Real resume leads with FHIR Cracker (capstone: Llama-3-8B + LoRA, 96.5% mapping accuracy, 0% hallucination, has a paper), Vivid Solutions (FastAPI + PostgreSQL + AWS S3/Glue/Athena on synthetic Medicare claims), and MedStar (PubMedBERT chemical risk scoring, MiniLM + KNN caregiver burnout prediction).
- **Baseball is project #3.** It's being built for real this weekend because mentor Kar (Karthik Arunapuram) had it on a fabricated resume and prepped an interview sheet around it. Since it interests me, I'm making it real so it becomes a defensible answer instead of a landmine.

## Working principles I've committed to

- Use AI to accelerate implementation. Don't pretend I hand-typed every line.
- Design decisions and math must be mine — I have to defend them from first principles.
- Never claim ownership of Kar's fantasy baseball repo (`C:\Users\venka\Documents\GitHub\baseball-algorithm` — that's his, not mine).
- Never bluff a definition. "I don't know but here's how I'd approach it" is a strong interview answer.
- Fix any misleading resume wording: "synthetic Medicare claims data" not "Medicare claims data."

## Project scope

Baseball trade recommendation platform. Seven boxes:

```
Public sources → Ingest → Entity Resolution → Canonical Store (Postgres)
                                                     ↓
                                                Valuation
                                                     ↓
                                          Constraint Search (Recommender)
                                                     ↓
                                                 FastAPI
                                                     ↑
                                Backtest harness ────┘  (reads canonical
                                                         store, PIT)
```

**Data sources and libraries:**
- **Steamer projections** — via `pybaseball.fangraphs_projections`. Fangraphs-hosted model built by Jared Cross et al.; weighted historical stats + regression to mean + aging adjustments; updates through the season.
- **Chadwick Bureau Register** — via `pybaseball.playerid_lookup`. Deterministic ID crosswalk for `key_mlbam`, `key_fangraphs`, `key_bbref`, `key_retro`. Solves ~95% of entity resolution without fuzzy matching.
- **Lahman salaries** — via `pybaseball.lahman.salaries`. Historical only; active contracts require manual curation.
- **Actual stats** — `pybaseball.batting_stats(year)` / `pitching_stats(year)` for ex-post backtest scoring.
- **Postgres 16** in Docker Compose; **SQLAlchemy** ORM; **FastAPI + uvicorn**; **rapidfuzz** as entity-resolution fallback; **XGBoost + scikit-learn** cut from scope (see below).

## Explicitly cut from scope (know why for interview)

- **XGBoost projection model** — cut. Interview answer: *"Building my own projection model would need multiple seasons of validated backtesting to trust it over Steamer, which is peer-reviewed and battle-tested. I consume Steamer and treat my ability to plug in a custom model as a future extension."*
- **2-for-1 (and larger) trade enumeration** — cut. Interview answer: *"Search space blows up quickly. I demonstrated the constraint-search pattern cleanly on 1-for-1s; extending to N-for-M is search-space engineering, not a modeling question."*
- **Auth, frontend, cloud deploy** — cut. *"It's a demo. Docker + the FastAPI docs page is enough to say 'containerized and reproducible.'"*

## Non-negotiable design decisions I must own

- **$8M/WAR** (`DOLLARS_PER_WAR`) — sourced from Fangraphs / MLB Trade Rumors free-agent recap articles. Range in literature: $8M–$10M. TODO: bookmark a specific URL to cite.
- **8% discount rate** (`DISCOUNT_RATE`) — future WAR discounted for uncertainty + time value. If asked, note sensitivity: 0.05–0.12 doesn't change rankings meaningfully.
- **3-year horizon** for surplus valuation — projection accuracy degrades past ~3 years.
- **Aging curve** is a simple polynomial (peak ~27, decline afterward), NOT learned from data. This is an honest limitation — put in README.

## What is built as of end-of-Saturday 2026-07-25

**Environment (all verified working):**
- Python 3.11.9 venv at `.venv/`
- All 14 dependencies installed (pandas, pybaseball, sqlalchemy, psycopg[binary], fastapi, xgboost, scikit-learn, rapidfuzz, pyarrow, pytest, etc.)
- Docker Desktop running Postgres 16 container `baseball_rec_proj-postgres-1`
- **Postgres exposed on host port `15432`, NOT 5432** — user has native Windows postgresql-x64-17 (on 5432) and postgresql-x64-18 (on 5433). Moved compose to 15432 to avoid collision. DBeaver connections must use 15432.
- `.env`: `DATABASE_URL=postgresql+psycopg://baseball:baseball@localhost:15432/baseball`, `DOLLARS_PER_WAR=8000000`, `DISCOUNT_RATE=0.08`
- SQLAlchemy verified end-to-end: `SELECT version()` returns `PostgreSQL 16.14`

**Files that exist and are correct:**
- `requirements.txt` — pinned
- `docker-compose.yml` — Postgres 16, port 15432:5432, named volume `pgdata`
- `.env`, `.env.example`, `.gitignore` — correct
- `src/config.py` — loads env vars, creates data dirs on import, fail-fast on missing `DATABASE_URL`
- `src/db.py` — SQLAlchemy engine + `SessionLocal` factory
- `src/models.py` — five ORM tables live in Postgres:
  - `players` PK `mlbam_id`
  - **`projections` PK `(mlbam_id, season, system, as_of_date)`** — the composite PK is what enables the point-in-time backtest
  - `contracts` PK `(mlbam_id, season)` — no `as_of_date` because contracts don't get revised
  - `rosters` PK `(mlbam_id, as_of_date)` — `as_of_date` (not `season`) so mid-season trades can be represented
  - `team_seasons` PK `(team, season)` — cap and positional needs
- Empty stubs: `src/ingest.py`, `src/resolve.py`, `src/valuation.py`, `src/main.py` — to be written Sunday

**Interview-defensible concepts I've drilled to a passing standard:**
- **WAR** — wins above replacement, per-season, combines hitting/fielding/baserunning/pitching into one number, counting stat (accumulates with playing time)
- **Replacement level** — league-wide constant (NOT team-specific), calibrated to what a Triple-A callup produces. Better than league-average as baseline because average would classify half the league as negative-value which is nonsensical
- **Surplus value** — `Σ over remaining years [projected_WAR_y × $/WAR × discount^y − salary_y]`. Ranks acquisition targets (not talent). Mike Trout = best player, bad target. Young pre-arb player = great target
- **Point-in-time** — every fact has an `as_of_date`. Backtest queries filter `WHERE as_of_date <= :decision_date ORDER BY as_of_date DESC LIMIT 1`. Regression test inserts a future-dated row and asserts backtest doesn't see it. `latest()` must not appear in valuation or backtest code
- **Chadwick + entity resolution** — deterministic first pass via `playerid_lookup` (key_mlbam), rapidfuzz fallback on residual, audit report showing % matched / ambiguous / unmatched. Silent join failures = wrong surplus = bad recommendations
- **Steamer** — projection *system* (not raw data) hosted on Fangraphs; weights recent seasons + regresses to mean + aging adjustments; revised throughout the season → hence `as_of_date` in PK

## Sunday sprint plan (the day we build the app)

Cadence per module: I explain concept (2 min) → I write code (10 min) → user reads + asks (15 min) → **user explains it back to me as if I'm the IBM interviewer** (15 min) → next module. If user can't defend it, we don't move on.

| # | Module | Defense checkpoint |
|---|---|---|
| 1 | `ingest.py` — Steamer + Lahman via pybaseball; snapshot to `data/snapshots/{system}_{season}_{YYYY-MM-DD}.parquet` | "Why snapshot instead of always reading current?" |
| 2 | `resolve.py` — Chadwick lookup + rapidfuzz fallback + audit | "What's the hardest bug and how'd you catch it?" |
| 3 | Postgres load from parquet → ORM | (schema already done) "Why is `as_of_date` in the projections PK?" |
| 4 | `valuation.py` — surplus value math | "Walk me through the calc. Why $8M? Why discount? Example of negative surplus." |
| 5 | `recommend.py` — 1-for-1 candidate enumeration, both-sides-positive filter | "Why not just rank by WAR?" |
| 6 | `backtest.py` + `test_backtest_pit.py` — replay a season + regression test | "How do I know your PIT claim isn't a lie?" |
| 7 | `api/main.py` + Dockerfile — 3 endpoints, containerized | "How do I run this?" |
| 8 | `README.md` — Limitations section | "What's this system's biggest weakness?" |

## Post-build plan

- **Monday** — read every file end-to-end without AI, rubber-duck out loud, redraw the 7-box architecture from memory 3 times, say 60-second intro out loud 5 times, update resume bullet.
- **Tue–Fri** — mock interviews with AI on the 6 hard questions from Kar's prep doc, focusing on baseball answers grounded in *my own* code (not the fabricated version). Also drill FHIR Cracker + Vivid Solutions + MedStar as primary story.

## Sunday morning startup ritual

```powershell
cd "C:\Users\venka\Desktop\Machine Learning Projects\baseball_rec_proj"
docker compose up -d
.venv\Scripts\Activate.ps1
```

Then message the AI: **"ready — start Module 1 (ingest.py)"**.

## Prep tasks for Sunday morning (5 min)

1. Google **"Fangraphs dollars per WAR 2024"** or **"MLB Trade Rumors free agent value per WAR"** — bookmark one article. Need URL to cite when asked "why $8M?"
2. Skim one paragraph on **"Steamer projections"** — enough to say who publishes it, roughly how it works, that it updates through the season.

## Key file paths on my machine

- **This project:** `C:\Users\venka\Desktop\Machine Learning Projects\baseball_rec_proj`
- **IBM interview prep sheet (Kar's, built around baseball):** `C:\Users\venka\Desktop\Machine Learning Projects\baseball_rec_proj\Venkata Kandalai IBM Interview Prep.pdf`
- **My real resume:** `C:\Users\venka\Downloads\Venkata_Krishna_Kandalai_Resume_2026.pdf`
- **Kar's version of resume (do NOT send):** `C:\Users\venka\Downloads\Venkata Kandalai AI Engineer Resume(1).pdf`
- **FHIR Cracker repo (my capstone):** `C:\Users\venka\Documents\GitHub\FHIR-Cracker`
- **Kar's fantasy baseball repo (NOT mine):** `C:\Users\venka\Documents\GitHub\baseball-algorithm`
- **Approved 7-day plan (now compressed to Sunday sprint):** `C:\Users\venka\.claude\plans\help-me-build-the-expressive-feigenbaum.md`

## What I need help with

[Fill in your specific ask — e.g., "start Sunday Module 1 (ingest.py)," "quiz me on surplus value math," "mock interview on the 6 hard questions from Kar's prep," "review the README Limitations section for honesty," etc.]
