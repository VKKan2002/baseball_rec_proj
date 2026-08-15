"""Entity resolution: projections_raw.player_name -> players.mlbam_id.

Layered strategy, most confident first:

    1. exact          - normalized name uniquely matches one Chadwick row
    2. active_narrow  - multiple Chadwick rows match, but only one is
                        "active" (appeared in war_actuals within the last
                        ACTIVE_LOOKBACK_SEASONS seasons)
    3. team_narrow    - still multiple active candidates, but only one has
                        recent war_actuals rows for the projection's team
    4. fuzzy          - rapidfuzz token_set_ratio against the active pool,
                        accepted iff score >= FUZZY_THRESHOLD. partial_ratio
                        was tried but rewarded substring matches too heavily
                        (e.g. "Max Anderson" -> "Tim Anderson") so it is out.
    5. ambiguous      - multiple active candidates; refuse to guess
    6. unresolved     - no candidate above threshold
    7. empty_name     - projection had a blank name

Manual overrides (data/manual_overrides.csv) are checked FIRST, ahead of the
automated layers above, for cases the algorithm can't or shouldn't guess:
nickname mismatches against Chadwick (e.g. "Matthew Boyd" -> "Matt Boyd"),
same-name collisions disambiguated by hand, and rookies with no MLBAM id in
Chadwick yet (tracked with a blank mlbam_id so they show up as
"rookie_pending" instead of a plain "unresolved").

Writes one row per projection into `resolutions`. Idempotent by full
rebuild (TRUNCATE + INSERT) since the table is fully derivable.

Run:
    python -m src.resolve
"""
from __future__ import annotations

import logging
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import text

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

OVERRIDES_PATH = _ROOT / "data" / "manual_overrides.csv"

from src.db import engine  # noqa: E402
from src.models import Base  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
)
log = logging.getLogger("resolve")

FUZZY_THRESHOLD = 88               # rapidfuzz score below which we refuse to match
ACTIVE_LOOKBACK_SEASONS = 3        # a player is "active" if seen within this many recent seasons

# Mr Cheat Sheet uses 2-letter/short codes for some teams; bwar uses the
# Baseball-Reference 3-letter codes. Map each side to its canonical set.
_TEAM_ALIASES: dict[str, set[str]] = {
    "SD":  {"SDP", "SD"},
    "SDP": {"SDP", "SD"},
    "SF":  {"SFG", "SF"},
    "SFG": {"SFG", "SF"},
    "TB":  {"TBR", "TB", "TBD"},
    "TBR": {"TBR", "TB", "TBD"},
    "KC":  {"KCR", "KC"},
    "KCR": {"KCR", "KC"},
    "WSN": {"WSN", "WAS"},
    "WAS": {"WSN", "WAS"},
    "CWS": {"CHW", "CWS"},
    "CHW": {"CHW", "CWS"},
    "AZ":  {"ARI", "AZ"},
    "ARI": {"ARI", "AZ"},
}


def _team_key(team: str | None) -> set[str]:
    if not team:
        return set()
    t = team.strip().upper()
    return _TEAM_ALIASES.get(t, {t})


# ---------------------------------------------------------------- normalize --

_SUFFIX_RE = re.compile(r"\s+(jr|sr|ii|iii|iv|v)$", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[.,'`\"’]")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*")   # e.g. "Max Muncy (LAD)" -> "Max Muncy"
_WS_RE = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    """Lowercase, strip accents/punctuation/parenthetical suffixes, collapse whitespace."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFD", name)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _PAREN_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    s = _WS_RE.sub(" ", s).strip()
    s = _SUFFIX_RE.sub("", s).strip()
    return s


# ----------------------------------------------------------------- loaders --

def _load_players() -> pd.DataFrame:
    with engine.connect() as c:
        df = pd.read_sql(
            text("SELECT mlbam_id, first_name, last_name FROM players"), c
        )
    df["full_name"] = (
        df["first_name"].fillna("") + " " + df["last_name"].fillna("")
    ).str.strip()
    df["norm"] = df["full_name"].map(normalize_name)
    return df


def _load_active_pool(current_season: int) -> set[int]:
    cutoff = current_season - ACTIVE_LOOKBACK_SEASONS + 1
    with engine.connect() as c:
        rows = c.execute(
            text("SELECT DISTINCT mlbam_id FROM war_actuals WHERE season >= :y"),
            {"y": cutoff},
        ).all()
    return {r[0] for r in rows}


def _load_recent_teams(current_season: int) -> dict[int, set[str]]:
    """Map each active mlbam_id -> set of teams they've played for recently."""
    cutoff = current_season - ACTIVE_LOOKBACK_SEASONS + 1
    with engine.connect() as c:
        rows = c.execute(
            text(
                "SELECT DISTINCT mlbam_id, team_id FROM war_actuals "
                "WHERE season >= :y AND team_id IS NOT NULL"
            ),
            {"y": cutoff},
        ).all()
    d: dict[int, set[str]] = defaultdict(set)
    for mlbam, team in rows:
        d[mlbam].add(team)
    return d


def _load_overrides() -> dict[tuple[str, str], int | None]:
    """(normalized name, team) -> mlbam_id, or None for a tracked-but-unresolved
    rookie (blank mlbam_id in the CSV -- no Chadwick id assigned yet)."""
    if not OVERRIDES_PATH.exists():
        return {}
    df = pd.read_csv(OVERRIDES_PATH, dtype={"mlbam_id": "Int64"})
    out: dict[tuple[str, str], int | None] = {}
    for row in df.itertuples(index=False):
        key = (normalize_name(row.player_name), str(row.team).strip())
        out[key] = int(row.mlbam_id) if pd.notna(row.mlbam_id) else None
    return out


def _load_projections() -> pd.DataFrame:
    with engine.connect() as c:
        return pd.read_sql(
            text(
                "SELECT id, player_name, team, primary_pos, season, as_of_date "
                "FROM projections_raw ORDER BY id"
            ),
            c,
        )


# ----------------------------------------------------------------- resolve --

def _build_indexes(
    players: pd.DataFrame, active_pool: set[int]
) -> tuple[dict[str, list[int]], list[str], list[int]]:
    by_norm: dict[str, list[int]] = defaultdict(list)
    for row in players.itertuples(index=False):
        by_norm[row.norm].append(row.mlbam_id)

    active = players[players["mlbam_id"].isin(active_pool)]
    return by_norm, active["norm"].tolist(), active["mlbam_id"].tolist()


def _resolve_one(
    name: str,
    team: str | None,
    by_norm: dict[str, list[int]],
    active_pool: set[int],
    active_names: list[str],
    active_ids: list[int],
    team_by_mlbam: dict[int, set[str]],
    overrides: dict[tuple[str, str], int | None],
) -> tuple[int | None, str, float, int]:
    """Return (mlbam_id_or_None, method, score, candidate_count)."""
    norm = normalize_name(name)
    if not norm:
        return None, "empty_name", 0.0, 0

    override_key = (norm, str(team).strip() if team else "")
    if override_key in overrides:
        mlbam = overrides[override_key]
        if mlbam is not None:
            return mlbam, "manual_override", 100.0, 1
        return None, "rookie_pending", 0.0, 0

    candidates = by_norm.get(norm, [])
    n = len(candidates)

    if n == 1:
        return candidates[0], "exact", 100.0, 1

    if n > 1:
        active_hits = [c for c in candidates if c in active_pool]
        if len(active_hits) == 1:
            return active_hits[0], "active_narrow", 95.0, n
        # tie-break by team: does exactly one active candidate have recent
        # war_actuals rows for the projection's team?
        wanted = _team_key(team)
        if len(active_hits) > 1 and wanted:
            team_hits = [c for c in active_hits if team_by_mlbam.get(c, set()) & wanted]
            if len(team_hits) == 1:
                return team_hits[0], "team_narrow", 92.0, n
        return None, "ambiguous", 0.0, n

    # no exact match -- fuzzy against the active pool with token_set_ratio only.
    if not active_names:
        return None, "unresolved", 0.0, 0
    best = process.extractOne(norm, active_names, scorer=fuzz.token_set_ratio)
    if best is None:
        return None, "unresolved", 0.0, 0
    _match_str, score, idx = best
    if score < FUZZY_THRESHOLD:
        return None, "unresolved", float(score), 0
    # Team check: if the projection specifies a team and the fuzzy candidate
    # has no recent rows for that team, refuse the match. This catches
    # same-last-name collisions like Emmanuel (MIN) vs Manuel (TBR) Rodriguez.
    cand_mlbam = active_ids[idx]
    wanted = _team_key(team)
    if wanted and cand_mlbam in team_by_mlbam:
        if not (team_by_mlbam[cand_mlbam] & wanted):
            return None, "unresolved", float(score), 0
    return cand_mlbam, "fuzzy", float(score), 1


# ---------------------------------------------------------------- writing ---

def _write(df: pd.DataFrame) -> None:
    df = df.copy()
    df["mlbam_id"] = df["mlbam_id"].astype("Int64")
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE resolutions"))
    df.to_sql("resolutions", engine, if_exists="append", index=False, chunksize=1000)
    log.info("wrote %d resolutions", len(df))


def _log_summary(df: pd.DataFrame, projections: pd.DataFrame) -> None:
    counts = df["method"].value_counts()
    log.info("resolution summary:")
    for method, count in counts.items():
        pct = 100 * count / len(df)
        log.info("  %-14s %4d (%.1f%%)", method, count, pct)

    resolved = int(df["mlbam_id"].notna().sum())
    log.info("resolved %d / %d (%.1f%%)", resolved, len(df), 100 * resolved / len(df))

    joined = df.merge(projections, left_on="projection_raw_id", right_on="id")

    fuzzy = joined[joined["method"] == "fuzzy"].sort_values("score")
    if not fuzzy.empty:
        log.info("fuzzy matches (lowest-scoring first, review these):")
        for row in fuzzy.head(5).itertuples(index=False):
            log.info("  %.0f  %-28s -> mlbam=%s", row.score, row.player_name, row.mlbam_id)

    unresolved = joined[joined["method"].isin(["unresolved", "ambiguous"])]
    if not unresolved.empty:
        log.info("unresolved samples (first 10):")
        for row in unresolved.head(10).itertuples(index=False):
            log.info("  [%s] %-28s team=%s pos=%s",
                     row.method, row.player_name, row.team, row.primary_pos)


# ------------------------------------------------------------------- main ---

def resolve_all(current_season: int = 2026) -> pd.DataFrame:
    Base.metadata.create_all(engine)

    players = _load_players()
    active_pool = _load_active_pool(current_season)
    team_by_mlbam = _load_recent_teams(current_season)
    by_norm, active_names, active_ids = _build_indexes(players, active_pool)
    overrides = _load_overrides()
    log.info(
        "loaded %d players; %d active since %d; %d manual overrides (%d pending rookie ids)",
        len(players), len(active_pool), current_season - ACTIVE_LOOKBACK_SEASONS + 1,
        len(overrides), sum(1 for v in overrides.values() if v is None),
    )

    projs = _load_projections()
    log.info("resolving %d projection rows...", len(projs))

    today = date.today()
    results = []
    for row in projs.itertuples(index=False):
        mlbam, method, score, cand = _resolve_one(
            row.player_name, row.team,
            by_norm, active_pool, active_names, active_ids, team_by_mlbam,
            overrides,
        )
        results.append({
            "projection_raw_id": row.id,
            "mlbam_id": mlbam,
            "method": method,
            "score": score,
            "candidate_count": cand,
            "resolved_at": today,
        })

    out = pd.DataFrame(results)
    _write(out)
    _log_summary(out, projs)
    return out


if __name__ == "__main__":
    resolve_all()
