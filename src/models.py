from __future__ import annotations

from datetime import date

from sqlalchemy import Boolean, Date, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Player(Base):
    __tablename__ = "players"

    mlbam_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(64))
    last_name: Mapped[str] = mapped_column(String(64))
    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    primary_position: Mapped[str | None] = mapped_column(String(4), nullable=True)


class Projection(Base):
    __tablename__ = "projections"

    mlbam_id: Mapped[int] = mapped_column(ForeignKey("players.mlbam_id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    system: Mapped[str] = mapped_column(String(16), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)

    projected_war: Mapped[float] = mapped_column(Float)
    projected_pa: Mapped[int | None] = mapped_column(Integer, nullable=True)
    projected_ip: Mapped[float | None] = mapped_column(Float, nullable=True)


class Contract(Base):
    __tablename__ = "contracts"

    mlbam_id: Mapped[int] = mapped_column(ForeignKey("players.mlbam_id"), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    salary_usd: Mapped[float] = mapped_column(Float)
    contract_status: Mapped[str] = mapped_column(String(16))


class Roster(Base):
    __tablename__ = "rosters"

    mlbam_id: Mapped[int] = mapped_column(ForeignKey("players.mlbam_id"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)
    team: Mapped[str] = mapped_column(String(4))


class TeamSeason(Base):
    __tablename__ = "team_seasons"

    team: Mapped[str] = mapped_column(String(4), primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    payroll_cap_usd: Mapped[float] = mapped_column(Float)
    positional_needs: Mapped[str] = mapped_column(String(64))


class WarActual(Base):
    """Historical WAR + salary from Baseball-Reference (via pybaseball.bwar_*).

    One row per (player, season, stint, role). Salary lives here because bwar
    ships it in the same call. as_of_date preserves point-in-time snapshots so
    backtests can filter to "what we knew as of X".
    """
    __tablename__ = "war_actuals"

    mlbam_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    stint: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(5), primary_key=True)  # 'bat' | 'pitch'
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)

    name_common: Mapped[str | None] = mapped_column(String(96), nullable=True)
    bbref_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    team_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    league_id: Mapped[str | None] = mapped_column(String(8), nullable=True)
    games: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    war: Mapped[float | None] = mapped_column(Float, nullable=True)
    war_rep: Mapped[float | None] = mapped_column(Float, nullable=True)
    waa: Mapped[float | None] = mapped_column(Float, nullable=True)
    # fielding runs above average. bwar_bat() only -- bwar_pitch() has no
    # equivalent column, so this is always NULL for role='pitch' rows.
    def_runs: Mapped[float | None] = mapped_column(Float, nullable=True)


class ProjectionRaw(Base):
    """Raw projections as ingested from external sources (Mr Cheat Sheet, etc.).

    Unlike `Projection`, this holds the source's native columns (counting stats,
    fantasy metrics) before entity resolution or WAR estimation. The downstream
    modelling layer reads from here and writes to `projections` with a WAR value.
    """
    __tablename__ = "projections_raw"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(32))
    as_of_date: Mapped[date] = mapped_column(Date)
    season: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(5))  # 'bat' | 'pitch'

    player_name: Mapped[str] = mapped_column(String(96))
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team: Mapped[str | None] = mapped_column(String(8), nullable=True)
    primary_pos: Mapped[str | None] = mapped_column(String(8), nullable=True)
    other_pos: Mapped[str | None] = mapped_column(String(32), nullable=True)

    pa: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ab: Mapped[int | None] = mapped_column(Integer, nullable=True)
    h: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hr: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    so: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    obp: Mapped[float | None] = mapped_column(Float, nullable=True)
    slg: Mapped[float | None] = mapped_column(Float, nullable=True)
    ops: Mapped[float | None] = mapped_column(Float, nullable=True)
    r: Mapped[int | None] = mapped_column(Integer, nullable=True)
    doubles: Mapped[int | None] = mapped_column(Integer, nullable=True)
    triples: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rbi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    w: Mapped[int | None] = mapped_column(Integer, nullable=True)
    l: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ip: Mapped[float | None] = mapped_column(Float, nullable=True)
    er: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sv: Mapped[int | None] = mapped_column(Integer, nullable=True)
    era: Mapped[float | None] = mapped_column(Float, nullable=True)
    whip: Mapped[float | None] = mapped_column(Float, nullable=True)
    k_per_9: Mapped[float | None] = mapped_column(Float, nullable=True)
    gs: Mapped[int | None] = mapped_column(Integer, nullable=True)


class SeasonStats(Base):
    """Season-level counting stats from Baseball-Reference (batting_stats_bref /
    pitching_stats_bref via pybaseball).

    One row per (mlbam_id, season, role, as_of_date). Traded players are
    collapsed to the highest-PA (bat) or highest-IP (pitch) row. Column names
    like `h`, `hr`, `bb`, `so`, `hbp`, `r` mean different things depending on
    role -- e.g. `hr` = home runs hit (bat) vs allowed (pitch). The `role`
    discriminator resolves the ambiguity.

    These are the features that feed the WAR estimator (forecast.py). The
    target `war` is joined in from `war_actuals` on (mlbam_id, season, role).
    """
    __tablename__ = "season_stats"

    mlbam_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(5), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)

    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    games: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # shared / batting
    pa: Mapped[int | None] = mapped_column(Integer, nullable=True)   # plate appearances
    ab: Mapped[int | None] = mapped_column(Integer, nullable=True)   # at-bats (PA minus walks/HBP/sac)
    r: Mapped[int | None] = mapped_column(Integer, nullable=True)    # runs scored
    h: Mapped[int | None] = mapped_column(Integer, nullable=True)    # hits
    doubles: Mapped[int | None] = mapped_column(Integer, nullable=True)  # two-base hits
    triples: Mapped[int | None] = mapped_column(Integer, nullable=True)  # three-base hits
    hr: Mapped[int | None] = mapped_column(Integer, nullable=True)   # home runs
    rbi: Mapped[int | None] = mapped_column(Integer, nullable=True)  # runs batted in
    bb: Mapped[int | None] = mapped_column(Integer, nullable=True)   # walks (batter); walks allowed (pitcher)
    so: Mapped[int | None] = mapped_column(Integer, nullable=True)   # strikeouts (batter); strikeouts recorded (pitcher)
    hbp: Mapped[int | None] = mapped_column(Integer, nullable=True)  # hit by pitch
    sb: Mapped[int | None] = mapped_column(Integer, nullable=True)   # stolen bases
    cs: Mapped[int | None] = mapped_column(Integer, nullable=True)   # caught stealing
    ba: Mapped[float | None] = mapped_column(Float, nullable=True)   # batting average (H/AB)
    obp: Mapped[float | None] = mapped_column(Float, nullable=True)  # on-base percentage (how often batter reaches base)
    slg: Mapped[float | None] = mapped_column(Float, nullable=True)  # slugging percentage (total bases / AB, measures power)
    ops: Mapped[float | None] = mapped_column(Float, nullable=True)  # on-base + slugging, composite offensive rate stat

    # pitching-only
    gs: Mapped[int | None] = mapped_column(Integer, nullable=True)   # games started
    w: Mapped[int | None] = mapped_column(Integer, nullable=True)    # wins
    l: Mapped[int | None] = mapped_column(Integer, nullable=True)    # losses
    sv: Mapped[int | None] = mapped_column(Integer, nullable=True)   # saves (closer stat)
    ip: Mapped[float | None] = mapped_column(Float, nullable=True)   # innings pitched
    er: Mapped[int | None] = mapped_column(Integer, nullable=True)   # earned runs allowed
    era: Mapped[float | None] = mapped_column(Float, nullable=True)  # earned run average (ER per 9 innings, lower = better)
    whip: Mapped[float | None] = mapped_column(Float, nullable=True) # walks + hits per inning pitched (lower = better)
    so9: Mapped[float | None] = mapped_column(Float, nullable=True)  # strikeouts per 9 innings (k/9)
    so_bb: Mapped[float | None] = mapped_column(Float, nullable=True) # strikeout-to-walk ratio (command metric)
    bf: Mapped[int | None] = mapped_column(Integer, nullable=True)   # batters faced (pitcher's equivalent of PA)


class Valuation(Base):
    """Surplus value from valuation.py: projections (year-1 WAR) + an aging
    curve for years 2-3, minus salary (war_actuals for year 0, manually
    curated data/manual_contracts.csv for years 1+).

    Fully derivable from those inputs, so it's rebuilt per as_of_date
    (delete-by-date then insert) same as the other derived tables.
    """
    __tablename__ = "valuations"

    mlbam_id: Mapped[int] = mapped_column(ForeignKey("players.mlbam_id"), primary_key=True)
    as_of_date: Mapped[date] = mapped_column(Date, primary_key=True)

    age: Mapped[int] = mapped_column(Integer)
    war_year1: Mapped[float] = mapped_column(Float)
    surplus_value: Mapped[float] = mapped_column(Float)
    # true if any horizon year fell back to LEAGUE_MINIMUM_SALARY because no
    # real salary was found (missing from war_actuals and manual_contracts.csv)
    salary_estimated: Mapped[bool] = mapped_column(Boolean)


class StatcastStats(Base):
    """Pre-aggregated Statcast leaderboard data from Baseball Savant (via pybaseball).

    One row per (mlbam_id, season, role). Covers 2015+. Used as additional
    features in forecast.py alongside season_stats. Role='bat' rows come from
    statcast_batter_expected_stats; role='pitch' rows from the pitcher equivalent.
    """
    __tablename__ = "statcast_stats"

    mlbam_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    season: Mapped[int] = mapped_column(Integer, primary_key=True)
    role: Mapped[str] = mapped_column(String(5), primary_key=True)  # 'bat' | 'pitch'

    xwoba: Mapped[float | None] = mapped_column(Float, nullable=True)
    barrel_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_velo_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    sweet_spot_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float, nullable=True)


class Resolution(Base):
    """Links a raw projection row to a canonical player (or marks it unresolved).

    Fully derivable from projections_raw + players + war_actuals, so it is
    rebuilt from scratch on every resolve run (TRUNCATE + INSERT).
    """
    __tablename__ = "resolutions"

    projection_raw_id: Mapped[int] = mapped_column(
        ForeignKey("projections_raw.id"), primary_key=True
    )
    mlbam_id: Mapped[int | None] = mapped_column(
        ForeignKey("players.mlbam_id"), nullable=True
    )
    method: Mapped[str] = mapped_column(String(24))           # exact | active_narrow | fuzzy | ambiguous | unresolved | empty_name
    score: Mapped[float] = mapped_column(Float)               # 0-100 (100 = exact)
    candidate_count: Mapped[int] = mapped_column(Integer)     # # of chadwick rows sharing the normalized name
    resolved_at: Mapped[date] = mapped_column(Date)
