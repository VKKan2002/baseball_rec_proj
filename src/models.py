from __future__ import annotations

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, Integer, String
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
