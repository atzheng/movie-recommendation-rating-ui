import asyncio
import json
import os
import re
from typing import Any

import asyncpg
import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Movie Recommender Judge UI")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TEAMS_TABLE = os.environ.get("TEAMS_TABLE", "teams")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "results")

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"
TMDB_CSV_PATH = os.environ.get(
    "TMDB_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv"),
)

if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", TEAMS_TABLE):
    raise RuntimeError("Invalid TEAMS_TABLE name")
if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", RESULTS_TABLE):
    raise RuntimeError("Invalid RESULTS_TABLE name")

TMDB_DF = pd.read_csv(TMDB_CSV_PATH)
TMDB_LOOKUP = {
    int(row.tmdb_id): {
        "title": str(row.title),
        "year": int(row.year) if pd.notna(row.year) else None,
        "poster_path": str(row.poster_path) if pd.notna(row.poster_path) else "",
        "tmdb_url": str(row.tmdb_url) if pd.notna(row.tmdb_url) else "",
    }
    for row in TMDB_DF.itertuples()
}

MOVIES_JSON = json.dumps(sorted(
    [{"tmdb_id": tid, "title": m["title"], "year": m["year"]} for tid, m in TMDB_LOOKUP.items()],
    key=lambda m: m["title"],
))


class VoteRequest(BaseModel):
    round_id: str
    winner_side: str
    left: dict[str, Any]
    right: dict[str, Any]
    preferences: str = ""


def enrich_recommendation(raw: dict[str, Any]) -> dict[str, Any]:
    tmdb_id = int(raw.get("tmdb_id", -1))
    movie = TMDB_LOOKUP.get(tmdb_id, {})
    poster_path = movie.get("poster_path", "")
    return {
        "tmdb_id": tmdb_id,
        "description": str(raw.get("description", ""))[:500],
        "title": movie.get("title", f"TMDB #{tmdb_id}"),
        "year": movie.get("year"),
        "tmdb_url": movie.get("tmdb_url", f"https://www.themoviedb.org/movie/{tmdb_id}"),
        "poster_url": f"{POSTER_BASE_URL}{poster_path}" if poster_path else "",
    }


async def call_team_api(
    client: httpx.AsyncClient, api_url: str, payload: dict[str, Any]
) -> dict[str, Any]:
    resp = await client.post(f"{api_url}/recommend", json=payload, timeout=8.0)
    if resp.status_code >= 300:
        raise HTTPException(
            status_code=502, detail=f"Team API failed ({api_url}): HTTP {resp.status_code}"
        )
    body = resp.json()
    if "tmdb_id" not in body or "description" not in body:
        raise HTTPException(
            status_code=502, detail=f"Team API returned invalid JSON shape ({api_url})"
        )
    return body


@app.on_event("startup")
async def startup() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be set")
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=10)


@app.on_event("shutdown")
async def shutdown() -> None:
    pool = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()


@app.get("/")
def health():
    return {"status": "ok", "app": "rating-ui"}


@app.get("/judge", response_class=HTMLResponse)
def judge_page(request: Request):
    return templates.TemplateResponse(request, "judge.html", {
        "movies_json": MOVIES_JSON,
    })


@app.get("/api/session/preferences")
async def get_preferences():
    return {"preferences": ""}


@app.get("/api/session/round")
async def get_round(preferences: str = Query(default="")):
    prefs = preferences.strip()
    if not prefs:
        raise HTTPException(status_code=400, detail="Please enter your preferences before loading recommendations")

    pool = app.state.db_pool
    teams_sql = (
        f"SELECT id, team_name, api_url "
        f"FROM {TEAMS_TABLE} WHERE enabled = TRUE ORDER BY random() LIMIT 2"
    )

    async with pool.acquire() as conn:
        teams = await conn.fetch(teams_sql)
        if len(teams) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 enabled teams")

    left_team, right_team = teams[0], teams[1]
    req_payload = {
        "user_id": 0,
        "preferences": prefs,
        "history": [],
    }

    async with httpx.AsyncClient() as client:
        left_raw, right_raw = await asyncio.gather(
            call_team_api(client, left_team["api_url"], req_payload),
            call_team_api(client, right_team["api_url"], req_payload),
        )

    return {
        "round_id": os.urandom(10).hex(),
        "preferences": prefs,
        "left": {
            "team": {
                "record_id": left_team["id"],
                "name": left_team["team_name"],
                "api_url": left_team["api_url"],
            },
            "recommendation": enrich_recommendation(left_raw),
        },
        "right": {
            "team": {
                "record_id": right_team["id"],
                "name": right_team["team_name"],
                "api_url": right_team["api_url"],
            },
            "recommendation": enrich_recommendation(right_raw),
        },
    }


@app.post("/api/session/vote")
async def submit_vote(vote: VoteRequest):
    if vote.winner_side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="winner_side must be 'left' or 'right'")

    winner = vote.left if vote.winner_side == "left" else vote.right
    loser = vote.right if vote.winner_side == "left" else vote.left

    pool = app.state.db_pool
    insert_sql = (
        f"INSERT INTO {RESULTS_TABLE} ("
        "round_id, student_codename, student_preferences, "
        "winner_team, loser_team, winner_api_url, loser_api_url, "
        "winner_tmdb_id, loser_tmdb_id, winner_description, loser_description"
        ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)"
    )

    async with pool.acquire() as conn:
        await conn.execute(
            insert_sql,
            vote.round_id,
            "",
            vote.preferences.strip(),
            winner.get("team", {}).get("name", ""),
            loser.get("team", {}).get("name", ""),
            winner.get("team", {}).get("api_url", ""),
            loser.get("team", {}).get("api_url", ""),
            int(winner.get("recommendation", {}).get("tmdb_id", -1)),
            int(loser.get("recommendation", {}).get("tmdb_id", -1)),
            str(winner.get("recommendation", {}).get("description", "")),
            str(loser.get("recommendation", {}).get("description", "")),
        )

    return {"ok": True}
