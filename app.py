import asyncio
import contextlib
import hashlib
import hmac as hmac_lib
import io
import json
import os
import re
from collections import defaultdict
from typing import Any

import asyncpg
import choix
import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Movie Recommender Judge UI")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

DATABASE_URL = os.environ.get("DATABASE_URL", "")
TEAMS_TABLE = os.environ.get("TEAMS_TABLE", "teams")
STUDENTS_TABLE = os.environ.get("STUDENTS_TABLE", "students")
RESULTS_TABLE = os.environ.get("RESULTS_TABLE", "results")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")
ADMIN_COOKIE = "rating_admin"

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w342"
TMDB_CSV_PATH = os.environ.get(
    "TMDB_CSV_PATH",
    os.path.join(os.path.dirname(__file__), "tmdb_top1000_movies.csv"),
)

for _name, _val in [("TEAMS_TABLE", TEAMS_TABLE), ("STUDENTS_TABLE", STUDENTS_TABLE), ("RESULTS_TABLE", RESULTS_TABLE)]:
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", _val):
        raise RuntimeError(f"Invalid {_name} name")

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


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _admin_token() -> str:
    return hmac_lib.new(ADMIN_PASSWORD.encode(), b"rating-ui-admin-v1", hashlib.sha256).hexdigest()


def _is_admin(request: Request) -> bool:
    if not ADMIN_PASSWORD:
        return False
    token = request.cookies.get(ADMIN_COOKIE, "")
    try:
        return hmac_lib.compare_digest(token, _admin_token())
    except Exception:
        return False


def _require_admin(request: Request) -> None:
    if not _is_admin(request):
        raise HTTPException(status_code=401, detail="Unauthorized")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class VoteRequest(BaseModel):
    round_id: str
    winner_side: str
    left: dict[str, Any]
    right: dict[str, Any]
    preferences: str = ""
    student_id: str = ""


class TeamRequest(BaseModel):
    team_name: str
    api_url: str
    enabled: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        raise ValueError(f"Team API failed ({api_url}): HTTP {resp.status_code}")
    body = resp.json()
    if "tmdb_id" not in body or "description" not in body:
        raise ValueError(f"Team API returned invalid JSON shape ({api_url})")
    return body


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

@contextlib.asynccontextmanager
async def get_conn():
    """Open a direct connection to Postgres (PgBouncer is the pool)."""
    conn = await asyncpg.connect(DATABASE_URL, statement_cache_size=0)
    try:
        yield conn
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL must be set")
    # Best-effort: create students table. Non-fatal so the app always starts
    # even if the DB is momentarily unavailable at boot time.
    try:
        async with get_conn() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {STUDENTS_TABLE} (
                    id bigserial PRIMARY KEY,
                    student_id text UNIQUE NOT NULL,
                    name text NOT NULL DEFAULT '',
                    team_id text NOT NULL DEFAULT ''
                )
            """)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("Startup DB migration skipped: %s", exc)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.get("/")
def health():
    return {"status": "ok", "app": "rating-ui"}


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    authenticated = _is_admin(request)
    error = request.query_params.get("error", "")
    return templates.TemplateResponse(request, "admin.html", {
        "authenticated": authenticated,
        "error": error,
    })


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page(request: Request):
    return templates.TemplateResponse(request, "leaderboard.html", {})


@app.get("/api/leaderboard")
async def get_leaderboard():
    async with get_conn() as conn:
        teams = await conn.fetch(
            f"SELECT team_name FROM {TEAMS_TABLE} ORDER BY id"
        )
        results = await conn.fetch(
            f"SELECT winner_team, loser_team FROM ("
            f"  SELECT winner_team, loser_team, student_codename,"
            f"         ROW_NUMBER() OVER (PARTITION BY student_codename ORDER BY created_at DESC) AS rn"
            f"  FROM {RESULTS_TABLE}"
            f"  WHERE winner_team <> '' AND loser_team <> ''"
            f") sub WHERE rn <= 100"
        )

    team_names = [r["team_name"] for r in teams]
    if not team_names:
        return {"ranking": [], "total_votes": 0, "method": None}

    team_index = {name: i for i, name in enumerate(team_names)}
    n = len(team_names)

    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    pairs: list[tuple[int, int]] = []

    for r in results:
        w, l = r["winner_team"], r["loser_team"]
        if w in team_index and l in team_index:
            pairs.append((team_index[w], team_index[l]))
            wins[w] += 1
            losses[l] += 1

    scores: list[float] | None = None
    method = "win_rate"
    if pairs and n >= 2:
        try:
            params = choix.ilsr_pairwise(n, pairs, alpha=0.01)
            scores = params.tolist()
            method = "Bradley-Terry (ILSR)"
        except Exception:
            scores = None

    ranking = []
    for i, name in enumerate(team_names):
        w, l = wins[name], losses[name]
        total = w + l
        ranking.append({
            "team_name": name,
            "wins": w,
            "losses": l,
            "total": total,
            "win_rate": round(w / total, 4) if total > 0 else None,
            "score": round(scores[i], 4) if scores else None,
        })

    ranking.sort(
        key=lambda x: (x["score"] if x["score"] is not None else float("-inf")),
        reverse=True,
    )
    if scores is None:
        ranking.sort(key=lambda x: (x["win_rate"] or 0), reverse=True)

    return {"ranking": ranking, "total_votes": len(pairs), "method": method}


@app.get("/judge", response_class=HTMLResponse)
def judge_page(request: Request):
    return templates.TemplateResponse(request, "judge.html", {
        "movies_json": MOVIES_JSON,
    })


# ---------------------------------------------------------------------------
# Admin auth
# ---------------------------------------------------------------------------

@app.post("/api/admin/login")
async def admin_login(password: str = Form(...)):
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="No admin password configured")
    if not hmac_lib.compare_digest(password, ADMIN_PASSWORD):
        resp = RedirectResponse(url="/admin?error=wrong+password", status_code=303)
        return resp
    resp = RedirectResponse(url="/admin", status_code=303)
    resp.set_cookie(ADMIN_COOKIE, _admin_token(), httponly=True, samesite="lax")
    return resp


@app.post("/api/admin/logout")
async def admin_logout():
    resp = RedirectResponse(url="/admin", status_code=303)
    resp.delete_cookie(ADMIN_COOKIE)
    return resp


# ---------------------------------------------------------------------------
# Teams API (admin)
# ---------------------------------------------------------------------------

@app.get("/api/teams")
async def list_teams(request: Request):
    _require_admin(request)
    async with get_conn() as conn:
        rows = await conn.fetch(
            f"SELECT id, team_name, api_url, enabled FROM {TEAMS_TABLE} ORDER BY id"
        )
    return [dict(r) for r in rows]


@app.post("/api/teams", status_code=201)
async def create_team(request: Request, team: TeamRequest):
    _require_admin(request)
    async with get_conn() as conn:
        row = await conn.fetchrow(
            f"INSERT INTO {TEAMS_TABLE} (team_name, api_url, enabled) VALUES ($1, $2, $3) RETURNING id, team_name, api_url, enabled",
            team.team_name.strip(), team.api_url.strip(), team.enabled,
        )
    return dict(row)


@app.patch("/api/teams/{team_id}")
async def update_team(team_id: int, request: Request, team: TeamRequest):
    _require_admin(request)
    async with get_conn() as conn:
        row = await conn.fetchrow(
            f"UPDATE {TEAMS_TABLE} SET team_name=$1, api_url=$2, enabled=$3 WHERE id=$4 RETURNING id, team_name, api_url, enabled",
            team.team_name.strip(), team.api_url.strip(), team.enabled, team_id,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Team not found")
    return dict(row)


@app.delete("/api/teams/{team_id}", status_code=204)
async def delete_team(team_id: int, request: Request):
    _require_admin(request)
    async with get_conn() as conn:
        result = await conn.execute(f"DELETE FROM {TEAMS_TABLE} WHERE id=$1", team_id)
    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Team not found")


# ---------------------------------------------------------------------------
# Students API
# ---------------------------------------------------------------------------

@app.get("/api/students")
async def list_students(request: Request):
    _require_admin(request)
    async with get_conn() as conn:
        rows = await conn.fetch(
            f"SELECT id, student_id, name, team_id FROM {STUDENTS_TABLE} ORDER BY team_id, name"
        )
    return [dict(r) for r in rows]


@app.post("/api/students/upload", status_code=201)
async def upload_students(request: Request, file: UploadFile):
    _require_admin(request)
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    col_map = {}
    for col in df.columns:
        if col in ("student_id", "id", "studentid"):
            col_map["student_id"] = col
        elif col in ("name", "student_name", "full_name"):
            col_map["name"] = col
        elif col in ("team_id", "team", "teamid", "group", "group_id"):
            col_map["team_id"] = col

    if "student_id" not in col_map:
        raise HTTPException(status_code=400, detail="CSV must have a student_id (or 'id') column")

    rows = []
    for _, row in df.iterrows():
        sid = str(row[col_map["student_id"]]).strip()
        if not sid:
            continue
        name = str(row[col_map["name"]]).strip() if "name" in col_map else sid
        team_id = str(row[col_map["team_id"]]).strip() if "team_id" in col_map else ""
        rows.append((sid, name, team_id))

    async with get_conn() as conn:
        await conn.executemany(
            f"INSERT INTO {STUDENTS_TABLE} (student_id, name, team_id) VALUES ($1, $2, $3) "
            f"ON CONFLICT (student_id) DO UPDATE SET name=EXCLUDED.name, team_id=EXCLUDED.team_id",
            rows,
        )
    return {"inserted": len(rows)}


@app.delete("/api/students", status_code=204)
async def clear_students(request: Request):
    _require_admin(request)
    async with get_conn() as conn:
        await conn.execute(f"DELETE FROM {STUDENTS_TABLE}")


@app.get("/api/students/{student_id}")
async def get_student(student_id: str):
    async with get_conn() as conn:
        row = await conn.fetchrow(
            f"SELECT student_id, name, team_id FROM {STUDENTS_TABLE} WHERE student_id=$1", student_id
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Student ID not found")
    return dict(row)


# ---------------------------------------------------------------------------
# Judge session
# ---------------------------------------------------------------------------

async def _fetch_teams(student_team: str):
    async with get_conn() as conn:
        if student_team:
            return await conn.fetch(
                f"SELECT id, team_name, api_url FROM {TEAMS_TABLE} "
                f"WHERE enabled = TRUE AND team_name <> $1 ORDER BY random() LIMIT 2",
                student_team,
            )
        return await conn.fetch(
            f"SELECT id, team_name, api_url FROM {TEAMS_TABLE} "
            f"WHERE enabled = TRUE ORDER BY random() LIMIT 2"
        )


async def _record_result(round_id, student_id, prefs, winner, loser, winner_rec, loser_rec):
    async with get_conn() as conn:
        await conn.execute(
            f"INSERT INTO {RESULTS_TABLE} ("
            "round_id, student_codename, student_preferences, "
            "winner_team, loser_team, winner_api_url, loser_api_url, "
            "winner_tmdb_id, loser_tmdb_id, winner_description, loser_description"
            ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            round_id, student_id, prefs,
            winner["team_name"], loser["team_name"],
            winner["api_url"], loser["api_url"],
            int(winner_rec.get("tmdb_id", -1)),
            int(loser_rec.get("tmdb_id", -1)),
            str(winner_rec.get("description", "")),
            str(loser_rec.get("description", "")),
        )


@app.get("/api/session/round")
async def get_round(preferences: str = Query(default=""), student_team: str = Query(default="")):
    prefs = preferences.strip()
    if not prefs:
        raise HTTPException(status_code=400, detail="Please enter your preferences before loading recommendations")

    req_payload = {"user_id": 0, "preferences": prefs, "history": []}

    # Attempt up to twice: once normally, once after recording an auto-win from a partial failure.
    for attempt in range(2):
        teams = await _fetch_teams(student_team.strip())
        if len(teams) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 enabled teams")

        left_team, right_team = teams[0], teams[1]

        async with httpx.AsyncClient() as client:
            left_raw, right_raw = await asyncio.gather(
                call_team_api(client, left_team["api_url"], req_payload),
                call_team_api(client, right_team["api_url"], req_payload),
                return_exceptions=True,
            )

        left_err = isinstance(left_raw, Exception)
        right_err = isinstance(right_raw, Exception)

        if not left_err and not right_err:
            # Both succeeded — return the round.
            return {
                "round_id": os.urandom(10).hex(),
                "preferences": prefs,
                "left": {
                    "team": {"record_id": left_team["id"], "name": left_team["team_name"], "api_url": left_team["api_url"]},
                    "recommendation": enrich_recommendation(left_raw),
                },
                "right": {
                    "team": {"record_id": right_team["id"], "name": right_team["team_name"], "api_url": right_team["api_url"]},
                    "recommendation": enrich_recommendation(right_raw),
                },
            }

        if left_err and right_err:
            if attempt == 0:
                continue  # retry with a fresh pair
            raise HTTPException(status_code=502, detail="Both team APIs failed")

        # Exactly one errored — record auto-win for the successful team.
        winner_team = right_team if left_err else left_team
        loser_team  = left_team  if left_err else right_team
        winner_raw  = right_raw  if left_err else left_raw
        winner_rec  = enrich_recommendation(winner_raw)
        await _record_result(
            os.urandom(10).hex(), "", prefs,
            winner_team, loser_team,
            winner_rec, {"tmdb_id": -1, "description": ""},
        )
        if attempt == 0:
            continue  # fetch a fresh pair to show the user
        raise HTTPException(status_code=502, detail="One team API kept failing after retry")

    raise HTTPException(status_code=502, detail="Could not load a valid round")


@app.post("/api/session/vote")
async def submit_vote(vote: VoteRequest):
    if vote.winner_side not in {"left", "right"}:
        raise HTTPException(status_code=400, detail="winner_side must be 'left' or 'right'")

    winner = vote.left if vote.winner_side == "left" else vote.right
    loser = vote.right if vote.winner_side == "left" else vote.left

    async with get_conn() as conn:
        await conn.execute(
            f"INSERT INTO {RESULTS_TABLE} ("
            "round_id, student_codename, student_preferences, "
            "winner_team, loser_team, winner_api_url, loser_api_url, "
            "winner_tmdb_id, loser_tmdb_id, winner_description, loser_description"
            ") VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
            vote.round_id,
            vote.student_id.strip(),
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
