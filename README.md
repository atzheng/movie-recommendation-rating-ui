# Rating UI (Standalone App, Postgres-backed)

Side-by-side movie recommendation judging UI. Students enter their student ID, describe their preferences, and repeatedly pick the better of two recommendations. Results feed a Bradley-Terry leaderboard.

## Pages

- `GET /judge` — student judging UI (shared link, no codename)
- `GET /leaderboard` — public leaderboard (Bradley-Terry ranking)
- `GET /admin` — admin UI (password-protected)

## API routes

- `GET /api/session/round?preferences=...&student_team=...` — fetch two random team recommendations
- `POST /api/session/vote` — persist selected winner
- `GET /api/leaderboard` — leaderboard JSON
- `GET /api/teams` — list teams (admin)
- `POST /api/teams` — add team (admin)
- `PATCH /api/teams/{id}` — update/enable/disable team (admin)
- `DELETE /api/teams/{id}` — delete team (admin)
- `GET /api/students` — list students (admin)
- `POST /api/students/upload` — upload students CSV (admin)
- `DELETE /api/students` — clear all students (admin)
- `GET /api/students/{student_id}` — look up one student (used by judge page)

## Environment variables

Required:

- `DATABASE_URL` — Postgres connection string (e.g. `postgresql://user:pass@host:5432/db?sslmode=require`)
- `ADMIN_PASSWORD` — password for the `/admin` page

Optional:

- `TEAMS_TABLE` (default: `teams`)
- `STUDENTS_TABLE` (default: `students`)
- `RESULTS_TABLE` (default: `results`)
- `TMDB_CSV_PATH` (default: `./tmdb_top1000_movies.csv`)

## Postgres schema

The `students` table is auto-created on startup. The other two must exist before the app starts.

```sql
CREATE TABLE IF NOT EXISTS teams (
  id        BIGSERIAL PRIMARY KEY,
  team_name TEXT    NOT NULL,
  api_url   TEXT    NOT NULL,
  enabled   BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS students (
  id         BIGSERIAL PRIMARY KEY,
  student_id TEXT NOT NULL UNIQUE,
  name       TEXT NOT NULL DEFAULT '',
  team_id    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS results (
  id                  BIGSERIAL PRIMARY KEY,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  round_id            TEXT NOT NULL,
  student_codename    TEXT NOT NULL,
  student_preferences TEXT NOT NULL,
  winner_team         TEXT NOT NULL,
  loser_team          TEXT NOT NULL,
  winner_api_url      TEXT NOT NULL,
  loser_api_url       TEXT NOT NULL,
  winner_tmdb_id      BIGINT NOT NULL,
  loser_tmdb_id       BIGINT NOT NULL,
  winner_description  TEXT NOT NULL,
  loser_description   TEXT NOT NULL
);
```

## Sample seed data

```sql
INSERT INTO teams (team_name, api_url, enabled)
VALUES
  ('Team Alpha', 'https://alpha.example.com', TRUE),
  ('Team Beta',  'https://beta.example.com',  TRUE);

INSERT INTO students (student_id, name, team_id)
VALUES
  ('s001', 'Alice', 'Team Alpha'),
  ('s002', 'Bob',   'Team Beta');
```

Or upload a CSV via the admin UI with columns: `student_id`, `name` (optional), `team_id` (optional).

## Team API contract

Each team must expose:

```
POST /recommend
Content-Type: application/json

{ "user_id": 0, "preferences": "...", "history": [] }
```

Response must include `tmdb_id` (integer) and `description` (string).

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
export ADMIN_PASSWORD=secret
uvicorn app:app --reload
```

Then open `http://localhost:8000/judge`.

## Leapcell deploy

Set `DATABASE_URL` and `ADMIN_PASSWORD` in Leapcell's environment variables UI. The other variables have sensible defaults.
