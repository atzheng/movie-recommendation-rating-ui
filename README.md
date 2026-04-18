# Rating UI (Standalone App, Postgres-backed)

This app is separate from the recommender starter API.

It provides a side-by-side judging UI where each student (via codename URL) sees two random team recommendations and picks a winner.

## Routes

- `GET /judge/{codename}`: main rating page
- `GET /api/session/{codename}/round`: fetch two random distinct team recommendations
- `POST /api/session/{codename}/vote`: persist selected winner into Postgres
- `GET /`: health check

## Environment variables

Required:

- `DATABASE_URL` (Leapcell Postgres connection string)

Optional:

- `TEAMS_TABLE` (default: `teams`)
- `STUDENTS_TABLE` (default: `students`)
- `RESULTS_TABLE` (default: `results`)
- `TMDB_CSV_PATH` (default points to `./tmdb_top1000_movies.csv`)

## Expected Postgres schema

```sql
CREATE TABLE IF NOT EXISTS teams (
  id BIGSERIAL PRIMARY KEY,
  team_name TEXT NOT NULL,
  api_url TEXT NOT NULL,
  enabled BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS students (
  id BIGSERIAL PRIMARY KEY,
  codename TEXT NOT NULL UNIQUE,
  preferences TEXT,
  history JSONB NOT NULL DEFAULT '[]'::jsonb,
  user_id BIGINT
);

CREATE TABLE IF NOT EXISTS results (
  id BIGSERIAL PRIMARY KEY,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  round_id TEXT NOT NULL,
  student_codename TEXT NOT NULL,
  student_preferences TEXT NOT NULL,
  winner_team TEXT NOT NULL,
  loser_team TEXT NOT NULL,
  winner_api_url TEXT NOT NULL,
  loser_api_url TEXT NOT NULL,
  winner_tmdb_id BIGINT NOT NULL,
  loser_tmdb_id BIGINT NOT NULL,
  winner_description TEXT NOT NULL,
  loser_description TEXT NOT NULL
);
```

## Sample seed data

```sql
INSERT INTO students (codename, history, user_id)
VALUES
  ('blue-otter', '[]'::jsonb, 1001),
  ('quiet-panda', '[]'::jsonb, 1002);

INSERT INTO teams (team_name, api_url, enabled)
VALUES
  ('Team Alpha', 'https://alpha.example.com/recommend', TRUE),
  ('Team Beta', 'https://beta.example.com/recommend', TRUE);
```

## Local run

```bash
cd rating-ui
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DBNAME
uvicorn app:app --reload
```

Then open a student link:

`http://localhost:8000/judge/blue-otter`

## Leapcell deploy

This folder includes `leapcell.yaml`:

- build: `pip install -r requirements.txt`
- run: `uvicorn app:app --host 0.0.0.0 --port 8080`

Set `DATABASE_URL` in Leapcell service env vars.
