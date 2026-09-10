import sqlite3
import random
import time
from pathlib import Path


N = 500_000

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# 기존 실험 데이터 초기화
cur.execute(
    "DROP TABLE IF EXISTS users"
)

cur.execute("""
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    score REAL
)
""")


# 50만 row 생성
rows = [
    (
        random.randint(18, 80),
        random.random() * 100,
    )
    for _ in range(N)
]

cur.executemany(
    """
    INSERT INTO users(age, score)
    VALUES (?, ?)
    """,
    rows,
)

conn.commit()


def explain():
    plan = cur.execute("""
        EXPLAIN QUERY PLAN
        SELECT *
        FROM users
        WHERE age = 42
    """).fetchall()

    for row in plan:
        print(row)


def run_query():
    start = time.perf_counter()

    rows = cur.execute("""
        SELECT *
        FROM users
        WHERE age = 42
    """).fetchall()

    elapsed = time.perf_counter() - start

    print(
        f"rows={len(rows)}, "
        f"time={elapsed * 1000:.3f} ms"
    )


print("=== NO INDEX ===")

explain()
run_query()


print()
print("=== CREATE INDEX ===")

cur.execute("""
    CREATE INDEX idx_users_age
    ON users(age)
""")

conn.commit()


print()
print("=== WITH INDEX ===")

explain()
run_query()


conn.close()