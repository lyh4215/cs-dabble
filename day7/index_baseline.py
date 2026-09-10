import random
import sqlite3
import statistics
import time
from pathlib import Path


N = 500_000
TARGET_AGE = 42
REPEAT = 30

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# --------------------------------------------------
# 1. 데이터 생성
# --------------------------------------------------

random.seed(42)

cur.execute("DROP TABLE IF EXISTS users")

cur.execute("""
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    age INTEGER,
    score REAL
)
""")


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


# --------------------------------------------------
# 2. index 생성
# --------------------------------------------------

cur.execute("""
CREATE INDEX idx_users_age
ON users(age)
""")

cur.execute("""
CREATE INDEX idx_users_age_score
ON users(age, score)
""")

conn.commit()


# --------------------------------------------------
# 3. 동일한 SELECT, 다른 access path
# --------------------------------------------------

queries = {
    "SCAN": """
        SELECT id, age, score
        FROM users NOT INDEXED
        WHERE age = ?
    """,

    "INDEX": """
        SELECT id, age, score
        FROM users INDEXED BY idx_users_age
        WHERE age = ?
    """,

    "COVERING": """
        SELECT id, age, score
        FROM users INDEXED BY idx_users_age_score
        WHERE age = ?
    """,
}


# --------------------------------------------------
# 4. 실행 계획 확인
# --------------------------------------------------

print("=== QUERY PLAN ===")

for name, sql in queries.items():
    plan = cur.execute(
        "EXPLAIN QUERY PLAN " + sql,
        (TARGET_AGE,),
    ).fetchall()

    print(f"\n{name}")

    for row in plan:
        print(row)


# --------------------------------------------------
# 5. warm-up
# --------------------------------------------------

for sql in queries.values():
    for _ in range(5):
        cur.execute(
            sql,
            (TARGET_AGE,),
        ).fetchall()


# --------------------------------------------------
# 6. benchmark
#
# 매 round마다 실행 순서를 섞는다.
# INDEX가 항상 먼저 실행된다든지 하는
# cache/order 효과를 줄이기 위해서다.
# --------------------------------------------------

times = {
    name: []
    for name in queries
}

row_counts = {
    name: None
    for name in queries
}


for _ in range(REPEAT):

    order = list(queries.keys())
    random.shuffle(order)

    for name in order:
        sql = queries[name]

        start = time.perf_counter_ns()

        result = cur.execute(
            sql,
            (TARGET_AGE,),
        ).fetchall()

        elapsed_ns = (
            time.perf_counter_ns()
            - start
        )

        elapsed_ms = (
            elapsed_ns / 1_000_000
        )

        times[name].append(
            elapsed_ms
        )

        row_counts[name] = len(
            result
        )


# --------------------------------------------------
# 7. 결과
# --------------------------------------------------

print()
print("=== ROW FETCH BENCHMARK ===")

for name in (
    "SCAN",
    "INDEX",
    "COVERING",
):
    values = times[name]

    print(
        f"{name:9s}"
        f" rows={row_counts[name]:5d}"
        f" median={statistics.median(values):8.3f} ms"
        f" mean={statistics.mean(values):8.3f} ms"
        f" min={min(values):8.3f} ms"
        f" max={max(values):8.3f} ms"
    )


# --------------------------------------------------
# 8. Python tuple 생성 영향을 더 줄인 benchmark
#
# 똑같은 SUM(score)를 계산한다.
#
# INDEX:
#   age index → table에서 score lookup
#
# COVERING:
#   age+score index에 score가 이미 있음
#
# 결과는 숫자 하나뿐이므로 fetchall()로
# 7~8천 tuple을 만드는 비용도 거의 없다.
# --------------------------------------------------

aggregate_queries = {
    "SCAN": """
        SELECT SUM(score)
        FROM users NOT INDEXED
        WHERE age = ?
    """,

    "INDEX": """
        SELECT SUM(score)
        FROM users INDEXED BY idx_users_age
        WHERE age = ?
    """,

    "COVERING": """
        SELECT SUM(score)
        FROM users INDEXED BY idx_users_age_score
        WHERE age = ?
    """,
}


for sql in aggregate_queries.values():
    for _ in range(5):
        cur.execute(
            sql,
            (TARGET_AGE,),
        ).fetchone()


aggregate_times = {
    name: []
    for name in aggregate_queries
}


for _ in range(REPEAT):

    order = list(
        aggregate_queries.keys()
    )

    random.shuffle(order)

    for name in order:
        sql = aggregate_queries[name]

        start = time.perf_counter_ns()

        result = cur.execute(
            sql,
            (TARGET_AGE,),
        ).fetchone()

        elapsed_ns = (
            time.perf_counter_ns()
            - start
        )

        aggregate_times[name].append(
            elapsed_ns / 1_000_000
        )


print()
print("=== DB-FOCUSED SUM BENCHMARK ===")

for name in (
    "SCAN",
    "INDEX",
    "COVERING",
):
    values = aggregate_times[name]

    print(
        f"{name:9s}"
        f" median={statistics.median(values):8.3f} ms"
        f" mean={statistics.mean(values):8.3f} ms"
        f" min={min(values):8.3f} ms"
        f" max={max(values):8.3f} ms"
    )


conn.close()