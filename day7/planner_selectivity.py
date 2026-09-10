import sqlite3
import statistics
import time
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"

REPEAT = 15

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# -----------------------------------------
# score용 index
# -----------------------------------------

# 이번 실험에 방해되는 이전 covering index 제거
cur.execute("""
DROP INDEX IF EXISTS idx_users_age_score
""")

cur.execute("""
DROP INDEX IF EXISTS idx_users_age
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_users_score
ON users(score)
""")

conn.commit()

cur.execute("ANALYZE")
conn.commit()

print()
print("=== SQLITE INFO ===")

print(
    "SQLite version:",
    sqlite3.sqlite_version,
)

compile_options = [
    row[0]
    for row in cur.execute(
        "PRAGMA compile_options"
    ).fetchall()
]

stat_options = [
    option
    for option in compile_options
    if "STAT" in option
]

print(
    "STAT compile options:",
    stat_options,
)


print()
print("=== sqlite_stat1 ===")

rows = cur.execute("""
    SELECT *
    FROM sqlite_stat1
    WHERE tbl = 'users'
""").fetchall()

for row in rows:
    print(row)


print()
print("=== sqlite_stat4 exists? ===")

stat4 = cur.execute("""
    SELECT name
    FROM sqlite_schema
    WHERE name = 'sqlite_stat4'
""").fetchone()

print(stat4)


thresholds = [
    99.9,
    99,
    95,
    90,
    75,
    50,
    25,
    0,
]


def benchmark(sql, value):
    # warm-up
    for _ in range(3):
        cur.execute(
            sql,
            (value,),
        ).fetchone()

    times = []

    for _ in range(REPEAT):
        start = time.perf_counter_ns()

        cur.execute(
            sql,
            (value,),
        ).fetchone()

        elapsed = (
            time.perf_counter_ns()
            - start
        )

        times.append(
            elapsed / 1_000_000
        )

    return statistics.median(times)


print(
    f"{'threshold':>10} "
    f"{'selectivity':>12} "
    f"{'planner':>45} "
    f"{'index(ms)':>10} "
    f"{'scan(ms)':>10}"
)

print("-" * 100)


for threshold in thresholds:

    # -----------------------------------------
    # 실제 몇 %의 row가 조건에 걸리는지
    # -----------------------------------------

    count = cur.execute("""
        SELECT COUNT(*)
        FROM users
        WHERE score >= ?
    """, (threshold,)).fetchone()[0]

    total = cur.execute("""
        SELECT COUNT(*)
        FROM users
    """).fetchone()[0]

    selectivity = (
        count / total * 100
    )


    # -----------------------------------------
    # planner가 자율적으로 고른 plan
    # -----------------------------------------

    planner_sql = """
        SELECT SUM(age)
        FROM users
        WHERE score >= ?
    """

    plan = cur.execute(
        "EXPLAIN QUERY PLAN "
        + planner_sql,
        (threshold,),
    ).fetchall()

    plan_text = " | ".join(
        row[3]
        for row in plan
    )


    # -----------------------------------------
    # 강제로 INDEX 사용
    # -----------------------------------------

    index_sql = """
        SELECT SUM(age)
        FROM users
        INDEXED BY idx_users_score
        WHERE score >= ?
    """


    # -----------------------------------------
    # 강제로 TABLE SCAN
    # -----------------------------------------

    scan_sql = """
        SELECT SUM(age)
        FROM users
        NOT INDEXED
        WHERE score >= ?
    """


    #
    # 실행 순서가 항상 같아서 생기는
    # cache bias를 조금 줄이기 위해
    # 순서를 랜덤하게 함
    #

    order = [
        ("index", index_sql),
        ("scan", scan_sql),
    ]

    random.shuffle(order)

    results = {}

    for name, sql in order:
        results[name] = benchmark(
            sql,
            threshold,
        )


    print(
        f"{threshold:10.1f} "
        f"{selectivity:11.1f}% "
        f"{plan_text:45s} "
        f"{results['index']:10.3f} "
        f"{results['scan']:10.3f}"
    )


conn.close()