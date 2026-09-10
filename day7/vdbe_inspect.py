import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_users_age
ON users(age)
""")

cur.execute("""
CREATE INDEX IF NOT EXISTS idx_users_age_score
ON users(age, score)
""")

conn.commit()


def show_explain(title, sql):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    rows = cur.execute(
        "EXPLAIN " + sql
    ).fetchall()

    print(
        f"{'addr':>4} "
        f"{'opcode':<15} "
        f"{'p1':>4} "
        f"{'p2':>4} "
        f"{'p3':>4} "
        f"{'p4':<25} "
        f"{'p5':>3}"
    )

    print("-" * 80)

    for row in rows:
        addr, opcode, p1, p2, p3, p4, p5, comment = row

        print(
            f"{addr:4d} "
            f"{opcode:<15} "
            f"{p1:4d} "
            f"{p2:4d} "
            f"{p3:4d} "
            f"{str(p4):<25} "
            f"{p5:3d}"
        )


show_explain(
    "TABLE SCAN",
    """
    SELECT SUM(score)
    FROM users NOT INDEXED
    WHERE age = 42
    """
)


show_explain(
    "NORMAL INDEX",
    """
    SELECT SUM(score)
    FROM users INDEXED BY idx_users_age
    WHERE age = 42
    """
)


show_explain(
    "COVERING INDEX",
    """
    SELECT SUM(score)
    FROM users INDEXED BY idx_users_age_score
    WHERE age = 42
    """
)


conn.close()