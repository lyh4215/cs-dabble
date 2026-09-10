import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "users.db"


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# --------------------------------------------------
# 1. SQLite page 기본 정보
# --------------------------------------------------

page_size = cur.execute(
    "PRAGMA page_size"
).fetchone()[0]

page_count = cur.execute(
    "PRAGMA page_count"
).fetchone()[0]


print("=== DATABASE ===")
print(f"page size : {page_size} bytes")
print(f"page count: {page_count}")
print(
    f"db size   : "
    f"{page_size * page_count / 1024 / 1024:.2f} MiB"
)


# --------------------------------------------------
# 2. 각 table/index의 root page
# --------------------------------------------------

print()
print("=== ROOT PAGES ===")

rows = cur.execute("""
    SELECT
        type,
        name,
        rootpage
    FROM sqlite_schema
    WHERE name IN (
        'users',
        'idx_users_age',
        'idx_users_age_score'
    )
    ORDER BY name
""").fetchall()

for row in rows:
    print(row)


# --------------------------------------------------
# 3. dbstat으로 실제 B-tree page 관찰
# --------------------------------------------------

cur.execute("""
    CREATE VIRTUAL TABLE temp.stat
    USING dbstat(main)
""")


names = [
    "users",
    "idx_users_age",
    "idx_users_age_score",
]


for name in names:

    pages = cur.execute("""
        SELECT
            path,
            pageno,
            pagetype,
            ncell,
            payload,
            unused,
            pgsize
        FROM temp.stat
        WHERE name = ?
        ORDER BY path
    """, (name,)).fetchall()

    internal = sum(
        1
        for row in pages
        if row[2] == "internal"
    )

    leaf = sum(
        1
        for row in pages
        if row[2] == "leaf"
    )

    total_payload = sum(
        row[4]
        for row in pages
    )

    total_unused = sum(
        row[5]
        for row in pages
    )

    def depth(path):
        if path == "/":
            return 0

        return (
            len(
                [
                    x
                    for x in path.split("/")
                    if x
                ]
            )
        )

    max_depth = max(
        depth(row[0])
        for row in pages
    )

    print()
    print(f"=== {name} ===")

    print(f"total pages : {len(pages)}")
    print(f"internal    : {internal}")
    print(f"leaf        : {leaf}")
    print(f"depth       : {max_depth}")
    print(
        f"payload     : "
        f"{total_payload / 1024 / 1024:.2f} MiB"
    )
    print(
        f"unused      : "
        f"{total_unused / 1024:.2f} KiB"
    )

    print()
    print("first pages:")

    for row in pages[:10]:
        print(row)


conn.close()