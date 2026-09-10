import sqlite3
import statistics
import time
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dist" / "join.db"

DB_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

if DB_PATH.exists():
    DB_PATH.unlink()


NUM_CUSTOMERS = 50_000
ORDERS_PER_CUSTOMER = 5
ITEMS_PER_ORDER = 2

RARE_CUSTOMERS = 50

REPEAT = 20


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


# 실험용 DB 생성 속도를 위한 설정.
# disposable benchmark DB라서 사용하는 것.
cur.execute("PRAGMA journal_mode = OFF")
cur.execute("PRAGMA synchronous = OFF")


# --------------------------------------------------
# 1. schema
# --------------------------------------------------

cur.executescript("""
CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    region TEXT NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL
);

CREATE TABLE items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER NOT NULL,
    kind TEXT NOT NULL
);
""")


# --------------------------------------------------
# 2. customers
#
# rare   = 50
# common = 49,950
#
# 극단적으로 skewed distribution
# --------------------------------------------------

cur.executemany(
    """
    INSERT INTO customers(id, region)
    VALUES (?, ?)
    """,
    (
        (
            customer_id,
            "rare"
            if customer_id <= RARE_CUSTOMERS
            else "common",
        )
        for customer_id
        in range(1, NUM_CUSTOMERS + 1)
    ),
)


# --------------------------------------------------
# 3. orders
#
# customer당 5개
# --------------------------------------------------

cur.executemany(
    """
    INSERT INTO orders(id, customer_id)
    VALUES (?, ?)
    """,
    (
        (
            (customer_id - 1)
            * ORDERS_PER_CUSTOMER
            + j
            + 1,
            customer_id,
        )
        for customer_id
        in range(1, NUM_CUSTOMERS + 1)
        for j
        in range(ORDERS_PER_CUSTOMER)
    ),
)


NUM_ORDERS = (
    NUM_CUSTOMERS
    * ORDERS_PER_CUSTOMER
)


# --------------------------------------------------
# 4. items
#
# order당 2개
#
# kind는 k0 ~ k99 균등분포
# --------------------------------------------------

cur.executemany(
    """
    INSERT INTO items(id, order_id, kind)
    VALUES (?, ?, ?)
    """,
    (
        (
            (order_id - 1)
            * ITEMS_PER_ORDER
            + j
            + 1,

            order_id,

            "k" + str(
                (
                    (order_id - 1)
                    * ITEMS_PER_ORDER
                    + j
                    + 1
                )
                % 100
            ),
        )
        for order_id
        in range(1, NUM_ORDERS + 1)
        for j
        in range(ITEMS_PER_ORDER)
    ),
)

conn.commit()


# --------------------------------------------------
# 5. indexes
# --------------------------------------------------

cur.executescript("""
CREATE INDEX idx_customers_region
ON customers(region);

CREATE INDEX idx_orders_customer
ON orders(customer_id);

CREATE INDEX idx_items_order
ON items(order_id);

CREATE INDEX idx_items_kind
ON items(kind);
""")

conn.commit()


# statistics 생성
cur.execute("ANALYZE")
conn.commit()


# --------------------------------------------------
# 6. 실제 cardinality
# --------------------------------------------------

actual_rare = cur.execute("""
    SELECT COUNT(*)
    FROM customers
    WHERE region = 'rare'
""").fetchone()[0]

actual_k1 = cur.execute("""
    SELECT COUNT(*)
    FROM items
    WHERE kind = 'k1'
""").fetchone()[0]


print("=== ACTUAL CARDINALITY ===")

print(
    f"region='rare': {actual_rare}"
)

print(
    f"kind='k1'    : {actual_k1}"
)


# --------------------------------------------------
# 7. planner가 보는 statistics
# --------------------------------------------------

print()
print("=== sqlite_stat1 ===")

stats = cur.execute("""
    SELECT tbl, idx, stat
    FROM sqlite_stat1
    WHERE tbl IN (
        'customers',
        'orders',
        'items'
    )
    ORDER BY tbl, idx
""").fetchall()

for row in stats:
    print(row)


# --------------------------------------------------
# 8. 같은 결과를 만드는 3가지 query
# --------------------------------------------------

#
# AUTO
#
# SQLite가 join order를 자유롭게 선택
#

auto_query = """
SELECT COUNT(*)
FROM customers AS c
JOIN orders AS o
    ON o.customer_id = c.id
JOIN items AS i
    ON i.order_id = o.id
WHERE c.region = 'rare'
  AND i.kind = 'k1'
"""


#
# GOOD
#
# customers
#   → orders
#   → items
#
# CROSS JOIN으로 loop order 강제
#

good_query = """
SELECT COUNT(*)
FROM customers AS c
CROSS JOIN orders AS o
CROSS JOIN items AS i
WHERE c.region = 'rare'
  AND i.kind = 'k1'
  AND o.customer_id = c.id
  AND i.order_id = o.id
"""


#
# BAD
#
# items
#   → orders
#   → customers
#
# 일부러 반대로 강제
#

bad_query = """
SELECT COUNT(*)
FROM items AS i
CROSS JOIN orders AS o
CROSS JOIN customers AS c
WHERE c.region = 'rare'
  AND i.kind = 'k1'
  AND i.order_id = o.id
  AND o.customer_id = c.id
"""


queries = {
    "AUTO": auto_query,
    "GOOD": good_query,
    "BAD": bad_query,
}


# --------------------------------------------------
# 9. EXPLAIN
# --------------------------------------------------

print()
print("=== QUERY PLANS ===")

for name, sql in queries.items():

    print()
    print(name)

    plan = cur.execute(
        "EXPLAIN QUERY PLAN " + sql
    ).fetchall()

    for row in plan:
        print(row)


# --------------------------------------------------
# 10. benchmark
# --------------------------------------------------

# warm-up
for sql in queries.values():
    for _ in range(5):
        cur.execute(sql).fetchone()


times = {
    name: []
    for name in queries
}

results = {}


for _ in range(REPEAT):

    order = list(queries.keys())
    random.shuffle(order)

    for name in order:

        sql = queries[name]

        start = time.perf_counter_ns()

        result = cur.execute(
            sql
        ).fetchone()[0]

        elapsed = (
            time.perf_counter_ns()
            - start
        )

        times[name].append(
            elapsed / 1_000_000
        )

        results[name] = result


print()
print("=== BENCHMARK ===")

for name in (
    "AUTO",
    "GOOD",
    "BAD",
):

    values = times[name]

    print(
        f"{name:5s}"
        f" result={results[name]:4d}"
        f" median={statistics.median(values):8.3f} ms"
        f" mean={statistics.mean(values):8.3f} ms"
        f" min={min(values):8.3f} ms"
        f" max={max(values):8.3f} ms"
    )


conn.close()