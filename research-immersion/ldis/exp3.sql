
-- EXP 3: Estimation Error → Join Strategy

DROP TABLE IF EXISTS lookup;

CREATE TABLE lookup AS
SELECT
    g AS id,
    g % 251 AS payload
FROM generate_series(0, 999999) AS g;

CREATE UNIQUE INDEX lookup_id_idx
ON lookup(id);

ANALYZE lookup;

SET max_parallel_workers_per_gather = 0;
SET jit = off;
SET work_mem = '64MB';

-- A. Baseline
DROP STATISTICS IF EXISTS corr_mcv, corr_dep;
ANALYZE correlated;

\echo '=== A: BASELINE JOIN ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT SUM(l.payload)
FROM correlated AS c
JOIN lookup AS l ON c.id = l.id
WHERE c.a = 42 AND c.b = 42;

-- B. Functional Dependencies
CREATE STATISTICS corr_dep (dependencies)
ON a, b FROM correlated;

ANALYZE correlated;

\echo '=== B: DEPENDENCIES JOIN ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT SUM(l.payload)
FROM correlated AS c
JOIN lookup AS l ON c.id = l.id
WHERE c.a = 42 AND c.b = 42;

-- C. MCV only
DROP STATISTICS corr_dep;

CREATE STATISTICS corr_mcv (mcv)
ON a, b FROM correlated;

ANALYZE correlated;

\echo '=== C: MCV JOIN ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT SUM(l.payload)
FROM correlated AS c
JOIN lookup AS l ON c.id = l.id
WHERE c.a = 42 AND c.b = 42;
