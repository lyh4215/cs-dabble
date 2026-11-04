
-- Reset
DROP TABLE IF EXISTS independent;
DROP TABLE IF EXISTS correlated;

-- Dataset A: independent columns
CREATE TABLE independent AS
SELECT
    g AS id,
    g % 100 AS a,
    (g / 100) % 100 AS b
FROM generate_series(0, 999999) AS g;

-- Dataset B: perfectly correlated columns
CREATE TABLE correlated AS
SELECT
    g AS id,
    g % 100 AS a,
    g % 100 AS b
FROM generate_series(0, 999999) AS g;

ANALYZE independent;
ANALYZE correlated;

-- Make cardinality output easier to interpret
SET max_parallel_workers_per_gather = 0;
SET jit = off;

-- ==========================================
-- EXP 1: No multivariate statistics
-- ==========================================

\echo '=== INDEPENDENT ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM independent
WHERE a = 42 AND b = 42;

\echo '=== CORRELATED / MATCHING ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM correlated
WHERE a = 42 AND b = 42;

\echo '=== CORRELATED / IMPOSSIBLE ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM correlated
WHERE a = 42 AND b = 43;

-- ==========================================
-- EXP 2: Functional dependencies
-- ==========================================

CREATE STATISTICS corr_dep (dependencies)
ON a, b FROM correlated;

ANALYZE correlated;

\echo '=== DEPENDENCIES / MATCHING ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM correlated
WHERE a = 42 AND b = 42;

\echo '=== DEPENDENCIES / IMPOSSIBLE ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM correlated
WHERE a = 42 AND b = 43;

-- ==========================================
-- EXP 3: Multivariate MCV
-- ==========================================

CREATE STATISTICS corr_mcv (mcv)
ON a, b FROM correlated;

ANALYZE correlated;

\echo '=== MCV / IMPOSSIBLE ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT *
FROM correlated
WHERE a = 42 AND b = 43;
