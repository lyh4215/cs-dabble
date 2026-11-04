
-- EXP 2: Cardinality Estimation → Access Path

DROP STATISTICS IF EXISTS corr_dep, corr_mcv;
DROP INDEX IF EXISTS corr_ab_idx;

CREATE INDEX corr_ab_idx ON correlated(a, b);
ANALYZE correlated;

SET max_parallel_workers_per_gather = 0;
SET jit = off;

-- A. Baseline: independent estimation
\echo '=== A: BASELINE ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT * FROM correlated
WHERE a = 42 AND b = 42;

-- B. Dependencies only
CREATE STATISTICS corr_dep (dependencies)
ON a, b FROM correlated;
ANALYZE correlated;

\echo '=== B: DEPENDENCIES ONLY ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT * FROM correlated
WHERE a = 42 AND b = 42;

-- C. MCV only: remove dependencies
DROP STATISTICS corr_dep;

CREATE STATISTICS corr_mcv (mcv)
ON a, b FROM correlated;
ANALYZE correlated;

\echo '=== C: MCV ONLY ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT * FROM correlated
WHERE a = 42 AND b = 42;

-- D. Impossible combination, MCV only
\echo '=== D: MCV IMPOSSIBLE ==='
EXPLAIN (ANALYZE, BUFFERS, TIMING OFF)
SELECT * FROM correlated
WHERE a = 42 AND b = 43;
