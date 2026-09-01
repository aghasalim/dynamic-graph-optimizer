-- Recompute every row of docs/ablations.txt from the per episode metrics.
--
-- The published table is nine means over ten evaluation episodes, produced by
-- compare_policies() in dgno/baselines.py. Nothing checked the averaging: the
-- table and the figures both read the same numpy output, so a mistake there
-- would show up identically in each. This does the same reduction in SQLite,
-- over the raw per episode rows in verify/data/episode-metrics.csv, and renders
-- it at exactly the precision the Python printed. verify/verify.sh diffs the
-- two tables as text.
--
-- Run: sqlite3 -init verify/summary.sql :memory: ""

.mode csv
.headers off
.import --csv verify/data/episode-metrics.csv metrics

SELECT policy,
       printf('%.2f', avg(ret)),
       printf('%.3f', avg(served)),
       printf('%.3f', avg(dropped)),
       printf('%.3f', avg(peak_q)),
       printf('%.3f', avg(mean_q)),
       printf('%.4f', avg(churn)),
       count(*)
FROM metrics
GROUP BY policy
ORDER BY policy;
