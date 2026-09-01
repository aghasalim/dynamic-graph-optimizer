// Structural and cross file validation of everything the repository publishes,
// plus a second independent recomputation of docs/ablations.txt.
//
// The tables under docs/ are the evidence for every number in the README, and
// nothing checked them. A truncated write, a column that drifted, a NaN out of
// a division, or a table regenerated against a different reward would all be
// invisible until a reader noticed the arithmetic did not add up.
//
// Four things happen here:
//
//	1. every published data file is parsed strictly: no ragged rows, no
//	   duplicate columns, no NaN or Inf, training curves that advance
//	2. the nine rows of docs/ablations.txt are recomputed from the per episode
//	   metrics in verify/data/episode-metrics.csv
//	3. the reward independent columns must be identical in all eight
//	   evaluation tables, which is what dgno/ablations.py claims and never
//	   checked
//	4. the two runs trained without the churn penalty must have a return column
//	   that is exactly the shaping return plus the churn the penalty would have
//	   cost, and the transfer table's node and edge counts must match the grid
//	   formula they came from
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Reward weights that make the cross table derivations closed form,
// from RewardConfig in dgno/env.py and NetworkConfig.horizon.
const (
	smoothnessWeight = 0.05
	horizon          = 300.0
)

var failures []string

func fail(format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	failures = append(failures, msg)
	fmt.Println("  FAIL " + msg)
}

// ---------------------------------------------------------------- reading

type table struct {
	rows map[string][]float64 // policy label -> the six metric columns
	name string
}

var metricNames = []string{"return", "served", "dropped", "peak_q", "mean_q", "churn"}

// readFixedWidth parses one of the printed tables under docs/. A row is a label
// followed by exactly six numbers; anything else is a header or a rule.
func readFixedWidth(path string) (*table, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	t := &table{rows: map[string][]float64{}, name: filepath.Base(path)}
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 7 {
			continue
		}
		values := make([]float64, 0, 6)
		ok := true
		for _, f := range fields[len(fields)-6:] {
			v, err := strconv.ParseFloat(f, 64)
			if err != nil {
				ok = false
				break
			}
			values = append(values, v)
		}
		if !ok {
			continue
		}
		label := strings.TrimSpace(strings.Join(fields[:len(fields)-6], " "))
		if label == "" {
			continue
		}
		t.rows[label] = values
	}
	if len(t.rows) == 0 {
		return nil, fmt.Errorf("%s: no metric rows", path)
	}
	return t, nil
}

func readCSV(path string) ([]string, [][]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	r.FieldsPerRecord = 0 // a ragged file is an error, which is the point
	rows, err := r.ReadAll()
	if err != nil {
		return nil, nil, err
	}
	if len(rows) < 2 {
		return nil, nil, fmt.Errorf("%s: only %d rows", path, len(rows))
	}
	return rows[0], rows[1:], nil
}

// ------------------------------------------------------- structural checks

func checkCSV(root, rel string, minRows int) {
	path := filepath.Join(root, rel)
	header, rows, err := readCSV(path)
	if err != nil {
		fail("%s: %v", rel, err)
		return
	}
	seen := map[string]bool{}
	for _, h := range header {
		if h == "" {
			fail("%s: empty column name", rel)
		}
		if seen[h] {
			fail("%s: duplicate column %q", rel, h)
		}
		seen[h] = true
	}
	bad := 0
	for i, row := range rows {
		for j, cell := range row {
			if cell == "" {
				continue // stable-baselines3 leaves the first iteration blank
			}
			v, err := strconv.ParseFloat(cell, 64)
			if err != nil {
				continue // a label column
			}
			if math.IsNaN(v) || math.IsInf(v, 0) {
				fail("%s: row %d column %s is %s", rel, i+2, header[j], cell)
				bad++
			}
		}
	}
	if len(rows) < minRows {
		fail("%s: %d data rows, expected at least %d", rel, len(rows), minRows)
	}
	if bad == 0 {
		fmt.Printf("  ok   %-38s %4d rows, %2d columns, no NaN or Inf\n",
			rel, len(rows), len(header))
	}
}

// checkCurve requires the training log to advance and to reach the budget the
// README claims for it.
func checkCurve(root, rel string, wantSteps float64) {
	header, rows, err := readCSV(filepath.Join(root, rel))
	if err != nil {
		fail("%s: %v", rel, err)
		return
	}
	idx := -1
	for i, h := range header {
		if h == "time/total_timesteps" {
			idx = i
		}
	}
	if idx < 0 {
		fail("%s: no time/total_timesteps column", rel)
		return
	}
	last := -1.0
	for i, row := range rows {
		v, err := strconv.ParseFloat(row[idx], 64)
		if err != nil {
			fail("%s: row %d has unparseable total_timesteps %q", rel, i+2, row[idx])
			return
		}
		if v <= last {
			fail("%s: total_timesteps went from %.0f to %.0f at row %d", rel, last, v, i+2)
			return
		}
		last = v
	}
	if last < wantSteps {
		fail("%s: stops at %.0f steps, the README claims %.0f", rel, last, wantSteps)
		return
	}
	fmt.Printf("  ok   %-38s reaches %.0f steps, monotone\n", rel, last)
}

// --------------------------------------------------- recompute the ablations

func recomputeAblations(root string) {
	header, rows, err := readCSV(filepath.Join(root, "verify", "data", "episode-metrics.csv"))
	if err != nil {
		fail("episode-metrics.csv: %v", err)
		return
	}
	col := map[string]int{}
	for i, h := range header {
		col[h] = i
	}
	for _, need := range []string{"policy", "ret", "served", "dropped", "peak_q", "mean_q", "churn"} {
		if _, ok := col[need]; !ok {
			fail("episode-metrics.csv: no %s column", need)
			return
		}
	}
	keys := []string{"ret", "served", "dropped", "peak_q", "mean_q", "churn"}
	sums := map[string][]float64{}
	counts := map[string]int{}
	for i, row := range rows {
		p := row[col["policy"]]
		if _, ok := sums[p]; !ok {
			sums[p] = make([]float64, len(keys))
		}
		for k, key := range keys {
			v, err := strconv.ParseFloat(row[col[key]], 64)
			if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
				fail("episode-metrics.csv row %d: %s is %q", i+2, key, row[col[key]])
				return
			}
			sums[p][k] += v
		}
		counts[p]++
	}

	published, err := readFixedWidth(filepath.Join(root, "docs", "ablations.txt"))
	if err != nil {
		fail("%v", err)
		return
	}
	formats := []string{"%.2f", "%.3f", "%.3f", "%.3f", "%.3f", "%.4f"}
	names := make([]string, 0, len(sums))
	for p := range sums {
		names = append(names, p)
	}
	sort.Strings(names)
	matched := 0
	for _, p := range names {
		want, ok := published.rows[p]
		if !ok {
			fail("docs/ablations.txt has no row for %q", p)
			continue
		}
		if counts[p] != 10 {
			fail("%s: %d episodes in the fixture, the table says 10", p, counts[p])
		}
		agree := true
		for k := range keys {
			got := fmt.Sprintf(formats[k], sums[p][k]/float64(counts[p]))
			if got != fmt.Sprintf(formats[k], want[k]) {
				fail("%s %s: Go %s, docs/ablations.txt %s", p, metricNames[k], got,
					fmt.Sprintf(formats[k], want[k]))
				agree = false
			}
		}
		if agree {
			matched++
		}
	}
	if len(published.rows) != len(names) {
		fail("docs/ablations.txt has %d rows, the fixture has %d policies",
			len(published.rows), len(names))
	}
	fmt.Printf("  ok   %d of %d rows of docs/ablations.txt recomputed from the "+
		"per episode metrics\n", matched, len(names))
}

// ------------------------------------------------------ cross table checks

// The reward independent columns, by index into a table row.
var rewardFree = []int{1, 2, 3, 4, 5}

func crossTables(root string) {
	files := []string{
		"ablations.txt", "evaluation-absolute.txt", "evaluation-bothBC.txt",
		"evaluation-churnC.txt", "evaluation-gainB.txt", "evaluation-long4M.txt",
		"evaluation-long4M_ent.txt", "evaluation-shaping.txt",
	}
	tables := map[string]*table{}
	for _, f := range files {
		t, err := readFixedWidth(filepath.Join(root, "docs", f))
		if err != nil {
			fail("%v", err)
			return
		}
		tables[f] = t
	}

	// Every table was scored on the same seeds, so the two classical baselines
	// must be identical everywhere on the columns the reward cannot move.
	base := tables["ablations.txt"]
	agree := 0
	for _, f := range files {
		for _, policy := range []string{"shortest-path", "backpressure"} {
			got, ok := tables[f].rows[policy]
			if !ok {
				fail("%s has no %s row", f, policy)
				continue
			}
			same := true
			for _, k := range rewardFree {
				if got[k] != base.rows[policy][k] {
					fail("%s %s %s: %v, ablations.txt %v", f, policy, metricNames[k],
						got[k], base.rows[policy][k])
					same = false
				}
			}
			if same {
				agree++
			}
		}
	}
	fmt.Printf("  ok   %d of %d baseline rows agree across the eight tables on "+
		"every reward independent column\n", agree, 2*len(files))

	// The two churn ablations were trained and scored with smoothness_weight 0,
	// so their return column must be the shaping return plus the penalty that
	// was removed: 0.05 * churn * 300 steps.
	pairs := []struct{ file, ablationRow string }{
		{"evaluation-churnC.txt", "ppo 400k no churn"},
		{"evaluation-bothBC.txt", "ppo 400k both"},
	}
	for _, p := range pairs {
		for _, policy := range []string{"shortest-path", "backpressure"} {
			checkNoChurnReturn(tables[p.file], p.file, policy, base.rows[policy])
		}
		row, ok := tables[p.file].rows["ppo-gnn"]
		if !ok {
			fail("%s has no ppo-gnn row", p.file)
			continue
		}
		checkNoChurnReturnRow(p.file, "ppo-gnn", row, base.rows[p.ablationRow])
	}
}

func checkNoChurnReturn(t *table, file, policy string, shaping []float64) {
	row, ok := t.rows[policy]
	if !ok {
		return
	}
	checkNoChurnReturnRow(file, policy, row, shaping)
}

func checkNoChurnReturnRow(file, policy string, row, shaping []float64) {
	predicted := shaping[0] + smoothnessWeight*shaping[5]*horizon
	// each side is printed to 2 dp and churn to 4 dp, so the rounding of the
	// inputs alone allows a little over 0.01
	const tol = 0.015
	if math.Abs(predicted-row[0]) > tol {
		fail("%s %s return: %.2f, but shaping %.2f plus the removed churn penalty "+
			"is %.4f", file, policy, row[0], shaping[0], predicted)
		return
	}
	fmt.Printf("  ok   %-26s %-14s return %7.2f = %7.2f + 0.05*%.4f*300\n",
		file, policy, row[0], shaping[0], shaping[5])
}

// ------------------------------------------------------------- the transfer

func transferTable(root string) {
	raw, err := os.ReadFile(filepath.Join(root, "docs", "transfer.txt"))
	if err != nil {
		fail("transfer.txt: %v", err)
		return
	}
	long, err := readFixedWidth(filepath.Join(root, "docs", "evaluation-long4M.txt"))
	if err != nil {
		fail("%v", err)
		return
	}
	seen := 0
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 9 || !strings.Contains(fields[0], "x") {
			continue
		}
		var rows, cols int
		if _, err := fmt.Sscanf(fields[0], "%dx%d", &rows, &cols); err != nil {
			continue
		}
		nodes, _ := strconv.Atoi(fields[1])
		edges, _ := strconv.Atoi(fields[2])
		// GridNetwork is 4 connected and both directions of every road exist
		wantNodes := rows * cols
		wantEdges := 2 * (rows*(cols-1) + (rows-1)*cols)
		if nodes != wantNodes || edges != wantEdges {
			fail("transfer.txt %s: %d nodes %d edges, the grid formula gives %d and %d",
				fields[0], nodes, edges, wantNodes, wantEdges)
			continue
		}
		seen++
		if rows == 4 && cols == 5 {
			// the training grid, so this row is the same evaluation as
			// docs/evaluation-long4M.txt and must repeat it
			want := [][2]string{
				{"shortest-path", fields[3]}, {"backpressure", fields[4]},
				{"ppo-gnn", fields[5]},
			}
			for i, w := range want {
				v, _ := strconv.ParseFloat(w[1], 64)
				if fmt.Sprintf("%.3f", v) != fmt.Sprintf("%.3f", long.rows[w[0]][1]) {
					fail("transfer.txt 4x5 served for %s is %s, evaluation-long4M.txt says %.3f",
						w[0], w[1], long.rows[w[0]][1])
				}
				p, _ := strconv.ParseFloat(fields[6+i], 64)
				if fmt.Sprintf("%.3f", p) != fmt.Sprintf("%.3f", long.rows[w[0]][3]) {
					fail("transfer.txt 4x5 peak_q for %s is %s, evaluation-long4M.txt says %.3f",
						w[0], fields[6+i], long.rows[w[0]][3])
				}
			}
		}
	}
	if seen != 5 {
		fail("transfer.txt: parsed %d grid rows, expected 5", seen)
		return
	}
	fmt.Printf("  ok   all 5 rows of docs/transfer.txt have node and edge counts " +
		"the grid formula reproduces\n")
	fmt.Printf("  ok   the 4x5 row of docs/transfer.txt repeats " +
		"docs/evaluation-long4M.txt\n")
}

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	fmt.Println("structural validation")
	for _, f := range []string{
		"docs/curve-bothBC.csv", "docs/curve-churnC.csv", "docs/curve-gainB.csv",
		"docs/curve-long4M.csv", "docs/curve-long4M_ent.csv",
		"verify/data/episode-metrics.csv", "verify/data/draws-capacity.csv",
		"verify/data/draws-phase.csv", "verify/data/draws-step.csv",
	} {
		checkCSV(*root, f, 40)
	}
	// the two small fixtures, which have their own row counts
	checkCSV(*root, "verify/data/reward-anatomy.csv", 4)
	checkCSV(*root, "verify/data/transfer-tensors.csv", 2)
	checkCurve(*root, "docs/curve-bothBC.csv", 400000)
	checkCurve(*root, "docs/curve-churnC.csv", 400000)
	checkCurve(*root, "docs/curve-gainB.csv", 400000)
	checkCurve(*root, "docs/curve-long4M.csv", 4000000)
	checkCurve(*root, "docs/curve-long4M_ent.csv", 4000000)

	fmt.Println("\nrecomputation")
	recomputeAblations(*root)

	fmt.Println("\ncross table consistency")
	crossTables(*root)
	transferTable(*root)

	fmt.Println()
	if len(failures) > 0 {
		fmt.Printf("%d check(s) failed\n", len(failures))
		os.Exit(1)
	}
	fmt.Println("Go: every published file parses, and the tables agree with each other")
}
