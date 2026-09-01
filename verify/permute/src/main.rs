//! The uncertainty nobody measured, computed exhaustively.
//!
//! Every margin in the README is a difference of two ten episode means, printed
//! without an interval. Ten paired episodes is small enough that the exact
//! randomisation test is not an approximation at all: under the null that the
//! two policies are exchangeable on a given seed, each of the ten paired
//! differences may independently flip sign, and there are only 2^10 = 1024 ways
//! for that to happen. So this enumerates all of them rather than sampling, and
//! the p values below are exact, not Monte Carlo.
//!
//! It then runs a 200,000 draw paired bootstrap for a percentile interval, which
//! is the part Python never did and which is why this is in Rust rather than
//! bolted onto verify.R.
//!
//! Run: cargo run --release -- <repo root>

use std::collections::HashMap;
use std::env;
use std::fs;
use std::process;

const EPISODES: usize = 10;
const ASSIGNMENTS: u32 = 1 << EPISODES;
const BOOTSTRAP: usize = 200_000;
const METRICS: [&str; 5] = ["served", "dropped", "peak_q", "mean_q", "churn"];
/// The README scores five metrics and claims to beat backpressure on four.
/// `true` means lower is better for that column.
const LOWER_IS_BETTER: [bool; 5] = [false, true, true, true, true];
const CLAIMED_WIN: [bool; 5] = [true, true, true, false, true];
/// Five tests, so a Bonferroni corrected 0.05.
const ALPHA: f64 = 0.01;

/// xorshift64*, so the bootstrap is reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> usize {
        (self.next_u64() % n) as usize
    }
}

fn read_metrics(root: &str) -> HashMap<String, HashMap<String, Vec<f64>>> {
    let path = format!("{root}/verify/data/episode-metrics.csv");
    let text = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("empty file").split(',').collect();
    let mut out: HashMap<String, HashMap<String, Vec<f64>>> = HashMap::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        assert_eq!(fields.len(), header.len(), "ragged row: {line}");
        let policy = fields[0].to_string();
        let entry = out.entry(policy).or_default();
        for (i, name) in header.iter().enumerate().skip(2) {
            let v: f64 = fields[i].parse().unwrap_or_else(|_| {
                panic!("unparseable {name} in {line}");
            });
            assert!(v.is_finite(), "non finite {name} in {line}");
            entry.entry(name.to_string()).or_default().push(v);
        }
    }
    out
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Exact two sided p value over every sign assignment of the paired differences.
fn exact_p(diffs: &[f64]) -> (f64, u32) {
    let observed = mean(diffs).abs();
    let mut extreme = 0u32;
    for bits in 0..ASSIGNMENTS {
        let mut total = 0.0;
        for (i, d) in diffs.iter().enumerate() {
            total += if bits >> i & 1 == 1 { -*d } else { *d };
        }
        // a tolerance, because flipping every sign reproduces the observed
        // statistic in floating point but not always to the last bit
        if (total / diffs.len() as f64).abs() >= observed - 1e-12 {
            extreme += 1;
        }
    }
    (extreme as f64 / ASSIGNMENTS as f64, extreme)
}

/// Percentile interval from a paired bootstrap over the episodes.
fn bootstrap_ci(diffs: &[f64], rng: &mut Rng) -> (f64, f64) {
    let n = diffs.len() as u64;
    let mut draws = Vec::with_capacity(BOOTSTRAP);
    for _ in 0..BOOTSTRAP {
        let mut total = 0.0;
        for _ in 0..n {
            total += diffs[rng.below(n)];
        }
        draws.push(total / n as f64);
    }
    draws.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (
        draws[(0.025 * BOOTSTRAP as f64) as usize],
        draws[(0.975 * BOOTSTRAP as f64) as usize],
    )
}

fn main() {
    let root = env::args().nth(1).unwrap_or_else(|| ".".to_string());
    let data = read_metrics(&root);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut failures = 0;

    let base = data.get("backpressure").expect("no backpressure rows");
    let agent = data.get("ppo 4M").expect("no ppo 4M rows");

    println!(
        "exhaustive paired sign flip test, all {ASSIGNMENTS} assignments of {EPISODES} episodes"
    );
    println!("ppo 4M against backpressure, with a {BOOTSTRAP} draw paired bootstrap\n");
    println!(
        "  {:<8} {:>10} {:>9} {:>7} {:>24}  {}",
        "metric", "margin", "exact p", "of 1024", "95% bootstrap CI", "verdict"
    );

    for (k, m) in METRICS.iter().enumerate() {
        let a = &agent[*m];
        let b = &base[*m];
        assert_eq!(a.len(), EPISODES, "{m}: {} episodes", a.len());
        assert_eq!(b.len(), EPISODES);
        let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
        let margin = mean(&diffs);
        let (p, extreme) = exact_p(&diffs);
        let (lo, hi) = bootstrap_ci(&diffs, &mut rng);

        let better = if LOWER_IS_BETTER[k] { margin < 0.0 } else { margin > 0.0 };
        let excludes_zero = lo > 0.0 || hi < 0.0;
        let mut verdict = String::new();
        if better != CLAIMED_WIN[k] {
            verdict.push_str("BROKEN sign");
            failures += 1;
        } else if CLAIMED_WIN[k] {
            if p <= ALPHA && excludes_zero {
                verdict.push_str("win, real");
            } else {
                verdict.push_str("BROKEN, claimed win is not separable");
                failures += 1;
            }
        } else if excludes_zero {
            verdict.push_str("loss, real");
        } else {
            verdict.push_str("loss, not separable from zero");
        }

        println!(
            "  {:<8} {:>+10.4} {:>9.2e} {:>7} [{:>+10.4}, {:>+10.4}]  {}",
            m, margin, p, extreme, lo, hi, verdict
        );
        if !(lo..=hi).contains(&margin) {
            println!("  FAIL {m}: the bootstrap interval does not contain the point margin");
            failures += 1;
        }
    }

    // Which of the ablation agents actually separate from backpressure at all.
    println!("\nserved margin over backpressure, every committed agent, exact p");
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        if name == "backpressure" || name == "shortest-path" {
            continue;
        }
        let diffs: Vec<f64> = data[name]["served"]
            .iter()
            .zip(&base["served"])
            .map(|(x, y)| x - y)
            .collect();
        let (p, _) = exact_p(&diffs);
        println!(
            "  {:<22} {:>+9.4}  p {:>8.2e}  {}",
            name,
            mean(&diffs),
            p,
            if p <= 0.05 { "separates" } else { "does not separate" }
        );
    }

    println!();
    if failures > 0 {
        println!("Rust: {failures} check(s) failed");
        process::exit(1);
    }
    println!("Rust: every margin the README claims as a win is exact test significant");
}
