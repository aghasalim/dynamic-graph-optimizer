# Independent statistics on the claim the README is built on.
#
# The README says the agent beats backpressure on four of the five metrics and
# loses the fifth. Every one of those numbers is a mean over ten evaluation
# episodes, and the repository published the means with no uncertainty at all,
# so nothing distinguished a real margin from ten episodes of luck. The
# episodes are paired: every policy sees the identical seed, demand and
# incident sequence, which is the setup a paired test wants.
#
# Two things happen here:
#
#   deterministic  the nine rows of docs/ablations.txt, recomputed in base R
#                  from verify/data/episode-metrics.csv, which must match the
#                  printed table exactly
#   inferential    a paired t interval on every margin, and the sign the README
#                  claims for it
#
# Plus the one claim the README makes about the training curve itself.
#
# base R only, so CI needs nothing beyond r-base-core.

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) > 0) args[1] else "."

failures <- 0
note <- function(...) cat(" ", ..., "\n")
bad <- function(...) {
    cat("  FAIL", ..., "\n")
    failures <<- failures + 1
}

metrics <- c("ret", "served", "dropped", "peak_q", "mean_q", "churn")
labels <- c("return", "served", "dropped", "peak_q", "mean_q", "churn")
digits <- c(2, 3, 3, 3, 3, 4)

episodes <- read.csv(file.path(root, "verify", "data", "episode-metrics.csv"),
                     check.names = FALSE)

# ------------------------------------------------- the published table, again

read_table <- function(path) {
    lines <- readLines(path)
    rows <- list()
    for (line in lines) {
        f <- strsplit(trimws(line), "[[:space:]]+")[[1]]
        if (length(f) < 7) next
        tail6 <- suppressWarnings(as.numeric(f[(length(f) - 5):length(f)]))
        if (any(is.na(tail6))) next
        label <- paste(f[1:(length(f) - 6)], collapse = " ")
        rows[[label]] <- tail6
    }
    rows
}

published <- read_table(file.path(root, "docs", "ablations.txt"))

cat("docs/ablations.txt recomputed in base R\n")
policies <- sort(unique(episodes$policy))
agreed <- 0
for (p in policies) {
    block <- episodes[episodes$policy == p, ]
    if (nrow(block) != 10) bad(p, "has", nrow(block), "episodes, the table says 10")
    if (is.null(published[[p]])) {
        bad("docs/ablations.txt has no row for", p)
        next
    }
    ok <- TRUE
    for (i in seq_along(metrics)) {
        got <- formatC(mean(block[[metrics[i]]]), format = "f", digits = digits[i])
        want <- formatC(published[[p]][i], format = "f", digits = digits[i])
        if (got != want) {
            bad(p, labels[i], "R", got, "table", want)
            ok <- FALSE
        }
    }
    if (ok) agreed <- agreed + 1
}
note(agreed, "of", length(policies), "rows reproduced exactly at the printed precision")

# ---------------------------------------------------- paired inference

# The README's five scored metrics and the direction it claims for each.
# TRUE means the agent is meant to come out lower than backpressure.
claims <- list(
    served  = list(lower = FALSE, beats = TRUE),
    dropped = list(lower = TRUE,  beats = TRUE),
    peak_q  = list(lower = TRUE,  beats = TRUE),
    churn   = list(lower = TRUE,  beats = TRUE),
    mean_q  = list(lower = TRUE,  beats = FALSE)
)

agent <- episodes[episodes$policy == "ppo 4M", ]
base <- episodes[episodes$policy == "backpressure", ]
agent <- agent[order(agent$episode), ]
base <- base[order(base$episode), ]
if (!identical(agent$episode, base$episode)) {
    bad("the two policies were not scored on the same episode ids")
}

cat("\nppo 4M against backpressure, ten paired episodes\n")
cat(sprintf("  %-8s %10s %10s %10s %22s %9s %s\n",
            "metric", "ppo 4M", "backpr.", "margin", "95% CI on the margin",
            "p", "claim"))
for (m in names(claims)) {
    a <- agent[[m]]
    b <- base[[m]]
    d <- a - b
    if (sd(d) == 0) {
        lo <- hi <- 0
        p <- 1
    } else {
        tt <- t.test(d)
        lo <- tt$conf.int[1]
        hi <- tt$conf.int[2]
        p <- tt$p.value
    }
    better <- if (claims[[m]]$lower) mean(d) < 0 else mean(d) > 0
    verdict <- if (better == claims[[m]]$beats) "held" else "BROKEN"
    cat(sprintf("  %-8s %10.4f %10.4f %+10.4f  [%+9.4f, %+9.4f] %9.2e %s\n",
                m, mean(a), mean(b), mean(d), lo, hi, p, verdict))
    if (verdict != "held") {
        bad("the README says the agent", if (claims[[m]]$beats) "beats" else "loses to",
            "backpressure on", m, "and the episode data says otherwise")
    }
}

# ------------------------------------------------- the learning curve claim

cat("\nthe learning curve claim in docs/curve-long4M.csv\n")
curve <- read.csv(file.path(root, "docs", "curve-long4M.csv"), check.names = FALSE)
reward <- curve[["rollout/ep_rew_mean"]]
steps <- curve[["time/total_timesteps"]]
keep <- !is.na(reward)
reward <- reward[keep]
steps <- steps[keep]

first <- reward[1]
below <- reward < first
fraction <- mean(below[-1])
recovered <- steps[max(which(below)) + 1]
note(sprintf("first reading %.2f at %d steps, final %.2f at %d steps",
             first, steps[1], reward[length(reward)], steps[length(steps)]))
note(sprintf("%.1f%% of the %d later readings sit below that first one",
             100 * fraction, length(reward) - 1))
note(sprintf("it climbs above it for good at %d steps, %.0f%% of the way in",
             recovered, 100 * recovered / max(steps)))
if (steps[1] != 4096) {
    bad("the README says the first reading is at 4k steps, the file says", steps[1])
}
if (fraction <= 0.10) {
    bad(sprintf("only %.1f%% of the curve is below its first reading, which is not "
                , 100 * fraction), "the long stretch the README describes")
}
if (reward[length(reward)] <= first) {
    bad("the curve never climbs back past its first reading")
}

cat("\n")
if (failures > 0) {
    cat(sprintf("R: %d check(s) failed\n", failures))
    quit(status = 1)
}
cat("R: the published table reproduces, and every margin the README claims holds\n")
