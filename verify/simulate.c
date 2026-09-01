/* An independent implementation of the queueing simulator and both classical
 * baselines, in C, checked against the numbers docs/ablations.txt publishes.
 *
 * Every figure in the README comes out of dgno/simulator.py by way of
 * dgno/baselines.py. The tests in tests/ check invariants of that code against
 * itself: that shaping telescopes, that batching does not change an answer.
 * None of them could catch the dynamics being wrong in a way the tests share,
 * because there was only ever one implementation of the dynamics.
 *
 * So this rebuilds the grid, the BFS routing prior, the segmented softmax, the
 * spillback rationing, the reward and both hand written policies from the
 * README's description, and requires the result to land on the published table.
 * numpy's generator is deliberately not reimplemented: the random draws are
 * replayed from verify/data/draws-*.csv, because reproducing PCG64 in C would
 * test numpy rather than this repository's queueing model.
 *
 * Build: cc -std=c99 -O2 -Wall -Wextra -Wpedantic -Werror -o simulate simulate.c -lm
 * Run:   ./simulate <repo root>
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ROWS 4
#define COLS 5
#define MAX_NODES (ROWS * COLS)
#define MAX_EDGES (2 * (ROWS * (COLS - 1) + (ROWS - 1) * COLS))
#define MAX_SOURCES ROWS
#define EPISODES 10
/* plus one more: the reward-anatomy episode on its own seed */
#define REPLAY_EPISODES (EPISODES + 1)
#define ANATOMY_EPISODE EPISODES
#define HORIZON 300

/* NetworkConfig defaults, dgno/simulator.py */
static const double QUEUE_CAPACITY = 40.0;
static const double EDGE_CAPACITY = 8.0;
static const double BASE_DEMAND = 7.0;
static const int DEMAND_PERIOD = 120;
static const double DEMAND_AMPLITUDE = 0.8;
static const double DEMAND_NOISE = 0.15;
static const double INCIDENT_SEVERITY = 0.15;
static const double SHORTEST_PATH_BIAS = 2.5;
static const double ACTION_GAIN = 3.0;

/* RewardConfig defaults, dgno/env.py */
static const double DROP_WEIGHT = 0.5;
static const double SMOOTHNESS_WEIGHT = 0.05;
static const double BOTTLENECK_TEMPERATURE = 0.25;
static const double GAMMA = 0.99;
static const double OBS_CLIP = 10.0;

static const double BACKPRESSURE_GAIN = 4.0;

/* -std=c99 does not declare M_PI, and this has to be the same double numpy
 * uses, which is the nearest double to pi. */
static const double PI = 3.14159265358979323846;

/* ---------------------------------------------------------------- topology */

typedef struct {
    int num_nodes, num_edges, num_sources;
    int src[MAX_EDGES], dst[MAX_EDGES], reverse[MAX_EDGES];
    int out_start[MAX_NODES], out_count[MAX_NODES];
    int is_source[MAX_NODES], is_sink[MAX_NODES];
    int source_ids[MAX_SOURCES], sink_ids[ROWS];
    int num_sinks;
    double hops[MAX_NODES];
} Net;

static void build_net(Net *n)
{
    int r, c, k, e, head, tail;
    int queue[MAX_NODES];

    n->num_nodes = ROWS * COLS;
    n->num_edges = 0;
    for (r = 0; r < ROWS; r++) {
        for (c = 0; c < COLS; c++) {
            static const int dr[4] = {-1, 1, 0, 0};
            static const int dc[4] = {0, 0, -1, 1};
            int node = r * COLS + c;
            n->out_start[node] = n->num_edges;
            for (k = 0; k < 4; k++) {
                int nr = r + dr[k], nc = c + dc[k];
                if (nr < 0 || nr >= ROWS || nc < 0 || nc >= COLS) continue;
                n->src[n->num_edges] = node;
                n->dst[n->num_edges] = nr * COLS + nc;
                n->num_edges++;
            }
            n->out_count[node] = n->num_edges - n->out_start[node];
        }
    }

    /* roads are two way, so every edge has an opposite */
    for (e = 0; e < n->num_edges; e++) {
        int f;
        n->reverse[e] = -1;
        for (f = 0; f < n->num_edges; f++) {
            if (n->src[f] == n->dst[e] && n->dst[f] == n->src[e]) {
                n->reverse[e] = f;
                break;
            }
        }
    }

    n->num_sources = n->num_sinks = 0;
    for (r = 0; r < n->num_nodes; r++) n->is_source[r] = n->is_sink[r] = 0;
    for (r = 0; r < ROWS; r++) {
        n->is_source[r * COLS] = 1;
        n->source_ids[n->num_sources++] = r * COLS;
        n->is_sink[r * COLS + COLS - 1] = 1;
        n->sink_ids[n->num_sinks++] = r * COLS + COLS - 1;
    }

    /* hop distance to the nearest sink: the shortest path routing prior */
    for (r = 0; r < n->num_nodes; r++) n->hops[r] = INFINITY;
    head = tail = 0;
    for (r = 0; r < n->num_sinks; r++) {
        n->hops[n->sink_ids[r]] = 0.0;
        queue[tail++] = n->sink_ids[r];
    }
    while (head < tail) {
        int node = queue[head++];
        for (e = n->out_start[node]; e < n->out_start[node] + n->out_count[node]; e++) {
            int nxt = n->dst[e];
            if (n->hops[nxt] > n->hops[node] + 1.0) {
                n->hops[nxt] = n->hops[node] + 1.0;
                queue[tail++] = nxt;
            }
        }
    }
}

/* -------------------------------------------------------------- csv inputs */

static double g_cap[REPLAY_EPISODES][MAX_EDGES];
static double g_phase[REPLAY_EPISODES][MAX_SOURCES];
static int g_inc_edge[REPLAY_EPISODES][HORIZON];
static int g_inc_dur[REPLAY_EPISODES][HORIZON];
static double g_z[REPLAY_EPISODES][HORIZON][MAX_SOURCES];

static FILE *open_or_die(const char *root, const char *rel)
{
    char path[1024];
    FILE *f;
    snprintf(path, sizeof path, "%s/%s", root, rel);
    f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    return f;
}

static void skip_header(FILE *f)
{
    int ch;
    while ((ch = fgetc(f)) != EOF && ch != '\n') { }
}

static void load_draws(const char *root, const Net *net)
{
    FILE *f;
    int ep, idx, s, t, edge, dur, got;
    double value;
    char line[4096], *p;

    f = open_or_die(root, "verify/data/draws-capacity.csv");
    skip_header(f);
    while (fscanf(f, "%d,%d,%lf", &ep, &idx, &value) == 3)
        g_cap[ep][idx] = value;
    fclose(f);

    f = open_or_die(root, "verify/data/draws-phase.csv");
    skip_header(f);
    while (fscanf(f, "%d,%d,%lf", &ep, &idx, &value) == 3)
        g_phase[ep][idx] = value;
    fclose(f);

    f = open_or_die(root, "verify/data/draws-step.csv");
    skip_header(f);
    got = 0;
    while (fgets(line, sizeof line, f)) {
        if (sscanf(line, "%d,%d,%d,%d", &ep, &t, &edge, &dur) != 4) continue;
        g_inc_edge[ep][t] = edge;
        g_inc_dur[ep][t] = dur;
        p = line;
        for (s = 0; s < 4; s++) {
            p = strchr(p, ',');
            if (!p) { fprintf(stderr, "short draws-step row\n"); exit(1); }
            p++;
        }
        for (s = 0; s < net->num_sources; s++) {
            g_z[ep][t][s] = strtod(p, &p);
            if (*p == ',') p++;
        }
        got++;
    }
    fclose(f);
    if (got != REPLAY_EPISODES * HORIZON) {
        fprintf(stderr, "draws-step.csv has %d rows, expected %d\n",
                got, REPLAY_EPISODES * HORIZON);
        exit(1);
    }
}

/* ------------------------------------------------------------- the dynamics */

typedef struct {
    double ret, served, dropped, peak_q, mean_q, churn;
    /* the four reward terms, cumulative over the episode, for the anatomy check.
     * Kept after the six so a Metrics can still be walked as six doubles. */
    double term_throughput, term_dropped, term_congestion, term_churn;
} Metrics;

static double smooth_max(const double *values, int n, double temperature)
{
    int i;
    double peak = -INFINITY, total = 0.0;
    for (i = 0; i < n; i++)
        if (values[i] / temperature > peak) peak = values[i] / temperature;
    for (i = 0; i < n; i++) total += exp(values[i] / temperature - peak);
    return temperature * (peak + log(total / (double)n));
}

static double clip(double x, double lo, double hi)
{
    return x < lo ? lo : (x > hi ? hi : x);
}

/* One evaluation episode. policy 0 is shortest-path, 1 is backpressure. */
static Metrics run_episode(const Net *net, int episode, int policy)
{
    int e, i, t, s;
    double queues[MAX_NODES] = {0};
    double capacity[MAX_EDGES], mult[MAX_EDGES];
    int timer[MAX_EDGES];
    double prev_action[MAX_EDGES] = {0}, action[MAX_EDGES];
    double split[MAX_EDGES], flow[MAX_EDGES], logits[MAX_EDGES];
    double arriving[MAX_NODES], inflow[MAX_NODES], outflow[MAX_NODES];
    double qnorm[MAX_NODES];
    double reference = BASE_DEMAND * (double)net->num_sources;
    double prev_potential, total_reward = 0.0;
    double sum_offered = 0.0, sum_served = 0.0, sum_dropped = 0.0;
    double sum_peak = 0.0, sum_mean = 0.0, sum_churn = 0.0;
    double t_through = 0.0, t_drop = 0.0, t_cong = 0.0, t_churn = 0.0;
    Metrics m;

    for (e = 0; e < net->num_edges; e++) {
        capacity[e] = g_cap[episode][e];
        mult[e] = 1.0;
        timer[e] = 0;
    }
    for (i = 0; i < net->num_nodes; i++) qnorm[i] = 0.0;
    prev_potential = smooth_max(qnorm, net->num_nodes, BOTTLENECK_TEMPERATURE);

    for (t = 0; t < HORIZON; t++) {
        double offered[MAX_SOURCES], dropped = 0.0, throughput = 0.0;
        double potential, congestion, churn_step = 0.0, reward;
        double max_q, mean_q = 0.0;

        for (i = 0; i < net->num_nodes; i++) qnorm[i] = queues[i] / QUEUE_CAPACITY;

        /* the policy sees the observation built after the previous step */
        for (e = 0; e < net->num_edges; e++) {
            if (policy == 0) {
                action[e] = 0.0;
                continue;
            } else {
                double live = capacity[e] * mult[e];
                double f0 = clip(live / EDGE_CAPACITY, -OBS_CLIP, OBS_CLIP);
                double f4 = clip(qnorm[net->src[e]] - qnorm[net->dst[e]],
                                 -OBS_CLIP, OBS_CLIP);
                /* the observation is float32, and the policy computes in it */
                float cap32 = (float)f0, press32 = (float)f4;
                float a32 = (float)((float)((float)BACKPRESSURE_GAIN * press32) * cap32);
                if (a32 > 1.0f) a32 = 1.0f;
                if (a32 < -1.0f) a32 = -1.0f;
                action[e] = (double)a32;
            }
        }
        for (e = 0; e < net->num_edges; e++) action[e] = clip(action[e], -1.0, 1.0);

        /* incidents: age the active ones, then maybe open a new pair */
        for (e = 0; e < net->num_edges; e++) {
            if (timer[e] > 0) {
                timer[e] -= 1;
                if (timer[e] == 0) mult[e] = 1.0;
            }
        }
        if (g_inc_edge[episode][t] >= 0) {
            int a = g_inc_edge[episode][t], b = net->reverse[a];
            mult[a] = mult[b] = INCIDENT_SEVERITY;
            timer[a] = timer[b] = g_inc_dur[episode][t];
        }

        /* demand: a per source rush hour cycle plus noise */
        for (s = 0; s < net->num_sources; s++) {
            double phase = 2.0 * PI * (double)t / (double)DEMAND_PERIOD
                           + g_phase[episode][s];
            double seasonal = 1.0 + DEMAND_AMPLITUDE * sin(phase);
            double noise = 1.0 + DEMAND_NOISE * g_z[episode][t][s];
            offered[s] = BASE_DEMAND * seasonal * noise;
            if (offered[s] < 0.0) offered[s] = 0.0;
        }

        /* inject at the sources, dropping whatever will not fit */
        for (s = 0; s < net->num_sources; s++) {
            int node = net->source_ids[s];
            double room = QUEUE_CAPACITY - queues[node];
            double accepted;
            if (room < 0.0) room = 0.0;
            accepted = offered[s] < room ? offered[s] : room;
            queues[node] += accepted;
            dropped += offered[s] - accepted;
            sum_offered += offered[s];
        }

        /* segmented softmax over each node's out edges */
        for (e = 0; e < net->num_edges; e++)
            logits[e] = -SHORTEST_PATH_BIAS * net->hops[net->dst[e]]
                        + ACTION_GAIN * action[e];
        for (i = 0; i < net->num_nodes; i++) {
            int start = net->out_start[i], count = net->out_count[i];
            double peak = -INFINITY, total = 0.0;
            for (e = start; e < start + count; e++)
                if (logits[e] > peak) peak = logits[e];
            for (e = start; e < start + count; e++) {
                split[e] = exp(logits[e] - peak);
                total += split[e];
            }
            for (e = start; e < start + count; e++) split[e] /= total;
        }

        for (e = 0; e < net->num_edges; e++) {
            double live = capacity[e] * mult[e];
            double want = queues[net->src[e]] * split[e];
            flow[e] = want < live ? want : live;
            if (net->is_sink[net->src[e]]) flow[e] = 0.0;
        }

        /* spillback: ration inflow when a downstream queue has no headroom */
        for (i = 0; i < net->num_nodes; i++) arriving[i] = 0.0;
        for (e = 0; e < net->num_edges; e++) arriving[net->dst[e]] += flow[e];
        for (e = 0; e < net->num_edges; e++) {
            int d = net->dst[e];
            double scale = 1.0;
            if (arriving[d] > 0.0 && !net->is_sink[d]) {
                double headroom = QUEUE_CAPACITY - queues[d];
                double ratio = headroom / arriving[d];
                scale = ratio < 1.0 ? ratio : 1.0;
            }
            flow[e] *= scale;
        }

        for (i = 0; i < net->num_nodes; i++) inflow[i] = outflow[i] = 0.0;
        for (e = 0; e < net->num_edges; e++) {
            outflow[net->src[e]] += flow[e];
            inflow[net->dst[e]] += flow[e];
        }
        for (i = 0; i < net->num_sinks; i++) throughput += inflow[net->sink_ids[i]];

        for (i = 0; i < net->num_nodes; i++)
            queues[i] = clip(queues[i] - outflow[i] + inflow[i], 0.0, QUEUE_CAPACITY);
        for (i = 0; i < net->num_sinks; i++) queues[net->sink_ids[i]] = 0.0;

        /* reward, and the episode statistics the evaluator keeps */
        for (i = 0; i < net->num_nodes; i++) qnorm[i] = queues[i] / QUEUE_CAPACITY;
        potential = smooth_max(qnorm, net->num_nodes, BOTTLENECK_TEMPERATURE);
        congestion = GAMMA * potential - prev_potential;
        for (e = 0; e < net->num_edges; e++) {
            double d = action[e] - prev_action[e];
            churn_step += d * d;
        }
        churn_step /= (double)net->num_edges;
        reward = throughput / reference - DROP_WEIGHT * dropped / reference
                 - congestion - SMOOTHNESS_WEIGHT * churn_step;

        max_q = qnorm[0];
        for (i = 0; i < net->num_nodes; i++) {
            if (qnorm[i] > max_q) max_q = qnorm[i];
            mean_q += qnorm[i];
        }
        mean_q /= (double)net->num_nodes;

        t_through += throughput / reference;
        t_drop += -DROP_WEIGHT * dropped / reference;
        t_cong += -congestion;
        t_churn += -SMOOTHNESS_WEIGHT * churn_step;

        total_reward += reward;
        sum_served += throughput;
        sum_dropped += dropped;
        sum_peak += max_q;
        sum_mean += mean_q;
        sum_churn += churn_step;

        for (e = 0; e < net->num_edges; e++) prev_action[e] = action[e];
        prev_potential = potential;
    }

    m.ret = total_reward;
    m.served = sum_served / sum_offered;
    m.dropped = sum_dropped / sum_offered;
    m.peak_q = sum_peak / (double)HORIZON;
    m.mean_q = sum_mean / (double)HORIZON;
    m.churn = sum_churn / (double)HORIZON;
    m.term_throughput = t_through;
    m.term_dropped = t_drop;
    m.term_congestion = t_cong;
    m.term_churn = t_churn;
    return m;
}

/* --------------------------------------------------------------- reference */

static int read_fixture(const char *root, const char *policy, Metrics out[EPISODES])
{
    FILE *f = open_or_die(root, "verify/data/episode-metrics.csv");
    char line[1024];
    size_t n = strlen(policy);
    int found = 0;
    skip_header(f);
    while (fgets(line, sizeof line, f)) {
        int ep;
        Metrics m;
        if (strncmp(line, policy, n) != 0 || line[n] != ',') continue;
        if (sscanf(line + n + 1, "%d,%lf,%lf,%lf,%lf,%lf,%lf", &ep, &m.ret,
                   &m.served, &m.dropped, &m.peak_q, &m.mean_q, &m.churn) != 7)
            continue;
        if (ep < 0 || ep >= EPISODES) continue;
        out[ep] = m;
        found++;
    }
    fclose(f);
    return found;
}

/* Pull one row out of the fixed width table in docs/ablations.txt. */
static int read_published(const char *root, const char *policy, Metrics *m)
{
    FILE *f = open_or_die(root, "docs/ablations.txt");
    char line[1024];
    size_t n = strlen(policy);
    int ok = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, policy, n) != 0 || line[n] != ' ') continue;
        if (sscanf(line + n, "%lf %lf %lf %lf %lf %lf", &m->ret, &m->served,
                   &m->dropped, &m->peak_q, &m->mean_q, &m->churn) == 6)
            ok = 1;
        break;
    }
    fclose(f);
    return ok;
}

int main(int argc, char **argv)
{
    const char *root = argc > 1 ? argv[1] : ".";
    const char *names[2] = {"shortest-path", "backpressure"};
    Net net;
    int policy, ep, failures = 0;

    build_net(&net);
    printf("grid %dx%d: %d nodes, %d edges, max hops to sink %.0f\n",
           ROWS, COLS, net.num_nodes, net.num_edges, net.hops[0]);
    if (net.num_nodes != MAX_NODES || net.num_edges != MAX_EDGES) {
        printf("FAIL topology: %d nodes %d edges\n", net.num_nodes, net.num_edges);
        return 1;
    }
    load_draws(root, &net);

    for (policy = 0; policy < 2; policy++) {
        Metrics fixture[EPISODES], mine[EPISODES], mean, want;
        double worst = 0.0;
        const char *labels[6] = {"return", "served", "dropped",
                                 "peak_q", "mean_q", "churn"};
        const char *fmts[6] = {"%.2f", "%.3f", "%.3f", "%.3f", "%.3f", "%.4f"};
        double *mp, *wp;
        int k;

        if (read_fixture(root, names[policy], fixture) != EPISODES) {
            printf("FAIL: episode-metrics.csv has no complete block for %s\n",
                   names[policy]);
            failures++;
            continue;
        }
        memset(&mean, 0, sizeof mean);
        for (ep = 0; ep < EPISODES; ep++) {
            int j;
            double *a, *b;
            mine[ep] = run_episode(&net, ep, policy);
            a = (double *)&mine[ep];
            b = (double *)&fixture[ep];
            for (j = 0; j < 6; j++) {
                double scale = fabs(b[j]) > 1.0 ? fabs(b[j]) : 1.0;
                double rel = fabs(a[j] - b[j]) / scale;
                if (rel > worst) worst = rel;
                ((double *)&mean)[j] += a[j] / (double)EPISODES;
            }
        }
        printf("\n%s: %d episodes replayed from the recorded draws\n", names[policy], EPISODES);
        printf("  worst relative disagreement with verify/data/episode-metrics.csv: %.1e\n",
               worst);
        if (worst > 1e-9) {
            printf("  FAIL: above the 1e-9 tolerance\n");
            failures++;
        }

        if (!read_published(root, names[policy], &want)) {
            printf("  FAIL: no %s row in docs/ablations.txt\n", names[policy]);
            failures++;
            continue;
        }
        mp = (double *)&mean;
        wp = (double *)&want;
        for (k = 0; k < 6; k++) {
            /* the published table is text, so compare it as text, at the same
             * precision compare_policies() printed it with */
            char got[64], published[64];
            int same;
            snprintf(got, sizeof got, fmts[k], mp[k]);
            snprintf(published, sizeof published, fmts[k], wp[k]);
            same = strcmp(got, published) == 0;
            printf("  %-8s C %10s  docs/ablations.txt %10s  %s\n",
                   labels[k], got, published, same ? "ok" : "FAIL");
            if (!same) failures++;
        }
    }

    {
        /* docs/reward-anatomy.png, and the two numbers the README quotes off it:
         * an episode is dominated by throughput while the shaping term, which
         * telescopes, contributes almost nothing. */
        Metrics anatomy = run_episode(&net, ANATOMY_EPISODE, 1);
        const char *keys[4] = {"throughput", "dropped", "congestion", "churn"};
        double mine[4] = {anatomy.term_throughput, anatomy.term_dropped,
                          anatomy.term_congestion, anatomy.term_churn};
        FILE *f = open_or_die(root, "verify/data/reward-anatomy.csv");
        char line[512];
        int k, matched = 0;
        printf("\nreward anatomy, backpressure on the figure's own seed\n");
        skip_header(f);
        while (fgets(line, sizeof line, f)) {
            double want;
            char *comma = strchr(line, ',');
            if (!comma) continue;
            *comma = '\0';
            want = strtod(comma + 1, NULL);
            for (k = 0; k < 4; k++) {
                double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
                double rel;
                if (strcmp(line, keys[k]) != 0) continue;
                rel = fabs(mine[k] - want) / scale;
                printf("  %-12s C %14.6f  python %14.6f  rel %.1e  %s\n",
                       keys[k], mine[k], want, rel, rel <= 1e-9 ? "ok" : "FAIL");
                if (rel > 1e-9) failures++;
                matched++;
            }
        }
        fclose(f);
        if (matched != 4) {
            printf("  FAIL: matched %d of 4 terms in reward-anatomy.csv\n", matched);
            failures++;
        }
        printf("  throughput is %.0fx the shaping term over one 300 step episode\n",
               mine[0] / mine[2]);
    }

    printf("\n%s\n", failures == 0
           ? "C reproduces both baseline rows of docs/ablations.txt"
           : "C disagrees with the published table");
    return failures == 0 ? 0 : 1;
}
