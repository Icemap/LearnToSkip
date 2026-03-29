/**
 * Realistic Benchmark: simulates actual HNSW candidate evaluation pattern
 *
 * - Allocates 100K vectors (128-dim) to stress L1/L2 cache
 * - Random access pattern (like real HNSW neighbor traversal)
 * - Measures per-candidate cost of: L2 only vs (approx_dist + tree + conditional L2)
 *
 * Compile: clang++ -O3 -std=c++17 -march=native benchmark_realistic.cpp -o benchmark_realistic
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int DIM = 128;
static constexpr int PROJ_DIM = 16;
static constexpr int N_VECTORS = 100'000;
static constexpr int N_QUERIES = 1'000;
static constexpr int CANDIDATES_PER_QUERY = 500;  // ~ef_construction=200 with expansion
static constexpr double THRESHOLD = 0.7;

// === Denormalized tree thresholds (from SIFT10K trained tree) ===
static constexpr double RANK_T0 = 545.455;
static constexpr double RANK_T1 = 205.390;
static constexpr double RANK_T2 = 921.405;
static constexpr double ADIST_T0 = 13.823;
static constexpr double ADIST_T1 = 15.035;
static constexpr double ADIST_T2 = 10.380;
static constexpr double INS_T0 = 2146.184;

struct SkipResult {
    bool should_skip;
    double proba;
};

inline SkipResult tree_decide(double rank, double adist, double ins_count) {
    double proba;
    if (rank <= RANK_T0) {
        if (rank <= RANK_T1) {
            proba = (ins_count <= INS_T0) ? 0.133092 : 0.241549;
        } else {
            proba = (adist <= ADIST_T0) ? 0.345719 : 0.537761;
        }
    } else {
        if (rank <= RANK_T2) {
            proba = (adist <= ADIST_T1) ? 0.559158 : 0.763484;
        } else {
            proba = (adist <= ADIST_T2) ? 0.738662 : 0.870165;
        }
    }
    return {proba > THRESHOLD, proba};
}

inline float l2_128(const float* __restrict a, const float* __restrict b) {
    float sum = 0.0f;
    for (int i = 0; i < DIM; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

inline float approx_l2(const float* __restrict a, const float* __restrict b) {
    float sum = 0.0f;
    for (int i = 0; i < PROJ_DIM; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

using Clock = std::chrono::high_resolution_clock;

int main() {
    printf("=== Realistic HNSW Candidate Evaluation Benchmark ===\n");
    printf("N_VECTORS=%d, N_QUERIES=%d, CANDIDATES_PER_QUERY=%d\n\n",
           N_VECTORS, N_QUERIES, CANDIDATES_PER_QUERY);

    // Allocate data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> vdist(0.0f, 1.0f);

    // Full vectors (128-dim) — ~49MB, won't fit in L2 cache
    std::vector<float> data(N_VECTORS * DIM);
    for (auto& v : data) v = vdist(rng);

    // Projected vectors (16-dim) — ~6MB
    std::vector<float> data_proj(N_VECTORS * PROJ_DIM);
    for (auto& v : data_proj) v = vdist(rng);

    // Query vectors
    std::vector<float> queries(N_QUERIES * DIM);
    for (auto& v : queries) v = vdist(rng);
    std::vector<float> queries_proj(N_QUERIES * PROJ_DIM);
    for (auto& v : queries_proj) v = vdist(rng);

    // Random candidate indices (simulates HNSW neighbor traversal)
    std::uniform_int_distribution<int> idx_dist(0, N_VECTORS - 1);
    std::vector<int> candidates(N_QUERIES * CANDIDATES_PER_QUERY);
    for (auto& c : candidates) c = idx_dist(rng);

    // Simulated features for each candidate
    std::uniform_real_distribution<double> rank_dist(0.0, 2000.0);
    std::uniform_real_distribution<double> ins_dist(0.0, 10000.0);
    std::vector<double> ranks(N_QUERIES * CANDIDATES_PER_QUERY);
    std::vector<double> ins_counts(N_QUERIES * CANDIDATES_PER_QUERY);
    for (auto& r : ranks) r = rank_dist(rng);
    for (auto& ic : ins_counts) ic = ins_dist(rng);

    int64_t total_candidates = (int64_t)N_QUERIES * CANDIDATES_PER_QUERY;

    // ====================================================================
    // Scenario A: Vanilla — compute L2 for every candidate
    // ====================================================================
    volatile float sink_f = 0.0f;
    auto t0 = Clock::now();
    for (int q = 0; q < N_QUERIES; ++q) {
        const float* qvec = &queries[q * DIM];
        for (int c = 0; c < CANDIDATES_PER_QUERY; ++c) {
            int idx = candidates[q * CANDIDATES_PER_QUERY + c];
            float d = l2_128(qvec, &data[idx * DIM]);
            sink_f = d;
        }
    }
    auto t1 = Clock::now();
    double vanilla_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    double vanilla_per = vanilla_ns / total_candidates;

    // ====================================================================
    // Scenario B: LearnToSkip — approx_dist + tree + conditional L2
    // ====================================================================
    int64_t skipped = 0;
    int64_t computed = 0;
    auto t2 = Clock::now();
    for (int q = 0; q < N_QUERIES; ++q) {
        const float* qvec = &queries[q * DIM];
        const float* qproj = &queries_proj[q * PROJ_DIM];
        for (int c = 0; c < CANDIDATES_PER_QUERY; ++c) {
            int flat = q * CANDIDATES_PER_QUERY + c;
            int idx = candidates[flat];

            // Step 1: approx distance
            float adist = approx_l2(qproj, &data_proj[idx * PROJ_DIM]);

            // Step 2: tree decision
            auto [skip, proba] = tree_decide(ranks[flat], (double)adist, ins_counts[flat]);

            if (skip) {
                skipped++;
                continue;
            }

            // Step 3: exact L2 (only if not skipped)
            float d = l2_128(qvec, &data[idx * DIM]);
            sink_f = d;
            computed++;
        }
    }
    auto t3 = Clock::now();
    double skip_ns = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();
    double skip_per = skip_ns / total_candidates;

    double actual_skip_rate = (double)skipped / total_candidates;
    double wall_speedup = vanilla_ns / skip_ns;

    printf("--- Scenario A: Vanilla (L2 for every candidate) ---\n");
    printf("  Total time:     %10.1f ms\n", vanilla_ns / 1e6);
    printf("  Per candidate:  %10.1f ns\n", vanilla_per);
    printf("  Candidates:     %10lld\n", total_candidates);
    printf("\n");

    printf("--- Scenario B: LearnToSkip (approx + tree + conditional L2) ---\n");
    printf("  Total time:     %10.1f ms\n", skip_ns / 1e6);
    printf("  Per candidate:  %10.1f ns\n", skip_per);
    printf("  Skipped:        %10lld (%.1f%%)\n", skipped, actual_skip_rate * 100.0);
    printf("  Computed:       %10lld (%.1f%%)\n", computed, (1 - actual_skip_rate) * 100.0);
    printf("\n");

    printf("=== Results ===\n");
    printf("  Wall-clock speedup:  %.2fx\n", wall_speedup);
    printf("  Skip rate:           %.1f%%\n", actual_skip_rate * 100.0);
    printf("  Overhead per skip:   %.1f ns (approx_dist + tree)\n",
           skip_per - vanilla_per * (1 - actual_skip_rate));
    printf("\n");

    if (wall_speedup > 1.0) {
        printf("  >>> POSITIVE: C++ LearnToSkip has real wall-clock speedup! <<<\n");
    } else {
        printf("  >>> NEGATIVE: Classifier overhead exceeds savings <<<\n");
    }

    // ====================================================================
    // Now simulate with the actual SIFT10K skip rate (32.7% at tau=0.7)
    // ====================================================================
    printf("\n=== Simulation with SIFT10K skip rates ===\n");
    for (double sr : {0.10, 0.20, 0.327, 0.40, 0.50}) {
        // Approx overhead per candidate (approx_dist + tree)
        double overhead = skip_per - vanilla_per * (1 - actual_skip_rate);
        double effective_per = vanilla_per * (1 - sr) + overhead;
        double spd = vanilla_per / effective_per;
        printf("  skip_rate=%.1f%%: estimated speedup=%.2fx\n", sr * 100, spd);
    }

    return 0;
}
