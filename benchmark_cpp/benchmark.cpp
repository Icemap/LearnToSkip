/**
 * POC Benchmark: Decision Tree Classifier Inference vs 128-dim L2 Distance
 *
 * Purpose: Determine if C++ classifier-guided candidate pruning can achieve
 * wall-clock speedup over vanilla HNSW construction.
 *
 * The tree is exported from a sklearn DecisionTreeClassifier (depth=3, 8 leaves)
 * trained on SIFT10K trace data with temporal split.
 *
 * Compile: clang++ -O3 -std=c++17 -march=native benchmark.cpp -o benchmark
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

// ============================================================================
// Normalization parameters (from FeatureExtractor fitted on SIFT10K train)
// Feature order: candidate_degree, candidate_layer, current_layer, approx_dist,
//                candidate_rank_in_beam, beam_size, inserted_count
// ============================================================================

static constexpr double MEAN[] = {
    31.684450, 0.081240, 0.018900, 13.493865,
    633.113831, 164.095581, 3893.813477
};
static constexpr double STD[] = {
    2.245863, 0.292609, 0.138502, 7.063949,
    576.219116, 62.991272, 1931.801147
};

// Feature indices
static constexpr int F_CANDIDATE_DEGREE      = 0;
static constexpr int F_CANDIDATE_LAYER        = 1;
static constexpr int F_CURRENT_LAYER          = 2;
static constexpr int F_APPROX_DIST            = 3;
static constexpr int F_CANDIDATE_RANK_IN_BEAM = 4;
static constexpr int F_BEAM_SIZE              = 5;
static constexpr int F_INSERTED_COUNT         = 6;

// ============================================================================
// Exported Decision Tree (depth=3, 8 leaves)
//
// Tree structure (on z-score normalized features):
//   Node 0: feature[4] (candidate_rank_in_beam) <= -0.152050
//     Node 1: feature[4] <= -0.742103
//       Node 2: feature[6] (inserted_count) <= -0.904500
//         Leaf 3: skip_prob = 0.133092
//       Node 2 else:
//         Leaf 4: skip_prob = 0.241549
//     Node 1 else:
//       Node 5: feature[3] (approx_dist) <= 0.046640
//         Leaf 6: skip_prob = 0.345719
//       Node 5 else:
//         Leaf 7: skip_prob = 0.537761
//   Node 0 else:
//     Node 8: feature[4] <= 0.500480
//       Node 9: feature[3] <= 0.218221
//         Leaf 10: skip_prob = 0.559158
//       Node 9 else:
//         Leaf 11: skip_prob = 0.763484
//     Node 8 else:
//       Node 12: feature[3] <= -0.440863
//         Leaf 13: skip_prob = 0.738662
//       Node 12 else:
//         Leaf 14: skip_prob = 0.870165
// ============================================================================

// Version 1: Full predict_proba (returns skip probability)
inline double tree_predict_proba(const double* raw_features) {
    // Normalize only the 3 features actually used by the tree
    double rank_norm = (raw_features[F_CANDIDATE_RANK_IN_BEAM] - MEAN[4]) / STD[4];
    double adist_norm = (raw_features[F_APPROX_DIST] - MEAN[3]) / STD[3];
    double ins_norm = (raw_features[F_INSERTED_COUNT] - MEAN[6]) / STD[6];

    if (rank_norm <= -0.152050) {
        if (rank_norm <= -0.742103) {
            if (ins_norm <= -0.904500)
                return 0.133092;
            else
                return 0.241549;
        } else {
            if (adist_norm <= 0.046640)
                return 0.345719;
            else
                return 0.537761;
        }
    } else {
        if (rank_norm <= 0.500480) {
            if (adist_norm <= 0.218221)
                return 0.559158;
            else
                return 0.763484;
        } else {
            if (adist_norm <= -0.440863)
                return 0.738662;
            else
                return 0.870165;
        }
    }
}

// Version 2: Binary skip decision (avoids returning float, just bool)
inline bool tree_should_skip(const double* raw_features, double threshold) {
    return tree_predict_proba(raw_features) > threshold;
}

// Version 3: Ultra-fast — work directly on raw (unnormalized) features
// Pre-compute denormalized thresholds to avoid normalization at inference time
inline bool tree_should_skip_raw(
    double candidate_rank_in_beam,
    double approx_dist,
    double inserted_count,
    double threshold
) {
    // Denormalized thresholds (threshold_raw = threshold_norm * std + mean)
    // rank thresholds: -0.152050 * 576.219116 + 633.113831 = 545.455
    //                  -0.742103 * 576.219116 + 633.113831 = 205.390
    //                   0.500480 * 576.219116 + 633.113831 = 921.405
    // adist thresholds: 0.046640 * 7.063949 + 13.493865 = 13.823
    //                   0.218221 * 7.063949 + 13.493865 = 15.035
    //                  -0.440863 * 7.063949 + 13.493865 = 10.380
    // ins threshold:   -0.904500 * 1931.801147 + 3893.813477 = 2146.184

    static constexpr double RANK_T0 = -0.152050 * 576.219116 + 633.113831;   // 545.455
    static constexpr double RANK_T1 = -0.742103 * 576.219116 + 633.113831;   // 205.390
    static constexpr double RANK_T2 =  0.500480 * 576.219116 + 633.113831;   // 921.405
    static constexpr double ADIST_T0 = 0.046640 * 7.063949 + 13.493865;      // 13.823
    static constexpr double ADIST_T1 = 0.218221 * 7.063949 + 13.493865;      // 15.035
    static constexpr double ADIST_T2 = -0.440863 * 7.063949 + 13.493865;     // 10.380
    static constexpr double INS_T0 = -0.904500 * 1931.801147 + 3893.813477;  // 2146.184

    if (candidate_rank_in_beam <= RANK_T0) {
        if (candidate_rank_in_beam <= RANK_T1) {
            return (inserted_count <= INS_T0) ? (0.133092 > threshold)
                                              : (0.241549 > threshold);
        } else {
            return (approx_dist <= ADIST_T0) ? (0.345719 > threshold)
                                             : (0.537761 > threshold);
        }
    } else {
        if (candidate_rank_in_beam <= RANK_T2) {
            return (approx_dist <= ADIST_T1) ? (0.559158 > threshold)
                                             : (0.763484 > threshold);
        } else {
            return (approx_dist <= ADIST_T2) ? (0.738662 > threshold)
                                             : (0.870165 > threshold);
        }
    }
}

// ============================================================================
// L2 distance computation (128-dim)
// ============================================================================

// Naive L2 (compiler will auto-vectorize with -O3 -march=native)
inline float l2_distance_128(const float* a, const float* b) {
    float sum = 0.0f;
    for (int i = 0; i < 128; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// ============================================================================
// Approximate distance computation (random projection, target_dim=16)
// ============================================================================
static constexpr int PROJ_DIM = 16;

inline float approx_l2_projected(const float* a_proj, const float* b_proj) {
    float sum = 0.0f;
    for (int i = 0; i < PROJ_DIM; ++i) {
        float d = a_proj[i] - b_proj[i];
        sum += d * d;
    }
    return sum;
}

// ============================================================================
// Benchmark harness
// ============================================================================

using Clock = std::chrono::high_resolution_clock;

template<typename Func>
double bench_ns(Func&& fn, int n_iters) {
    // Warmup
    for (int i = 0; i < n_iters / 10; ++i) fn();

    auto t0 = Clock::now();
    for (int i = 0; i < n_iters; ++i) fn();
    auto t1 = Clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return total_ns / n_iters;
}

// Prevent compiler from optimizing away the result
template<typename T>
void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

int main() {
    constexpr int N = 10'000'000;
    constexpr int DIM = 128;
    constexpr double THRESHOLD = 0.7;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_real_distribution<double> feat_dist(0.0, 10000.0);

    // Generate random vectors (128-dim)
    alignas(64) float vec_a[DIM], vec_b[DIM];
    for (int i = 0; i < DIM; ++i) { vec_a[i] = dist(rng); vec_b[i] = dist(rng); }

    // Generate random projected vectors (16-dim)
    alignas(64) float proj_a[PROJ_DIM], proj_b[PROJ_DIM];
    for (int i = 0; i < PROJ_DIM; ++i) { proj_a[i] = dist(rng); proj_b[i] = dist(rng); }

    // Generate random features
    double features[7];
    for (int i = 0; i < 7; ++i) features[i] = feat_dist(rng);

    printf("=== C++ Classifier vs L2 Distance Benchmark ===\n");
    printf("Iterations per benchmark: %d\n\n", N);

    // --- Benchmark 1: 128-dim L2 distance ---
    volatile float l2_result;
    double l2_ns = bench_ns([&]() {
        l2_result = l2_distance_128(vec_a, vec_b);
        do_not_optimize(l2_result);
    }, N);
    printf("128-dim L2 distance:            %7.1f ns\n", l2_ns);

    // --- Benchmark 2: 16-dim approx L2 (projected) ---
    volatile float approx_result;
    double approx_ns = bench_ns([&]() {
        approx_result = approx_l2_projected(proj_a, proj_b);
        do_not_optimize(approx_result);
    }, N);
    printf("16-dim approx L2 (projected):   %7.1f ns\n", approx_ns);

    // --- Benchmark 3: Tree predict_proba (with normalization) ---
    volatile double proba_result;
    double tree_proba_ns = bench_ns([&]() {
        proba_result = tree_predict_proba(features);
        do_not_optimize(proba_result);
    }, N);
    printf("Tree predict_proba (normalize): %7.1f ns\n", tree_proba_ns);

    // --- Benchmark 4: Tree should_skip (binary, with normalization) ---
    volatile bool skip_result;
    double tree_skip_ns = bench_ns([&]() {
        skip_result = tree_should_skip(features, THRESHOLD);
        do_not_optimize(skip_result);
    }, N);
    printf("Tree should_skip (normalize):   %7.1f ns\n", tree_skip_ns);

    // --- Benchmark 5: Tree should_skip_raw (no normalization, denormalized thresholds) ---
    double rank = features[F_CANDIDATE_RANK_IN_BEAM];
    double adist = features[F_APPROX_DIST];
    double ins = features[F_INSERTED_COUNT];
    double tree_raw_ns = bench_ns([&]() {
        skip_result = tree_should_skip_raw(rank, adist, ins, THRESHOLD);
        do_not_optimize(skip_result);
    }, N);
    printf("Tree should_skip_raw (no norm): %7.1f ns\n", tree_raw_ns);

    // --- Benchmark 6: Full skip decision pipeline ---
    //   approx_dist(16-dim projected) + tree_should_skip_raw
    //   This is the real cost per candidate in C++ HNSW
    double pipeline_ns = bench_ns([&]() {
        float ad = approx_l2_projected(proj_a, proj_b);
        skip_result = tree_should_skip_raw(rank, (double)ad, ins, THRESHOLD);
        do_not_optimize(skip_result);
    }, N);
    printf("Full pipeline (approx+tree):    %7.1f ns\n", pipeline_ns);

    // --- Analysis ---
    printf("\n=== Analysis ===\n");
    printf("L2 distance cost:                   %7.1f ns\n", l2_ns);
    printf("Skip decision cost (full pipeline):  %7.1f ns\n", pipeline_ns);
    printf("Skip decision / L2 ratio:            %7.1f%%\n", pipeline_ns / l2_ns * 100.0);
    printf("\n");

    double skip_rate = 0.327;  // from SIFT10K at tau=0.7
    double saved_per_candidate = skip_rate * l2_ns;
    double cost_per_candidate = pipeline_ns;
    double net_per_candidate = saved_per_candidate - cost_per_candidate;
    printf("At skip_rate=%.1f%% (tau=0.7, SIFT10K):\n", skip_rate * 100);
    printf("  Expected L2 savings per candidate:  %7.1f ns (%.1f%% * %.1f ns)\n",
           saved_per_candidate, skip_rate * 100, l2_ns);
    printf("  Classifier cost per candidate:      %7.1f ns\n", cost_per_candidate);
    printf("  Net savings per candidate:           %7.1f ns\n", net_per_candidate);
    printf("  Net positive? %s\n", net_per_candidate > 0 ? "YES" : "NO");
    printf("\n");

    if (net_per_candidate > 0) {
        double speedup = l2_ns / (l2_ns - net_per_candidate);
        printf("  Estimated wall-clock speedup:       %.2fx\n", speedup);
        printf("  (This is the theoretical max for the distance computation portion)\n");
    } else {
        printf("  Break-even skip_rate: %.1f%%\n", cost_per_candidate / l2_ns * 100.0);
        printf("  (Need skip_rate above this to be net positive)\n");
    }

    // Additional: vary skip rates
    printf("\n=== Sensitivity: Net savings at different skip rates ===\n");
    printf("  skip_rate | L2_saved | clf_cost | net_save | wall_speedup\n");
    printf("  ----------|----------|----------|----------|-------------\n");
    for (double sr : {0.10, 0.20, 0.30, 0.40, 0.50, 0.60}) {
        double saved = sr * l2_ns;
        double net = saved - pipeline_ns;
        double spd = (net > 0) ? l2_ns / (l2_ns - net) : 0.0;
        printf("  %7.0f%%  | %6.1fns | %6.1fns | %6.1fns | %s\n",
               sr * 100, saved, pipeline_ns, net,
               net > 0 ? (std::string(std::to_string(spd).substr(0, 4) + "x")).c_str() : "  N/A");
    }

    return 0;
}
