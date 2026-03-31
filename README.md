# LearnToSkip

Lightweight classifier-guided candidate pruning for HNSW index construction.

During HNSW construction, the beam search evaluates far more candidates than it retains as neighbors. On SIFT1M with standard parameters (M=16, ef\_construction=200), **99.2% of candidate distance evaluations are wasted**. LearnToSkip trains a depth-3 decision tree to predict which candidates will be discarded *before* computing their exact distances, achieving up to **3.3x single-threaded** and **3.6x multi-threaded** (4 threads) wall-clock speedup with sub-1.1% recall@10 loss.

## Repository Structure

```
src/learn_to_skip/          # Core library
  builders/                 # Index construction strategies (vanilla, learned skip, C++ skip)
  classifiers/              # Classifier wrappers (decision tree, logistic, SVM, XGBoost)
  experiments/              # Reproducible experiment scripts
  features/                 # Feature extraction (random projection, encounter rank)
  datasets/                 # Dataset registry and loaders
  tracer/                   # HNSW construction tracing (records all candidate evaluations)
  orchestrator/             # CLI runner for experiments and plots
  adaptive/                 # Adaptive ef_construction
  visualization/            # Plotting utilities
hnswlib_fork/               # Forked hnswlib with skip injection (git submodule)
benchmark_cpp/              # C++ micro-benchmarks (tree inference vs L2 distance)
results/                    # Experiment output JSONs (checked in for reproducibility)
paper/                      # LaTeX paper source
tests/                      # Unit tests
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone with submodule
git clone --recurse-submodules git@github.com:Icemap/LearnToSkip.git
cd LearnToSkip

# Install all dependencies (including the forked hnswlib)
uv sync
```

The forked hnswlib is declared as a path dependency in `pyproject.toml` and installed automatically by `uv sync`. To force-reinstall it after modifying C++ code:

```bash
uv cache clean hnswlib && uv sync --reinstall-package hnswlib
```

## Data

Datasets are downloaded automatically on first use and stored under `data/` (gitignored).

| Dataset | Vectors | Dimensions | Distance | Source |
|---------|---------|------------|----------|--------|
| SIFT1M | 1,000,000 | 128 | L2 | [IRISA TEXMEX](http://corpus-texmex.irisa.fr/) |
| Deep1M | 1,000,000 | 96 | Cosine | [VectorBench](https://github.com/harsha-simhadri/big-ann-benchmarks) |
| GIST1M | 1,000,000 | 960 | L2 | [IRISA TEXMEX](http://corpus-texmex.irisa.fr/) |
| GloVe-200 | 1,183,514 | 200 | Cosine | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |
| SIFT10K | 10,000 | 128 | L2 | Auto-generated (for quick tests) |

## Running Experiments

### Via Orchestrator (SIFT10K quick tests)

```bash
# Check experiment status
uv run python -m learn_to_skip.orchestrator status

# Run all base experiments in dependency order
uv run python -m learn_to_skip.orchestrator run-all

# Run a single experiment
uv run python -m learn_to_skip.orchestrator run motivation

# Force re-run
uv run python -m learn_to_skip.orchestrator run-all --force
```

### Million-Scale Experiments (paper results)

These scripts produce the results reported in the paper:

```bash
# SIFT1M + Deep1M end-to-end benchmark
uv run python src/learn_to_skip/experiments/sift1m_deep1m_benchmark.py

# GIST1M (d=960) benchmark
uv run python src/learn_to_skip/experiments/gist1m_benchmark.py

# Heuristic baselines (rank-only, distance-only) with multi-threading
uv run python src/learn_to_skip/experiments/heuristic_baselines_multithread.py

# Alpha-pruning baseline (OptHNSW comparison)
uv run python src/learn_to_skip/experiments/alpha_pruning_baseline.py

# Combined alpha-pruning + learned tree
uv run python src/learn_to_skip/experiments/combined_baseline.py

# GloVe-200 + visited-set grid
uv run python src/learn_to_skip/experiments/visited_grid_glove.py

# Fallback, connectivity, adaptive threshold
uv run python src/learn_to_skip/experiments/fallback_connectivity_adaptive.py

# x86 co-location benchmark (run on x86 server)
uv run python src/learn_to_skip/experiments/x86_colocation_benchmark.py
```

Results are written to `results/` as JSON files with merge-on-write semantics (existing data is preserved).

## Generating Plots

```bash
# All figures
uv run python -m learn_to_skip.orchestrator plot-all

# Individual figures
uv run python -m learn_to_skip.orchestrator plot fig1   # Waste ratio
uv run python -m learn_to_skip.orchestrator plot fig2   # Speedup
uv run python -m learn_to_skip.orchestrator plot fig3   # Recall-speedup Pareto
uv run python -m learn_to_skip.orchestrator plot fig4   # Classifier ROC
uv run python -m learn_to_skip.orchestrator plot fig5   # Threshold sensitivity
uv run python -m learn_to_skip.orchestrator plot fig6   # Scalability
uv run python -m learn_to_skip.orchestrator plot fig7   # Adaptive ef
```

## Tests

```bash
uv run pytest tests/ -v

# Skip slow tests
uv run pytest tests/ -v -m "not slow"
```

## Key Results

### SIFT1M (d=128, L2, Apple M4)

| Mode | Threshold | Skip Rate | Speedup | R@10 |
|------|-----------|-----------|---------|------|
| Vanilla (1T) | - | 0% | 1.00x | .996 |
| Vanilla (10T) | - | 0% | 2.36x | .996 |
| Universal | 0.5 | 91.3% | **3.31x** | .927 |
| Universal | 0.7 | 77.6% | **2.19x** | .985 |
| Universal | 0.8 | 62.9% | **1.73x** | .992 |
| Multi-thread 4T | 0.7 | 77.6% | **3.60x** | .985 |

### GIST1M (d=960, L2, Apple M4)

| Mode | Threshold | Skip Rate | Speedup | R@10 |
|------|-----------|-----------|---------|------|
| Universal | 0.5 | 95.7% | **2.90x** | .870 |
| Combined (alpha+tree) | 0.8 | 77.2% | **2.08x** | .966 |

### Complementarity

LearnToSkip composes with existing acceleration techniques:

- **Stacked with reduced ef\_c=100**: 2.66x speedup at 0.980 R@10
- **Combined with alpha-pruning**: up to 5.45x at aggressive settings (0.915 R@10)
- **Combined quality-preserving** (alpha=5, tau=0.8): 1.74x at 0.990 R@10

## Forked hnswlib

The `hnswlib_fork/` submodule ([Icemap/hnswlib](https://github.com/Icemap/hnswlib)) extends the original [nmslib/hnswlib](https://github.com/nmslib/hnswlib) with:

- `BaseCandidateSkipFunctor` interface in `searchBaseLayer()` — zero overhead when disabled (single pointer null-check)
- `DecisionTreeSkipFunctor` with denormalized depth-3 tree and per-thread query projection buffers
- `AlphaPruningSkipFunctor` implementing OptHNSW-style alpha-pruning baseline
- Combined alpha-pruning + learned tree functor
- Co-located projected vector storage (64 bytes/node) for cache-friendly access
- SIMD-optimized 16-D L2 projection kernel (SSE on x86, NEON on ARM)
- Multi-threaded construction support with thread-safe skip functors
- Python bindings: `create_skip_functor()`, `add_items_with_skip()`, `enable_projection_storage()`, `get_construction_metrics()`
