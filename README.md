# LearnToSkip

Lightweight classifier-guided candidate pruning for HNSW index construction.

During HNSW construction, the beam search evaluates far more candidates than it retains as neighbors. On SIFT10K with standard parameters (M=16, ef\_construction=200), **98.25% of candidate distance evaluations are wasted**. LearnToSkip trains a depth-3 decision tree to predict which candidates will be discarded *before* computing their exact distances, achieving up to **2.6x wall-clock speedup** with modest recall trade-off.

## Repository Structure

```
src/learn_to_skip/          # Core library
  builders/                 # Index construction strategies (vanilla, learned skip, C++ skip)
  classifiers/              # Classifier wrappers (decision tree, logistic, SVM, XGBoost)
  experiments/              # Reproducible experiment definitions
  features/                 # Feature extraction (random projection, encounter rank)
  datasets/                 # Dataset registry and loaders
  tracer/                   # HNSW construction tracing (records all candidate evaluations)
  orchestrator/             # CLI runner for experiments and plots
  adaptive/                 # Adaptive ef_construction
  visualization/            # Plotting utilities
hnswlib_fork/               # Forked hnswlib with skip injection (git submodule)
benchmark_cpp/              # C++ micro-benchmarks (tree inference vs L2 distance)
results/                    # Experiment output CSVs (checked in for reproducibility)
tests/                      # Unit tests
```

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/Icemap/new_proj2.git
cd new_proj2

# Install dependencies
uv sync

# Install the forked hnswlib (required for C++ skip experiments)
python3 -m pip install --force-reinstall --no-deps ./hnswlib_fork
```

## Data

SIFT10K is auto-generated on first run (synthetic 10K vectors, 128-dim). No manual download needed.

For larger datasets, the framework downloads automatically on first use:

| Dataset | Vectors | Dimensions | Source |
|---------|---------|------------|--------|
| sift10k | 10,000 | 128 | Auto-generated |
| sift1m | 1,000,000 | 128 | [IRISA TEXMEX](http://corpus-texmex.irisa.fr/) |
| gist1m | 1,000,000 | 960 | [IRISA TEXMEX](http://corpus-texmex.irisa.fr/) |
| glove200 | 1,183,514 | 200 | [Stanford GloVe](https://nlp.stanford.edu/projects/glove/) |

Data is stored under `data/raw/` and `data/processed/` (gitignored due to size).

## Running Experiments

All experiments are managed through the CLI orchestrator:

```bash
# Check experiment status
uv run python -m learn_to_skip.orchestrator status

# Run all 9 experiments in dependency order (SIFT10K, ~10 min)
uv run python -m learn_to_skip.orchestrator run-all

# Run a single experiment
uv run python -m learn_to_skip.orchestrator run motivation
uv run python -m learn_to_skip.orchestrator run classifier
uv run python -m learn_to_skip.orchestrator run threshold

# Force re-run
uv run python -m learn_to_skip.orchestrator run-all --force
```

### Experiment Dependency Order

```
motivation ──┬── build_speed ──── recall
             ├── classifier ──┬── ablation
             │                └── threshold
             ├── scalability
             ├── generalization
             └── adaptive
```

### Available Experiments

| Experiment | Description | Output |
|-----------|-------------|--------|
| `motivation` | Waste ratio analysis across parameters | `results/motivation/` |
| `build_speed` | Distance computation speedup | `results/build_speed/` |
| `classifier` | Classifier comparison (tree, logistic, SVM, XGBoost) | `results/classifier/` |
| `ablation` | Feature ablation study | `results/ablation/` |
| `threshold` | Skip threshold sensitivity | `results/threshold/` |
| `recall` | Recall@k impact analysis | `results/recall/` |
| `scalability` | Scaling behavior with dataset size | `results/scalability/` |
| `generalization` | Cross-distribution transfer | `results/generalization/` |
| `adaptive` | Adaptive ef_construction compensation | `results/adaptive/` |

### C++ End-to-End Benchmark

The C++ benchmark requires the forked hnswlib to be installed:

```bash
# Make sure hnswlib_fork is installed
python3 -m pip install --force-reinstall --no-deps ./hnswlib_fork

# Run via orchestrator (included as additional experiments)
uv run python -m learn_to_skip.orchestrator run online_training
uv run python -m learn_to_skip.orchestrator run universal_classifier
```

Or run the E2E benchmark directly:

```bash
uv run python -c "
from learn_to_skip.experiments.cpp_e2e_benchmark import CppE2EBenchmark
bench = CppE2EBenchmark()
bench.run()
"
```

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

## Key Results (SIFT10K)

| Mode | Threshold | Skip Rate | Wall-Clock Speedup | Recall@10 |
|------|-----------|-----------|-------------------|-----------|
| Vanilla (1 thread) | - | 0% | 1.00x | 0.970 |
| Universal | 0.5 | 88.5% | **2.43x** | 0.865 |
| Universal | 0.7 | 74.9% | **1.50x** | 0.928 |
| Universal | 0.8 | 60.8% | **1.26x** | 0.944 |

## Forked hnswlib

The `hnswlib_fork/` submodule ([Icemap/hnswlib](https://github.com/Icemap/hnswlib)) extends the original [nmslib/hnswlib](https://github.com/nmslib/hnswlib) with:

- `BaseCandidateSkipFunctor` interface in `searchBaseLayer()`
- `DecisionTreeSkipFunctor` with denormalized depth-3 tree
- Python bindings: `create_skip_functor()`, `add_items_with_skip()`, `get_construction_metrics()`
- Zero overhead when disabled (single pointer null-check)

## License

MIT
