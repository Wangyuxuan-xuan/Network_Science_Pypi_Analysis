# PyPI Dependency Network Analysis

## Project Overview

This project analyzes the PyPI (Python Package Index) dependency network to identify core packages and understand the ecosystem structure. The network consists of **397,797 packages** and **1,819,936 dependencies**, forming a large-scale directed acyclic graph (DAG).

---

## Objective 1: Core Package Identification using Centrality Analysis

Results in `results/centrality` folder
---

## Running the Analysis

### Prerequisites

```bash
# Create and activate conda environment
conda create -n pypi-net python=3.10 pandas networkx matplotlib seaborn tqdm -y
conda activate pypi-net
```

### 1. Compute Centrality Measures (Original Network)

Analyzes the real PyPI dependency network and computes all centrality measures:

```bash
python centrality.py
```

**Output:**
- `results/centrality/centrality_results.csv` - Full results for all packages
- `results/centrality/top_packages.csv` - Top 100 packages by PageRank
- `results/centrality/centrality_summary.txt` - Statistical summary
- `results/centrality/centrality_analysis.png` - 4-panel visualization
- `results/centrality/objective1_report.md` - Comprehensive analysis report

**Runtime:** ~50-75 minutes (with betweenness centrality with k=5000 sample)
without betweeness: 5mins
---

## Objective 5: Baseline Validation with Random Graphs


To determine if the observed centrality patterns are **genuine structural properties** or just **artifacts of the degree distribution**, we compare against randomized null models. 

The randomization algorithm:
- Preserves the **degree sequence** (in-degree and out-degree for each package)
- Preserves the **DAG property** (no cycles)
- Randomizes the **topology** (which packages depend on which)

This isolates the effect of network structure from the effect of popularity (degree).

### 2. Generate Random Graphs

Creates 5 randomized versions of the network using order-preserving edge swaps:

```bash
python generate_random_graphs.py
```

**Output:**
- `results/baseline/random_graphs/random_graph_1.pkl` through `random_graph_5.pkl`
- `results/baseline/randomization_stats.csv` - Generation statistics
- `results/baseline/random_graphs_config.json` - Configuration used

**Runtime:** ~20-60 minutes (5 graphs × 4-12 min each)

**Algorithm:** For each random graph, performs 18,199,370 edge swap attempts (10× the number of edges) to thoroughly randomize while preserving properties.

### 3. Compute Baseline Centrality

Computes centrality measures on all 5 random graphs and compares with the original:

```bash
python baseline_centrality.py
```

**Configuration** (in script):
```python
CONFIG = {
    'compute_betweenness': False,  # Skip betweenness (very slow, 75mins * 2)
    'use_parallel': True,          # Use parallel processing
    'n_jobs': min(5, cpu_count()), # Number of workers
}
```

**Output:**
- `results/baseline/baseline_comparison.csv` - Full comparison with z-scores
- `results/baseline/significant_packages.csv` - Statistically significant packages (|z| > 2)
- `results/baseline/comparison_summary.txt` - Statistical summary
- `results/baseline/baseline_comparison.png` - 4-panel visualization


### 4. Analyze Results (Notebook)

Interactive analysis comparing original vs baseline:

```bash
jupyter notebook results/baseline/centrality/result_analysis.ipynb
```

Or open the notebook in VS Code for interactive exploration.

---

## Z-Scores

**Z-score** = (Original Value - Baseline Mean) / Baseline Std Dev

- **z > 3**: Package is significantly **MORE** important than expected (99.7% confidence)
- **2 < z < 3**: Moderately more important (95% confidence)
- **|z| < 2**: Not significantly different from random
- **z < -2**: Significantly **LESS** important than expected

**Interpretation:** High positive z-scores indicate genuine structural importance beyond just having many dependents.

---

## Project Structure

```
Pypi_Project/
├── centrality.py                      # Main centrality analysis
├── generate_random_graphs.py          # Generate randomized graphs
├── baseline_centrality.py             # Baseline comparison
├── data/
│   ├── adjacency.jsonl               # Dependency edges
│   ├── nodes.csv                     # Package metadata
│   ├── edges.csv                     # Edge list format
│   └── graph_cache.pkl               # Cached graph (auto-generated)
└── results/
    ├── centrality/                    # Original network results
    │   ├── centrality_results.csv
    │   ├── objective1_report.md      # Full analysis report
    │   └── centrality_analysis.png
    └── baseline/                      # Baseline validation results
        ├── random_graphs/            # 5 randomized graphs
        ├── baseline_comparison.csv   # Comparison with z-scores
        └── centrality/
            └── result_analysis.ipynb # Interactive comparison

```
