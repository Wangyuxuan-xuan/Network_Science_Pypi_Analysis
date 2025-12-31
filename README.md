# PyPI Dependency Network Analysis

## Project Overview

This project provides a comprehensive Network Science analysis of the Python Package Index (PyPI) ecosystem. By modeling over 700,000 projects as a complex directed graph, we investigate the hidden structural dynamics, dependencies, and systemic risks within the world’s largest Python software repository.

Our analysis transforms raw crawl data into a verified Directed Acyclic Graph (DAG) to map the "Global Supply Chain" of Python software through five core lenses:

* **Core Identification**: Leveraging centrality measures (PageRank, Betweenness) and Trophic Level analysis to distinguish foundational infrastructure from high-level applications.

* **Macro-Structural Topology**: Applying the Bow-Tie Model and connectivity analysis to identify the "Giant Component" of the ecosystem and isolate structural flaws like circular dependencies.

* **Community Detection**: Utilizing Louvain and LPA algorithms to uncover "Technology Stacks"—functional clusters of packages that are frequently co-dependent and used together.

* **Ecosystem Resilience**: Simulating targeted attacks on critical nodes to quantify Cascade Failures, measuring how the removal of a single core library (e.g., numpy) impacts the integrity of the entire network.

* **Baseline Validation**: Comparing real-world data against Randomized Null Models (degree-preserving DAGs) to ensure that discovered patterns are genuine structural properties rather than statistical artifacts.

This multi-stage pipeline provides researchers and stakeholders with a high-fidelity map of Python’s modular architecture and its vulnerability to supply chain disruptions.


## Data Acquisition & Processing

This project analyzes the global dependency network of the Python Package Index (PyPI). To ensure reproducibility and distinct analytical stages, the data acquisition pipeline is organized into three sequential versions, ranging from the raw crawl to the final Directed Acyclic Graph (DAG) used for analysis.

All datasets and the scripts used to generate them are located in the `dataset/` directory.

### Directory Structure

```text
dataset/
├── 01_pypi_graph_raw/          # Stage 1: Initial Crawl
│   ├── crawl_pypi_global.py    # Crawler script
│   └── pypi_global_graph/      # Raw output files
├── 02_pypi_graph_cleaned/      # Stage 2: Ghost Node Removal
│   ├── clean_ghost_nodes.py    # Cleaning script
│   └── pypi_clean_graph/       # Cleaned output files
└── 03_pypi_graph_dag_acyclic/  # Stage 3: Cycle Removal (Final Dataset)
    ├── convert_to_dag.py       # DAG conversion script
    ├── verify_dag.py           # Verification script
    └── pypi_dag/               # Final DAG dataset used for analysis

```

### Dataset Versions & Pipeline

The data processing pipeline consists of three stages. Each stage produces a self-contained dataset with consistent file formats (`nodes.csv`, `edges.csv`, `adjacency.jsonl`).

#### Stage 1: Raw Data (`01_pypi_graph_raw`)

* **Source:** The **PyPI Simple Index** (for the project list) and **PyPI JSON API** (for dependency metadata).
* **Methodology:** The crawler (`crawl_pypi_global.py`) fetched metadata for all available projects, normalizing names according to **PEP 503**. Dependencies were extracted from the `requires_dist` field, ignoring optional dependencies (extras) and version specifiers.

* **Statistics:**
* **Nodes:** 700,036 projects
* **Edges:** 1,842,517 dependency links 

#### Stage 2: Cleaned Data (`02_pypi_graph_cleaned`)

* **Objective:** Remove "Ghost Nodes"—packages that appear as dependencies (targets) but do not exist as primary entries (sources) in the index (e.g., deleted packages, typos in requirements, or private packages).
* **Methodology:** The `clean_ghost_nodes.py` script filtered the raw graph to ensure strictly valid edges where both source and target exist in the node list.
* **Statistics:**
* **Nodes:** 700,036 (Unchanged, as ghosts were only targets)
* **Edges:** 1,839,339 (Dropped 3,178 invalid edges)
* **Ghosts Removed:** 1,432 unique invalid package names 

#### Stage 3: Final DAG (`03_pypi_graph_dag_acyclic`)

* **Objective:** Construct a strict **Directed Acyclic Graph (DAG)** to enable flow-based centrality analysis (e.g., PageRank, Trophic Levels) and consistent randomization.
* **Methodology:** The `convert_to_dag.py` script identified Strongly Connected Components (SCCs) and self-loops.
* **Complex Cycles (Case A):** SCCs with size > 1 were removed (787 nodes).
* **Self-Loops (Case B):** Single nodes depending on themselves were removed (346 nodes).

* **Verification:** The resulting graph was verified using `verify_dag.py`, which performed a topological sort to confirm zero cycles.

* **Final Statistics (Used for Analysis):**
* **Nodes:** 698,903
* **Edges:** 1,819,937 

### File Formats

Each dataset folder contains the graph data in multiple formats to support different analysis tools:

| File | Description | Format Example |
| :--- | :--- | :--- |
| **nodes.csv** | List of all unique, normalized package names. | `package`<br>`numpy`<br>`pandas` |
| **edges.csv** | Directed edge list (Source &rarr; Target). | `source,target`<br>`pandas,numpy` |
| **adjacency.jsonl** | JSON Lines file mapping each package to its dependency list. Efficient for traversal. | `{"package": "pandas", "dependencies": ["numpy", ...]}` |
| **metadata.json** | Generation timestamp and processing statistics. | JSON Object |

### Reproducibility

To regenerate the datasets from scratch, execute the scripts in numerical order from the `dataset/` root:

1. **Crawl:** `python 01_pypi_graph_raw/crawl_pypi_global.py`
2. **Clean:** `python 02_pypi_graph_cleaned/clean_ghost_nodes.py 01_pypi_graph_raw/pypi_global_graph 02_pypi_graph_cleaned/pypi_clean_graph`
3. **DAG Conversion:** `python 03_pypi_graph_dag_acyclic/convert_to_dag.py 02_pypi_graph_cleaned/pypi_clean_graph 03_pypi_graph_dag_acyclic/pypi_dag`

## Objective 1: Core Package Identification using Centrality Analysis

Results in `results/centrality` folder:

### Running the Analysis

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



## Objective 2: Macro-Structural Analysis & Ecosystem Topology

This section analyzes the global architecture of the PyPI network to understand its fragmentation, circularity, and vertical hierarchy. The analysis is divided into three distinct sub-modules, each addressing a specific topological question using different graph representations.

All scripts and outputs for this objective are located in `02_macro_structure_analysis/`.

### 1. Connectivity Analysis (Fragmentation)

**Directory:** `02_macro_structure_analysis/01_connectivity_wcc`

* **Aim:** To determine if the Python ecosystem is a single unified "continent" or a fragmented archipelago. It treats the graph as **undirected** to compute Weakly Connected Components (WCC).
* **Input Data:** `pypi_clean_graph` (Stage 2 Data - contains ghost-node-free raw dependencies).
* **Key Insight:** Identifies the **Giant Connected Component (GCC)** and the "Long Tail" of isolated packages. This validates whether a global analysis (like centrality) applies to the whole ecosystem or just a subset.

| Output File | Description |
| --- | --- |
| `connectivity_metrics.json` | Global stats: Size of the GCC, percentage of isolated nodes, and fragmentation indices. |
| `connected_components_summary.csv` | List of all components ranked by size. |
| `wcc_rank_size_distribution.png` | **Zipf Plot**: Log-Log plot showing the power-law distribution of component sizes. |
| `wcc_size_frequency.png` | Histogram showing how frequent specific component sizes are (e.g., how many size-1 islands exist). |

### 2. Topology & Cycle Analysis (SCCs & Bow-Tie)

**Directory:** `02_macro_structure_analysis/02_topology_scc_bowtie`

* **Aim:** To detect "structural design flaws" (Circular Dependencies) and map the macroscopic flow of information using the **Bow-Tie Model** (Core, In, Out, Tendrils).
* **Input Data:** `pypi_clean_graph` (Stage 2 Data).
* **Methodology:**
1. **SCC Detection:** Identifies Strongly Connected Components. An SCC of size > 1 indicates a circular dependency (Case A: Complex Cycle). An SCC of size 1 with a self-edge is a Self-Loop (Case B).
2. **Bow-Tie Decomposition:** Classifies nodes based on their relationship to the largest SCC (The Core).

* **Key Insight:** Quantifies technical debt in the ecosystem by isolating packages involved in dependency cycles.

| Output File | Description |
| --- | --- |
| `circular_dependencies.json` | **Critical:** List of all complex cycles (SCC > 1) and the packages involved. |
| `self_loops.json` | List of packages that depend on themselves. |
| `node_health_distribution.png` | Bar chart comparing "Healthy" nodes vs. those involved in structural flaws. |
| `bowtie_region_sizes.png` | Visualization of the Bow-Tie structure (Core vs. Disconnected components). |
| `scc_summary.csv` | Detailed metrics for every Strongly Connected Component. |

### 3. Trophic Level Analysis (The Hierarchy)

**Directory:** `02_macro_structure_analysis/03_trophic_hierarchy`

* **Aim:** To determine the "Vertical Supply Chain" of the ecosystem. This analysis assigns a **Trophic Level (Height)** to every package, organizing the ecosystem from foundational libraries (Level 0) to high-level applications (Level N).
* **Input Data:** `pypi_dag` (Stage 3 Data - **Cycles Removed**). *Note: This analysis requires a strict DAG.*
* **Methodology:** Uses a longest-path algorithm where .
* **Level 0 (Basal):** Packages with no dependencies (e.g., `numpy`, `idna`).
* **Level N:** Packages that rely on deep chains of dependencies.

* **Key Insight:** Reveals the "Shape" of the ecosystem (Pyramid vs. Tower). A wide base indicates a stable, library-heavy ecosystem.

| Output File | Description |
| --- | --- |
| `trophic_levels.json` | Comprehensive stats: Package counts per level, basal fraction, and max depth. |
| `level_representatives.json` | The most connected/popular package at each trophic level (e.g., Level 2 Representative: `pandas`). |
| `trophic_pyramid.png` | **The Pyramid Plot**: Horizontal bar chart identifying the structural shape of PyPI. |



### Running the Analysis

To reproduce the macro-structural results, execute the scripts in the following order. Ensure the prerequisite datasets (`pypi_clean_graph` and `pypi_dag`) are generated as described in the "Data Acquisition" section.

```bash
# 1. Connectivity (WCC)
cd 02_macro_structure_analysis/01_connectivity_wcc
python analyze_connectivity.py --graph-dir pypi_clean_graph --output-dir objective2_connectivity_outputs

# 2. Topology & Cycles (SCC/Bow-Tie)
cd ../02_topology_scc_bowtie
python analyze_macro_structure.py --graph-dir pypi_clean_graph --output-dir objective2_outputs

# 3. Trophic Hierarchy (Requires DAG)
cd ../03_trophic_hierarchy
python analyze_trophic_levels.py --graph-dir pypi_dag --output-dir objective2_outputs

```


## Objective 3: Community Detection

This section identifies groups of Python packages that are frequently used together, which we term Technology Stacks. These stacks reveal the modular organization of the PyPI ecosystem and highlight how packages are clustered functionally.

All scripts and outputs for this objective are located in `03_community_detection/`. This section contains two methods for community detection: 01_Louvain_method and 02_LPA_method. Each method is applied to both Real Data (pypi_dag) and the Random Data generated in Objective 5, allowing comparison between the real PyPI network and randomized baselines.

### 1. Louvain Community Detection

**Directory:** `03_community_detection/01_Louvain_method`

* **Input Data:** Real Data: `pypi_dag` (Stage 3 Data - **Cycles Removed**). Random Data: the randomized datasets generated for Objective 5. *Note: This analysis requires a strict DAG.*
* **Methodology:** We project the directed PyPI dependency graph into an undirected graph. Then we apply the Louvain algorithm, which iteratively maximizes modularity (Q).
**The modularity score:** The Q value measures the strength of community division compared to a random graph with the same degree distribution.

Output Files Example: We take RealData as output files example.
| Output File | Description |
| --- | --- |
| `analysis_summary.txt` | Summary for top 5 largest communities, including modularity and community sizes. |
| `1_pagerank_core_for_gephi.gexf` | GEXF file for Gephi visualization of core nodes ranked by PageRank. |
| `top_5_stacks_summary.csv` | Summary table for the top 5 largest communities. |
| `Abstract Dependency Network of Top 10 PyPI Stacks.png` | Visualization of the dependency network for the top 10 largest PyPI communities. |
| `Core dependency structure for stack 6.png` | Visualization of the core dependency structure for stack 6. |
| `2_top_5_communities_for_gephi.gexf` | GEXF file for Gephi visualization of the top 5 largest communities. |
| `3_abstract_community_network.gexf` | GEXF file for Gephi visualization of the abstract community network. |
| `pypi_full_partition_realdata.csv` | Detailed full partition of all nodes for real data. |

* **Key Insight:** 
The real PyPI network exhibits a markedly stronger community structure than the randomized baselines. The observed modularity exceeds the baseline average by a wide margin, indicating that packages are organized into cohesive groups that are far denser internally than expected under random connectivity.


### 2. LPA (Label Propagation Algorithm)

**Directory:** `03_community_detection/02_LPA_method`

* **Input Data:** Real Data: `pypi_dag` (Stage 3 Data - **Cycles Removed**). Random Data: the randomized datasets generated for Objective 5. *Note: This analysis requires a strict DAG.*
* **Methodology:** We apply the Label Propagation Algorithm (LPA) on the same graph. LPA is a fast, heuristic approach that assigns nodes to communities based on iterative label propagation.
**Compared to Louvain:** While less aggressive than Louvain, it provides a robustness check to confirm that the modular structure identified is meaningful.

Output Files Example: We take RealData as output files example.
| Output File | Description |
| --- | --- |
| `analysis_summary_lpa.txt` | Summary for top 5 largest communities detected using LPA. |
| `1_pagerank_core_lpa.gexf` | GEXF file for Gephi visualization of core nodes ranked by PageRank using LPA. |
| `2_abstract_community_network_lpa.csv` | Summary table for the top 5 largest communities detected using LPA. |

* **Key Insight:**
The real network’s LPA Q is much more higher than the random baseline, highlighting strong modular structure. LPA reveals a dominant cluster containing 86% of packages, whereas the random network shows >98% in one cluster. This confirms that while the ecosystem is highly interconnected, Louvain better uncovers subtle modular boundaries.



## Objective 4: Assess Ecosystem Resilience

This section performs targeted node removal experiments to assess the resilience of the PyPI dependency network. It evaluates how the network integrity degrades when critical packages (identified by centrality analysis) are removed, simulating real-world scenarios where critical packages become unavailable.

All scripts and outputs for this objective are located in `04_assess_ecosystem_resilience/`.

### Methodology

**Targeted Node Removal Experiments:**
- Removes top-k nodes identified by centrality analysis (using PageRank from Objective 1)
- Tests removal values: k = [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]
- Uses the final DAG dataset (Stage 3) for analysis

**Cascade Failure Model:**
- When a package is removed, all packages that depend on it (directly or indirectly) also fail
- This simulates real-world scenarios where removing a critical package (e.g., `numpy`) causes all dependent packages to become non-functional
- Cascade failures are computed by finding all reachable nodes in the reverse dependency graph

**Network Integrity Metrics:**
- Total failed nodes (initial removal + cascade failures)
- Cascade failure count (additional failures beyond initial removal)
- Largest Weakly Connected Component (LWCC) size and retention
- Component fragmentation (weakly and strongly connected components)
- Network density and average path length

**Null Model Comparison (Objective 5 Baseline):**
- Compares original network resilience against randomized null models
- Runs the same removal experiments on 5 randomized graphs
- Randomized graphs preserve: degree sequence, DAG property, topological order
- Validates that resilience findings reflect genuine structural properties

### Running the Analysis

```bash
cd 04_assess_ecosystem_resilience
python resilience.py
```

**Prerequisites:**
- Requires completed Objective 1 (centrality analysis results)
- Requires randomized graphs from Objective 5 baseline analysis (optional, for null model comparison)

**Runtime:** ~40-45 minutes
- Original network removal experiments: ~3-4 minutes
- Null model comparison (5 randomized graphs): ~35-40 minutes

### Output Files

| Output File | Description |
| --- | --- |
| `results/resilience_results.csv` | Complete dataset for all removal experiments on original network |
| `results/null_model_results.csv` | Averaged dataset from 5 randomized graphs (for comparison) |
| `results/resilience_summary.txt` | Comprehensive analysis summary with key findings and critical thresholds |
| `results/resilience_analysis.png` | Visualization plots showing network fragmentation and LWCC retention |
| `results/resilience_null_model_comparison.png` | Comparison plot: original network vs averaged randomized networks |

### Key Findings

**Critical Thresholds:**
- **50% LWCC retention lost** at k = 2 (removing top 2 packages causes 266,102 total failures)
- **10% LWCC retention lost** at k = 20 (severe fragmentation occurs)
- **Significant fragmentation** (>100 components) begins at k = 1

**Impact of Removing Top Package:**
- Removing `numpy` (top package by PageRank) causes 134,710 total failures
- Cascade failures: 134,709 additional packages fail due to dependencies
- LWCC retention drops to 64.21% of baseline
- Creates 7,240 disconnected components

**Most Critical Packages:**
The top 10 most critical packages by removal impact:
1. `numpy` (PageRank: 0.0329, In-Degree: 81,923)
2. `typing-extensions` (PageRank: 0.0314, In-Degree: 15,444)
3. `requests` (PageRank: 0.0253, In-Degree: 72,340)
4. `odoo` (PageRank: 0.0138, In-Degree: 17,859)
5. `colorama` (PageRank: 0.0114, In-Degree: 7,372)
6. `six` (PageRank: 0.0109, In-Degree: 9,643)
7. `pandas` (PageRank: 0.0104, In-Degree: 53,306)
8. `click` (PageRank: 0.0076, In-Degree: 27,115)
9. `pydantic` (PageRank: 0.0073, In-Degree: 29,279)
10. `certifi` (PageRank: 0.0064, In-Degree: 5,572)

**Resilience Pattern:**
- Network shows **sudden fragmentation** pattern with cascade failures
- Most damage occurs from removing the top few packages
- Null model comparison validates that this fragility is a genuine structural property, not just an artifact of degree distribution

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

**Randomization Algorithm (Order-Preserving Edge Swaps for DAGs):**

The algorithm randomizes the network topology while preserving critical properties through a sophisticated edge-swapping procedure:

1. **Topological Ordering**: First, compute a topological sort of all nodes (packages), assigning each node a unique index that respects the DAG structure
2. **Edge Swap Selection**: Randomly select two edges: (u→v) and (x→y)
3. **Validity Checks**: Before swapping, verify four conditions:
   - All four nodes (u, v, x, y) are distinct
   - **DAG preservation**: In topological order, u < y AND x < v (ensures new edges maintain forward direction)
   - No multi-edges: (u→y) and (x→v) don't already exist
   - No self-loops: u ≠ y and x ≠ v
4. **Perform Swap**: If all checks pass, remove edges (u→v) and (x→y), add edges (u→y) and (x→v)
5. **Repeat**: Perform 10× number of edges swap attempts (18,199,370 attempts per graph) to ensure thorough randomization

This algorithm guarantees:
- **Exact degree preservation**: Every node maintains its original in-degree and out-degree
- **DAG property**: No cycles are ever introduced (proven by topological ordering constraint)
- **Topology randomization**: Only the specific dependency relationships change, not node importance by degree

The topological ordering constraint is the key innovation that allows randomization of DAGs without cycle detection overhead.

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



### Z-Scores

**Z-score** = (Original Value - Baseline Mean) / Baseline Std Dev

- **z > 3**: Package is significantly **MORE** important than expected (99.7% confidence)
- **2 < z < 3**: Moderately more important (95% confidence)
- **|z| < 2**: Not significantly different from random
- **z < -2**: Significantly **LESS** important than expected

**Interpretation:** High positive z-scores indicate genuine structural importance beyond just having many dependents.


