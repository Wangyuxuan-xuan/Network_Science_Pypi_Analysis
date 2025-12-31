"""
Configuration module for resilience analysis.
Contains all paths, directories, and configuration parameters.
"""

from pathlib import Path

# Set up paths
DATA_DIR = Path("data")
GRAPH_CACHE_FILE = DATA_DIR / "graph_cache.pkl"
METADATA_FILE = DATA_DIR / "metadata.json"

# Load centrality dataset
CENTRALITY_DIR = Path("dataset/centrality")
CENTRALITY_RESULTS_FILE = CENTRALITY_DIR / "centrality_results.csv"
TOP_PACKAGES_FILE = CENTRALITY_DIR / "top_packages.csv"

# Output files - all outputs go to results/ directory in 04_assess_ecosystem_resilience
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "resilience_results.csv"
SUMMARY_FILE = RESULTS_DIR / "resilience_summary.txt"
PLOTS_DIR = RESULTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Random graphs directory (for null model comparison)
RANDOM_GRAPHS_DIR = Path("dataset/baseline/random_graphs")

# Configuration
CONFIG = {
    'removal_k_values': [1, 2, 3, 5, 10, 20, 50, 100, 200, 500],  # Reduced range due to cascade failures
    'centrality_metric': 'pagerank',  # Primary metric for ranking
    'enable_cascade_failures': True,  # Enable cascade failure simulation
    'num_random_graphs': 5,  # Number of randomized graphs for null model comparison
}

