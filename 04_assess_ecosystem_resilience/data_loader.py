"""
Data loading module for resilience analysis.
Handles loading of graphs, centrality results, and random graphs.
"""

import pandas as pd
import networkx as nx
import numpy as np
import pickle
from pathlib import Path

from config import (
    GRAPH_CACHE_FILE,
    CENTRALITY_RESULTS_FILE,
    RANDOM_GRAPHS_DIR,
    CONFIG
)


def load_graph():
    """Load the cached graph"""
    print("=" * 80)
    print("PHASE 1: LOADING GRAPH")
    print("=" * 80)
    
    print(f"\nLoading graph from {GRAPH_CACHE_FILE}...")
    with open(GRAPH_CACHE_FILE, 'rb') as f:
        G = pickle.load(f)
    
    # Clean nan nodes if any
    nan_nodes = [n for n in G.nodes() if (isinstance(n, float) and np.isnan(n)) or (isinstance(n, str) and n == 'nan')]
    
    if nan_nodes:
        print(f"  ⚠ Found {len(nan_nodes)} nan nodes, cleaning...")
        G_clean = nx.DiGraph()
        valid_edges = [(u, v) for u, v in G.edges() 
                      if not ((isinstance(u, float) and np.isnan(u)) or 
                             (isinstance(v, float) and np.isnan(v)) or
                             (isinstance(u, str) and u == 'nan') or
                             (isinstance(v, str) and v == 'nan'))]
        G_clean.add_edges_from(valid_edges)
        G = G_clean
    
    print(f"✓ Graph loaded:")
    print(f"  - Nodes: {G.number_of_nodes():,}")
    print(f"  - Edges: {G.number_of_edges():,}")
    print(f"  - Density: {nx.density(G):.6f}")
    
    # Verify DAG
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"  - Is DAG: {is_dag}")
    
    return G


def load_centrality_results():
    """Load centrality dataset to identify top-k nodes"""
    print("\n" + "=" * 80)
    print("PHASE 2: LOADING CENTRALITY RESULTS")
    print("=" * 80)
    
    print(f"\nLoading centrality dataset from {CENTRALITY_RESULTS_FILE}...")
    centrality_df = pd.read_csv(CENTRALITY_RESULTS_FILE)
    
    # Sort by the primary centrality metric
    metric = CONFIG['centrality_metric']
    centrality_df = centrality_df.sort_values(metric, ascending=False).reset_index(drop=True)
    
    print(f"✓ Loaded centrality dataset for {len(centrality_df):,} packages")
    print(f"  - Primary metric: {metric}")
    print(f"  - Top 10 packages by {metric}:")
    for idx, row in centrality_df.head(10).iterrows():
        print(f"    {idx+1}. {row['package']} ({metric}={row[metric]:.8f})")
    
    return centrality_df


def compute_centrality_for_graph(G, graph_label="graph"):
    """
    Compute centrality measures for a graph and return as DataFrame.
    This is used for randomized graphs where we need to compute centrality
    on each graph separately (same methodology as original).
    
    Args:
        G: NetworkX directed graph
        graph_label: Label for logging purposes
        
    Returns:
        pd.DataFrame: Centrality dataset with columns ['package', 'pagerank', 'in_degree', ...]
    """
    print(f"  Computing PageRank for {graph_label}...")
    
    # Compute PageRank (same parameters as original analysis)
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    
    # Compute in-degree (for compatibility with existing code)
    in_degree = dict(G.in_degree())
    
    # Create DataFrame matching the structure of loaded centrality dataset
    centrality_data = []
    for node in G.nodes():
        centrality_data.append({
            'package': node,
            'pagerank': pagerank.get(node, 0.0),
            'in_degree': in_degree.get(node, 0),
        })
    
    centrality_df = pd.DataFrame(centrality_data)
    
    # Sort by the primary centrality metric
    metric = CONFIG['centrality_metric']
    centrality_df = centrality_df.sort_values(metric, ascending=False).reset_index(drop=True)
    
    return centrality_df


def load_random_graphs():
    """
    Load randomized graphs for null model comparison.
    
    Returns:
        list: List of (graph_id, graph) tuples
    """
    print("\n" + "=" * 80)
    print("LOADING RANDOMIZED GRAPHS FOR NULL MODEL COMPARISON")
    print("=" * 80)
    
    random_graphs = []
    num_graphs = CONFIG['num_random_graphs']
    
    for i in range(1, num_graphs + 1):
        graph_path = RANDOM_GRAPHS_DIR / f"random_graph_{i}.pkl"
        
        if not graph_path.exists():
            print(f"  ⚠ Warning: {graph_path} not found, skipping...")
            continue
        
        print(f"\n  Loading random graph {i} from {graph_path}...")
        with open(graph_path, 'rb') as f:
            G_random = pickle.load(f)
        
        # Clean nan nodes if any (same as original graph loading)
        nan_nodes = [n for n in G_random.nodes() if (isinstance(n, float) and np.isnan(n)) or (isinstance(n, str) and n == 'nan')]
        
        if nan_nodes:
            print(f"    ⚠ Found {len(nan_nodes)} nan nodes, cleaning...")
            G_clean = nx.DiGraph()
            valid_edges = [(u, v) for u, v in G_random.edges() 
                          if not ((isinstance(u, float) and np.isnan(u)) or 
                                 (isinstance(v, float) and np.isnan(v)) or
                                 (isinstance(u, str) and u == 'nan') or
                                 (isinstance(v, str) and v == 'nan'))]
            G_clean.add_edges_from(valid_edges)
            G_random = G_clean
        
        random_graphs.append((i, G_random))
        print(f"    ✓ Loaded: {G_random.number_of_nodes():,} nodes, {G_random.number_of_edges():,} edges")
    
    print(f"\n✓ Loaded {len(random_graphs)} randomized graphs")
    
    if len(random_graphs) == 0:
        print("  ⚠ No randomized graphs found. Run 'python generate_random_graphs.py' first.")
    
    return random_graphs

