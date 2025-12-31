"""
Step 1: Generate Random Graphs using Order-Degree-Preserving DAG Randomization
PyPI Dependency Network - Network Science Project

This script generates randomized null models that preserve:
- Degree sequence (in-degree and out-degree)
- DAG property (no cycles)
- Node set
"""

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import random
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import json
import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
DATA_DIR = Path("data")
GRAPH_CACHE_FILE = DATA_DIR / "graph_cache.pkl"

# Output directories
RESULTS_DIR = Path("results/baseline")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_GRAPHS_DIR = RESULTS_DIR / "random_graphs"
RANDOM_GRAPHS_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    'num_random_graphs': 5,
    'swaps_per_edge': 10,  # Standard: 10-100x number of edges
    'seed': 42,
}

# Set random seed for reproducibility
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])


def load_graph():
    """Load the cached graph"""
    print("=" * 80)
    print("PHASE 1: LOADING GRAPH")
    print("=" * 80)
    
    print(f"\nLoading graph from {GRAPH_CACHE_FILE}...")
    with open(GRAPH_CACHE_FILE, 'rb') as f:
        G_original = pickle.load(f)
    
    print(f"✓ Graph loaded")
    
    # Check for nan nodes and rebuild if necessary
    nan_nodes = [n for n in G_original.nodes() if (isinstance(n, float) and np.isnan(n)) or (isinstance(n, str) and n == 'nan')]
    
    if nan_nodes:
        print(f"  ⚠ Found {len(nan_nodes)} nan nodes, rebuilding graph...")
        # Rebuild graph without nan nodes
        G = nx.DiGraph()
        valid_edges = [(u, v) for u, v in G_original.edges() 
                      if not ((isinstance(u, float) and np.isnan(u)) or 
                             (isinstance(v, float) and np.isnan(v)) or
                             (isinstance(u, str) and u == 'nan') or
                             (isinstance(v, str) and v == 'nan'))]
        G.add_edges_from(valid_edges)
        print(f"  ✓ Cleaned graph rebuilt")
    else:
        G = G_original
    
    print(f"  - Nodes: {G.number_of_nodes():,}")
    print(f"  - Edges: {G.number_of_edges():,}")
    print(f"  - Density: {nx.density(G):.6f}")
    
    # Verify DAG
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"  - Is DAG: {is_dag}")
    
    if not is_dag:
        raise ValueError("Graph must be a DAG!")
    
    return G


def compute_topological_order(G):
    """Compute topological ordering for DAG"""
    print("\n" + "=" * 80)
    print("PHASE 2: COMPUTING TOPOLOGICAL ORDER")
    print("=" * 80)
    
    print("\nComputing topological sort...")
    topo_order = list(nx.topological_sort(G))
    
    # Create node -> index mapping
    order_map = {node: idx for idx, node in enumerate(topo_order)}
    
    print(f"✓ Topological ordering computed for {len(order_map):,} nodes")
    
    # Verify ordering is valid
    print("\nVerifying topological order...")
    violations = 0
    for u, v in G.edges():
        if order_map[u] >= order_map[v]:
            violations += 1
    
    if violations > 0:
        raise ValueError(f"Topological order has {violations} violations!")
    
    print("✓ Topological order is valid (all edges go forward)")
    
    return order_map


def edge_swap_randomization(G, order_map, num_swaps, graph_id):
    """
    Perform order-preserving edge swaps to randomize graph while preserving:
    - Degree sequence
    - DAG property (via topological order)
    """
    print(f"\nRandomizing graph {graph_id}...")
    print(f"  Target swaps: {num_swaps:,}")
    
    # Create copy
    G_random = G.copy()
    
    # Pre-compute edge list for faster sampling
    edges_list = list(G_random.edges())
    
    successful_swaps = 0
    attempts = 0
    
    # Progress bar
    pbar = tqdm(total=num_swaps, desc=f"  Graph {graph_id} swaps")
    
    while successful_swaps < num_swaps and attempts < num_swaps * 2:
        attempts += 1
        
        # Randomly select two edges
        if len(edges_list) < 2:
            break
            
        edge1_idx = random.randint(0, len(edges_list) - 1)
        edge2_idx = random.randint(0, len(edges_list) - 1)
        
        if edge1_idx == edge2_idx:
            continue
        
        u, v = edges_list[edge1_idx]
        x, y = edges_list[edge2_idx]
        
        # Check: all four nodes distinct
        if len({u, v, x, y}) != 4:
            continue
        
        # Check: order preserved (u < y and x < v in topological order)
        if not (order_map[u] < order_map[y] and order_map[x] < order_map[v]):
            continue
        
        # Check: new edges don't already exist
        if G_random.has_edge(u, y) or G_random.has_edge(x, v):
            continue
        
        # Perform swap
        G_random.remove_edge(u, v)
        G_random.remove_edge(x, y)
        G_random.add_edge(u, y)
        G_random.add_edge(x, v)
        
        # Update edges list
        edges_list[edge1_idx] = (u, y)
        edges_list[edge2_idx] = (x, v)
        
        successful_swaps += 1
        pbar.update(1)
        
        # Update progress bar with stats
        if successful_swaps % 1000 == 0:
            success_rate = successful_swaps / attempts if attempts > 0 else 0
            pbar.set_postfix({'success_rate': f'{success_rate:.2%}', 'attempts': attempts})
    
    pbar.close()
    
    success_rate = successful_swaps / attempts if attempts > 0 else 0
    print(f"  ✓ Completed: {successful_swaps:,} successful swaps")
    print(f"    Attempts: {attempts:,}, Success rate: {success_rate:.2%}")
    
    if success_rate < 0.05:
        print(f"  ⚠ Warning: Low success rate ({success_rate:.2%})")
    
    return G_random, successful_swaps, attempts


def validate_randomized_graph(G_original, G_random, graph_id):
    """Validate that randomized graph preserves required properties"""
    print(f"\nValidating randomized graph {graph_id}...")
    
    # Check: Still a DAG
    is_dag = nx.is_directed_acyclic_graph(G_random)
    print(f"  - Is DAG: {is_dag}")
    if not is_dag:
        raise ValueError(f"Randomized graph {graph_id} contains cycles!")
    
    # Check: Same number of edges
    edges_match = G_original.number_of_edges() == G_random.number_of_edges()
    print(f"  - Edges preserved: {edges_match} ({G_random.number_of_edges():,})")
    
    # Check: Same number of nodes
    nodes_match = G_original.number_of_nodes() == G_random.number_of_nodes()
    print(f"  - Nodes preserved: {nodes_match} ({G_random.number_of_nodes():,})")
    
    # Check: Degree sequence preserved
    in_deg_original = dict(G_original.in_degree())
    in_deg_random = dict(G_random.in_degree())
    out_deg_original = dict(G_original.out_degree())
    out_deg_random = dict(G_random.out_degree())
    
    degree_violations = 0
    for node in G_original.nodes():
        if in_deg_original[node] != in_deg_random[node]:
            degree_violations += 1
        if out_deg_original[node] != out_deg_random[node]:
            degree_violations += 1
    
    print(f"  - Degree sequence preserved: {degree_violations == 0} ({degree_violations} violations)")
    
    if not (is_dag and edges_match and nodes_match and degree_violations == 0):
        raise ValueError(f"Validation failed for graph {graph_id}!")
    
    print(f"  ✓ All validations passed for graph {graph_id}")
    
    return True


def generate_random_graphs(G, order_map, num_graphs):
    """Generate multiple randomized graphs"""
    print("\n" + "=" * 80)
    print("PHASE 3: GENERATING RANDOMIZED GRAPHS")
    print("=" * 80)
    
    num_edges = G.number_of_edges()
    num_swaps = num_edges * CONFIG['swaps_per_edge']
    
    print(f"\nConfiguration:")
    print(f"  - Number of random graphs: {num_graphs}")
    print(f"  - Swaps per edge ratio: {CONFIG['swaps_per_edge']}")
    print(f"  - Total swaps per graph: {num_swaps:,}")
    print(f"  - Estimated time: ~{num_graphs * 5}-{num_graphs * 15} minutes")
    
    stats = []
    
    for i in range(1, num_graphs + 1):
        print(f"\n{'-' * 80}")
        print(f"Generating Random Graph {i}/{num_graphs}")
        print(f"{'-' * 80}")
        
        start_time = time.time()
        
        # Randomize
        G_random, successful_swaps, attempts = edge_swap_randomization(
            G, order_map, num_swaps, i
        )
        
        # Validate
        validate_randomized_graph(G, G_random, i)
        
        # Save graph
        save_path = RANDOM_GRAPHS_DIR / f"random_graph_{i}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(G_random, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved to {save_path}")
        
        elapsed = time.time() - start_time
        stats.append({
            'graph_id': i,
            'successful_swaps': successful_swaps,
            'attempts': attempts,
            'success_rate': successful_swaps / attempts if attempts > 0 else 0,
            'time_seconds': elapsed,
            'saved_path': str(save_path)
        })
        
        print(f"  ✓ Graph {i} completed in {elapsed:.1f}s")
    
    # Save stats
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(RESULTS_DIR / "randomization_stats.csv", index=False)
    print(f"\n✓ Randomization statistics saved to {RESULTS_DIR / 'randomization_stats.csv'}")
    
    # Save configuration
    config_file = RESULTS_DIR / "random_graphs_config.json"
    with open(config_file, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"✓ Configuration saved to {config_file}")
    
    return stats


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATE RANDOM GRAPHS")
    print("Order-Degree-Preserving DAG Randomization")
    print("=" * 80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Phase 1: Load graph
    G = load_graph()
    
    # Phase 2: Compute topological order
    order_map = compute_topological_order(G)
    
    # Phase 3: Generate random graphs
    stats = generate_random_graphs(G, order_map, CONFIG['num_random_graphs'])
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("RANDOM GRAPH GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"✓ Generated {CONFIG['num_random_graphs']} randomized graphs")
    print(f"✓ All graphs saved to: {RANDOM_GRAPHS_DIR}/")
    print(f"\nNext step: Run 'python baseline_centrality.py' to compute centrality on random graphs")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
