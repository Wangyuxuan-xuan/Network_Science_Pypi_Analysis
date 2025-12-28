"""
Experiments module for resilience analysis.
Handles removal experiments and null model comparison.
"""

import pandas as pd
import numpy as np
import time

from config import CONFIG
from network_analysis import compute_network_metrics
from data_loader import load_random_graphs, compute_centrality_for_graph


def run_removal_experiment_on_graph(G, centrality_df, graph_label="graph", original_num_nodes=None):
    """
    Run removal experiments on a single graph.
    This extracts the core logic from perform_removal_experiments to allow
    running the same experiment on multiple graphs (original + randomized).
    
    Args:
        G: NetworkX directed graph
        centrality_df: DataFrame with centrality dataset (must have 'package' column and centrality metric)
        graph_label: Label for logging
        original_num_nodes: Original number of nodes (for normalization). If None, uses G.number_of_nodes()
        
    Returns:
        pd.DataFrame: Results for each k value
    """
    metric = CONFIG['centrality_metric']
    k_values = CONFIG['removal_k_values']
    
    # Get top packages sorted by centrality
    top_packages = centrality_df['package'].tolist()
    
    # Use original_num_nodes for normalization if provided, otherwise use current graph size
    if original_num_nodes is None:
        original_num_nodes = G.number_of_nodes()
    
    # Compute baseline metrics (no removal)
    baseline_metrics = compute_network_metrics(G, removed_nodes=[])
    baseline_lwcc_size = baseline_metrics['lwcc_size']
    
    # Perform removal experiments
    results = []
    
    for k in k_values:
        if k > len(top_packages):
            continue
        
        # Get nodes to remove (top k by centrality)
        nodes_to_remove = top_packages[:k]
        
        # Compute metrics after removal (with cascade failures)
        metrics = compute_network_metrics(G, removed_nodes=nodes_to_remove, enable_cascade=True)
        
        # Calculate relative changes
        lwcc_size_retention = metrics['lwcc_size'] / baseline_lwcc_size if baseline_lwcc_size > 0 else 0.0
        
        # Normalize LWCC size by original number of nodes (for null model comparison)
        lwcc_relative_size = metrics['lwcc_size'] / original_num_nodes if original_num_nodes > 0 else 0.0
        
        result = {
            'k': k,
            'lwcc_size': metrics['lwcc_size'],
            'lwcc_relative_size': lwcc_relative_size,  # Normalized by original graph size
            'lwcc_size_retention': lwcc_size_retention,  # Retention relative to this graph's baseline
            'nodes_failed_total': metrics['nodes_failed_total'],
            'cascade_failures': metrics['cascade_failures'],
        }
        
        results.append(result)
    
    return pd.DataFrame(results)


def perform_removal_experiments(G, centrality_df):
    """Perform targeted node removal experiments"""
    print("\n" + "=" * 80)
    print("PHASE 3: PERFORMING REMOVAL EXPERIMENTS")
    print("=" * 80)
    
    metric = CONFIG['centrality_metric']
    k_values = CONFIG['removal_k_values']
    
    # Get top packages sorted by centrality
    top_packages = centrality_df['package'].tolist()
    
    # Compute baseline metrics (no removal)
    print("\nComputing baseline metrics (no removal)...")
    baseline_metrics = compute_network_metrics(G, removed_nodes=[])
    baseline_lwcc_size = baseline_metrics['lwcc_size']
    baseline_lwcc_fraction = baseline_metrics['lwcc_fraction']
    
    print(f"  Baseline LWCC size: {baseline_lwcc_size:,} ({baseline_lwcc_fraction:.2%})")
    print(f"  Baseline nodes: {baseline_metrics['nodes_remaining']:,}")
    print(f"  Baseline edges: {baseline_metrics['edges_remaining']:,}")
    
    # Perform removal experiments
    results = []
    removed_nodes_so_far = []
    
    print(f"\nPerforming removal experiments for k = {k_values}...")
    
    for k in k_values:
        if k > len(top_packages):
            print(f"  ⚠ Skipping k={k} (only {len(top_packages)} packages available)")
            continue
        
        print(f"\n  Experiment: Removing top-{k} nodes...")
        start_time = time.time()
        
        # Get nodes to remove (top k)
        nodes_to_remove = top_packages[:k]
        removed_nodes_so_far = nodes_to_remove.copy()
        
        # Compute metrics after removal (with cascade failures)
        metrics = compute_network_metrics(G, removed_nodes=removed_nodes_so_far, enable_cascade=True)
        
        # Calculate relative changes
        lwcc_size_change = baseline_lwcc_size - metrics['lwcc_size']
        lwcc_fraction_change = baseline_lwcc_fraction - metrics['lwcc_fraction']
        lwcc_size_retention = metrics['lwcc_size'] / baseline_lwcc_size if baseline_lwcc_size > 0 else 0.0
        lwcc_fraction_retention = metrics['lwcc_fraction'] / baseline_lwcc_fraction if baseline_lwcc_fraction > 0 else 0.0
        
        # Normalize LWCC size by original number of nodes (for null model comparison)
        original_num_nodes = G.number_of_nodes()
        lwcc_relative_size = metrics['lwcc_size'] / original_num_nodes if original_num_nodes > 0 else 0.0
        
        elapsed = time.time() - start_time
        
        result = {
            'k': k,
            'nodes_removed': k,  # Initial removal
            'nodes_failed_total': metrics['nodes_failed_total'],  # Total failed (cascade)
            'cascade_failures': metrics['cascade_failures'],  # Additional cascade failures
            'nodes_remaining': metrics['nodes_remaining'],
            'edges_remaining': metrics['edges_remaining'],
            'weak_component_count': metrics['weak_component_count'],
            'lwcc_size': metrics['lwcc_size'],
            'lwcc_fraction': metrics['lwcc_fraction'],
            'lwcc_relative_size': lwcc_relative_size,  # Normalized by original graph size (for null model comparison)
            'lwcc_size_change': lwcc_size_change,
            'lwcc_fraction_change': lwcc_fraction_change,
            'lwcc_size_retention': lwcc_size_retention,
            'lwcc_fraction_retention': lwcc_fraction_retention,
            'strong_component_count': metrics['strong_component_count'],
            'lscc_size': metrics['lscc_size'],
            'density': metrics['density'],
            'avg_path_length': metrics['avg_path_length'],
            'computation_time': elapsed,
        }
        
        results.append(result)
        
        print(f"    ✓ Completed in {elapsed:.1f}s")
        print(f"      Initial removal: {k} nodes")
        print(f"      Total failed (cascade): {metrics['nodes_failed_total']:,} nodes ({metrics['cascade_failures']:,} cascade)")
        print(f"      LWCC size: {metrics['lwcc_size']:,} ({metrics['lwcc_fraction']:.2%})")
        print(f"      Retention: {lwcc_size_retention:.2%} of baseline")
        print(f"      Components: {metrics['weak_component_count']} weak, {metrics['strong_component_count']} strong")
    
    results_df = pd.DataFrame(results)
    
    print(f"\n✓ Completed {len(results)} removal experiments")
    
    return results_df, baseline_metrics


def run_null_model_comparison(G_original, original_num_nodes, original_centrality_df):
    """
    Run resilience experiments on randomized graphs and compute averaged dataset.
    This implements the null model comparison (Objective 5 baseline).
    
    Args:
        G_original: Original graph (for reference)
        original_num_nodes: Number of nodes in original graph (for normalization)
        original_centrality_df: Centrality dataset from original graph (for reference)
        
    Returns:
        pd.DataFrame: Averaged dataset across randomized graphs, with same structure as removal experiment dataset
    """
    print("\n" + "=" * 80)
    print("PHASE 3B: NULL MODEL COMPARISON (RANDOMIZED GRAPHS)")
    print("=" * 80)
    
    # Load randomized graphs
    random_graphs = load_random_graphs()
    
    if len(random_graphs) == 0:
        print("\n⚠ No randomized graphs available. Skipping null model comparison.")
        return None
    
    # Run experiments on each randomized graph
    all_results = []
    
    for graph_id, G_random in random_graphs:
        print(f"\n{'-' * 80}")
        print(f"Processing Random Graph {graph_id}/{len(random_graphs)}")
        print(f"{'-' * 80}")
        
        # Compute centrality on this randomized graph (same methodology)
        print(f"  Step 1: Computing centrality measures...")
        centrality_df_random = compute_centrality_for_graph(G_random, f"random_graph_{graph_id}")
        print(f"    ✓ Computed centrality for {len(centrality_df_random):,} packages")
        
        # Run removal experiments on this randomized graph
        print(f"  Step 2: Running removal experiments...")
        results_random = run_removal_experiment_on_graph(
            G_random, 
            centrality_df_random, 
            f"random_graph_{graph_id}",
            original_num_nodes=original_num_nodes  # Normalize by original graph size
        )
        
        all_results.append(results_random)
        print(f"    ✓ Completed {len(results_random)} removal experiments")
    
    # Average dataset across all randomized graphs
    print(f"\n{'-' * 80}")
    print("Averaging dataset across randomized graphs...")
    print(f"{'-' * 80}")
    
    # Group by k and compute mean
    k_values = CONFIG['removal_k_values']
    averaged_results = []
    
    for k in k_values:
        # Collect LWCC relative sizes for this k across all random graphs
        lwcc_relative_sizes = []
        lwcc_sizes = []
        lwcc_retentions = []
        
        for results_df in all_results:
            k_results = results_df[results_df['k'] == k]
            if len(k_results) > 0:
                lwcc_relative_sizes.append(k_results.iloc[0]['lwcc_relative_size'])
                lwcc_sizes.append(k_results.iloc[0]['lwcc_size'])
                lwcc_retentions.append(k_results.iloc[0]['lwcc_size_retention'])
        
        if len(lwcc_relative_sizes) > 0:
            averaged_results.append({
                'k': k,
                'lwcc_size': np.mean(lwcc_sizes),
                'lwcc_relative_size': np.mean(lwcc_relative_sizes),  # Mean normalized LWCC size
                'lwcc_size_retention': np.mean(lwcc_retentions),
                'lwcc_relative_size_std': np.std(lwcc_relative_sizes),  # Standard deviation for error bars
            })
    
    averaged_df = pd.DataFrame(averaged_results)
    
    print(f"✓ Computed averaged dataset across {len(random_graphs)} randomized graphs")
    print(f"  - Number of k values: {len(averaged_df)}")
    
    return averaged_df

