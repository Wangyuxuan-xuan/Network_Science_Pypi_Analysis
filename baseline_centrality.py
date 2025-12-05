"""
Step 2: Compute Baseline Centrality on Random Graphs
PyPI Dependency Network - Network Science Project

This script:
1. Loads the original graph and random graphs
2. Computes centrality measures (same as centrality.py)
3. Compares original vs baseline statistics
4. Computes z-scores
5. Generates visualizations
"""

import pandas as pd
import networkx as nx
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from multiprocessing import Pool, cpu_count

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Set up paths
DATA_DIR = Path("data")
GRAPH_CACHE_FILE = DATA_DIR / "graph_cache.pkl"

RESULTS_DIR = Path("results/baseline/centrality")
RANDOM_GRAPHS_DIR = RESULTS_DIR / "random_graphs"
PLOTS_DIR = RESULTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load original centrality results
CENTRALITY_DIR = Path("results/centrality")
ORIGINAL_CENTRALITY_FILE = CENTRALITY_DIR / "centrality_results.csv"

# Configuration
CONFIG = {
    'compute_betweenness': True,  # Set to True to compute betweenness (very slow, 75min for k=5000 )
    'use_parallel': True,  # Use parallel processing for random graphs
    'n_jobs': min(5, cpu_count()),  # Number of parallel jobs
}


def load_graph():
    """Load the original cached graph"""
    print("=" * 80)
    print("PHASE 1: LOADING ORIGINAL GRAPH")
    print("=" * 80)
    
    print(f"\nLoading graph from {GRAPH_CACHE_FILE}...")
    with open(GRAPH_CACHE_FILE, 'rb') as f:
        G_original = pickle.load(f)
    
    # Clean nan nodes if any
    nan_nodes = [n for n in G_original.nodes() if (isinstance(n, float) and np.isnan(n)) or (isinstance(n, str) and n == 'nan')]
    
    if nan_nodes:
        print(f"  ⚠ Found {len(nan_nodes)} nan nodes, rebuilding graph...")
        G = nx.DiGraph()
        valid_edges = [(u, v) for u, v in G_original.edges() 
                      if not ((isinstance(u, float) and np.isnan(u)) or 
                             (isinstance(v, float) and np.isnan(v)) or
                             (isinstance(u, str) and u == 'nan') or
                             (isinstance(v, str) and v == 'nan'))]
        G.add_edges_from(valid_edges)
    else:
        G = G_original
    
    print(f"✓ Original graph loaded:")
    print(f"  - Nodes: {G.number_of_nodes():,}")
    print(f"  - Edges: {G.number_of_edges():,}")
    
    return G


def load_random_graphs():
    """Load all random graphs"""
    print("\n" + "=" * 80)
    print("PHASE 2: LOADING RANDOM GRAPHS")
    print("=" * 80)
    
    # Find all random graph files
    graph_files = sorted(RANDOM_GRAPHS_DIR.glob("random_graph_*.pkl"))
    
    if not graph_files:
        raise FileNotFoundError(f"No random graphs found in {RANDOM_GRAPHS_DIR}/")
    
    print(f"\nFound {len(graph_files)} random graphs")
    
    random_graphs = []
    for graph_file in graph_files:
        print(f"  Loading {graph_file.name}...", end=" ")
        with open(graph_file, 'rb') as f:
            G_random = pickle.load(f)
        print(f"✓ ({G_random.number_of_nodes():,} nodes, {G_random.number_of_edges():,} edges)")
        random_graphs.append(G_random)
    
    print(f"\n✓ Loaded {len(random_graphs)} random graphs")
    
    return random_graphs


def compute_centrality(G, graph_id, is_original=False, compute_betweenness=False):
    """Compute centrality measures for a graph (same metrics as centrality.py)"""
    label = "ORIGINAL" if is_original else f"RANDOM {graph_id}"
    print(f"\nComputing centrality for {label}...")
    
    centrality = {}
    
    # PageRank
    print(f"  - PageRank...", end=" ", flush=True)
    start = time.time()
    centrality['pagerank'] = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    print(f"✓ ({time.time() - start:.1f}s)")
    
    # In-degree centrality
    print(f"  - In-degree centrality...", end=" ", flush=True)
    centrality['in_degree_centrality'] = nx.in_degree_centrality(G)
    print(f"✓")
    
    # Out-degree centrality
    print(f"  - Out-degree centrality...", end=" ", flush=True)
    centrality['out_degree_centrality'] = nx.out_degree_centrality(G)
    print(f"✓")
    
    # Eigenvector centrality
    print(f"  - Eigenvector centrality...", end=" ", flush=True)
    try:
        centrality['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=200, tol=1e-6)
        print(f"✓")
    except:
        centrality['eigenvector_centrality'] = {node: 0.0 for node in G.nodes()}
        print(f"⚠ Failed (using zeros)")
    
    # Betweenness centrality (optional - very slow)
    if compute_betweenness:
        print(f"  - Betweenness centrality (k=5000)...", end=" ", flush=True)
        start = time.time()
        try:
            k_sample = min(5000, G.number_of_nodes())
            centrality['betweenness_centrality'] = nx.betweenness_centrality(G, k=k_sample, normalized=True, seed=42)
            print(f"✓ ({time.time() - start:.1f}s)")
        except:
            centrality['betweenness_centrality'] = {node: 0.0 for node in G.nodes()}
            print(f"⚠ Failed (using zeros)")
    else:
        print(f"  - Betweenness centrality... ⊘ Skipped (too slow)")
        centrality['betweenness_centrality'] = {node: 0.0 for node in G.nodes()}
    
    return centrality


def compute_centrality_wrapper(args):
    """Wrapper function for parallel processing"""
    graph_file, graph_id, compute_betweenness = args
    
    # Load graph
    with open(graph_file, 'rb') as f:
        G = pickle.load(f)
    
    # Compute centrality
    centrality = compute_centrality(G, graph_id, is_original=False, compute_betweenness=compute_betweenness)
    
    return centrality


def compute_baseline_centrality(G_original, random_graphs):
    """Compute centrality on all graphs and create baseline statistics"""
    print("\n" + "=" * 80)
    print("PHASE 3: COMPUTING BASELINE CENTRALITY")
    print("=" * 80)
    
    compute_betweenness = CONFIG['compute_betweenness']
    use_parallel = CONFIG['use_parallel']
    
    if compute_betweenness:
        print(f"\n⚠ Betweenness centrality is ENABLED (this will be very slow)")
    else:
        print(f"\n✓ Betweenness centrality is DISABLED (faster execution)")
    
    # Compute for original
    print("\nOriginal Graph:")
    original_centrality = compute_centrality(G_original, 0, is_original=True, compute_betweenness=compute_betweenness)
    
    # Compute for each random graph
    if use_parallel and len(random_graphs) > 1:
        print(f"\n✓ Using parallel processing with {CONFIG['n_jobs']} workers")
        print("  (Progress output may be interleaved)\n")
        
        # Get graph files
        graph_files = sorted(RANDOM_GRAPHS_DIR.glob("random_graph_*.pkl"))
        
        # Prepare arguments
        args_list = [(str(f), i+1, compute_betweenness) for i, f in enumerate(graph_files)]
        
        # Parallel computation
        with Pool(processes=CONFIG['n_jobs']) as pool:
            random_centralities = pool.map(compute_centrality_wrapper, args_list)
    else:
        print(f"\n✓ Using sequential processing")
        random_centralities = []
        for i, G_random in enumerate(random_graphs, 1):
            centrality = compute_centrality(G_random, i, compute_betweenness=compute_betweenness)
            random_centralities.append(centrality)
    
    # Aggregate baseline statistics
    print("\n" + "=" * 80)
    print("PHASE 4: COMPUTING BASELINE STATISTICS")
    print("=" * 80)
    
    print("\nAggregating results across random graphs...")
    
    nodes = list(G_original.nodes())
    
    # Only include betweenness if it was computed
    if CONFIG['compute_betweenness']:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality', 'betweenness_centrality']
    else:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality']
    
    baseline_stats = {}
    
    for metric in metrics:
        print(f"  - Processing {metric}...")
        baseline_stats[metric] = {}
        
        for node in tqdm(nodes, desc=f"    {metric}", leave=False):
            # Get values from all random graphs
            random_values = [
                centrality[metric][node] 
                for centrality in random_centralities
            ]
            
            baseline_stats[metric][node] = {
                'mean': np.mean(random_values),
                'std': np.std(random_values, ddof=1) if len(random_values) > 1 else 0.0,
                'min': np.min(random_values),
                'max': np.max(random_values),
                'median': np.median(random_values)
            }
    
    print("✓ Baseline statistics computed")
    
    return original_centrality, baseline_stats


def compute_z_scores(original_centrality, baseline_stats):
    """Compute z-scores for each node and metric"""
    print("\n" + "=" * 80)
    print("PHASE 5: COMPUTING Z-SCORES")
    print("=" * 80)
    
    nodes = list(original_centrality['pagerank'].keys())
    
    # Only include betweenness if it was computed
    if CONFIG['compute_betweenness']:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality', 'betweenness_centrality']
    else:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality']
    
    z_scores = {}
    
    for metric in metrics:
        print(f"\n  Computing z-scores for {metric}...")
        z_scores[metric] = {}
        
        for node in tqdm(nodes, desc=f"    {metric}", leave=False):
            real_value = original_centrality[metric][node]
            baseline_mean = baseline_stats[metric][node]['mean']
            baseline_std = baseline_stats[metric][node]['std']
            
            if baseline_std > 0:
                z = (real_value - baseline_mean) / baseline_std
            else:
                z = 0.0  # No variation in baseline
            
            z_scores[metric][node] = z
    
    print("\n✓ Z-scores computed for all nodes and metrics")
    
    return z_scores


def create_comparison_dataframe(original_centrality, baseline_stats, z_scores):
    """Create comprehensive comparison dataframe"""
    print("\nCreating comparison dataframe...")
    
    nodes = list(original_centrality['pagerank'].keys())
    
    # Only include betweenness if it was computed
    if CONFIG['compute_betweenness']:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality', 'betweenness_centrality']
    else:
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
                   'eigenvector_centrality']
    
    data = {'package': nodes}
    
    for metric in metrics:
        data[f'real_{metric}'] = [original_centrality[metric][node] for node in nodes]
        data[f'baseline_mean_{metric}'] = [baseline_stats[metric][node]['mean'] for node in nodes]
        data[f'baseline_std_{metric}'] = [baseline_stats[metric][node]['std'] for node in nodes]
        data[f'z_score_{metric}'] = [z_scores[metric][node] for node in nodes]
    
    df = pd.DataFrame(data)
    
    # Sort by z-score PageRank (descending)
    df = df.sort_values('z_score_pagerank', ascending=False).reset_index(drop=True)
    
    print(f"✓ Dataframe created with {len(df):,} packages")
    
    return df


def analyze_results(comparison_df):
    """Analyze and print key results"""
    print("\n" + "=" * 80)
    print("PHASE 6: ANALYZING RESULTS")
    print("=" * 80)
    
    # Overall z-score statistics
    print("\n1. Z-Score Distribution (PageRank):")
    z_pr = comparison_df['z_score_pagerank']
    print(f"   Mean: {z_pr.mean():.2f}")
    print(f"   Median: {z_pr.median():.2f}")
    print(f"   Std: {z_pr.std():.2f}")
    print(f"   Min: {z_pr.min():.2f}")
    print(f"   Max: {z_pr.max():.2f}")
    
    # Significance counts
    print("\n2. Significance Levels:")
    sig_2 = (z_pr.abs() > 2).sum()
    sig_3 = (z_pr.abs() > 3).sum()
    total = len(z_pr)
    print(f"   |z| > 2 (95% confidence): {sig_2:,} ({sig_2/total*100:.1f}%)")
    print(f"   |z| > 3 (99.7% confidence): {sig_3:,} ({sig_3/total*100:.1f}%)")
    
    # Top packages by z-score
    print("\n3. Top 20 Packages by Z-Score (Most significant deviation):")
    print(f"   {'Rank':<6} {'Package':<35} {'Z-Score':<10} {'Real PR':<12} {'Base PR':<12}")
    print("   " + "-" * 75)
    for idx, row in comparison_df.head(20).iterrows():
        print(f"   {idx+1:<6} {row['package']:<35} {row['z_score_pagerank']:<10.2f} "
              f"{row['real_pagerank']:<12.8f} {row['baseline_mean_pagerank']:<12.8f}")


def create_visualizations(comparison_df):
    """Create visualizations comparing original vs baseline"""
    print("\n" + "=" * 80)
    print("PHASE 7: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\nGenerating comparison plots...")
    
    try:
        sns.set_style("whitegrid")
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Z-score distribution
        print("  - Z-score distribution histogram...")
        z_pr = comparison_df['z_score_pagerank']
        axes[0, 0].hist(z_pr[z_pr.abs() < 50], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 0].axvline(x=2, color='red', linestyle='--', label='|z|=2 (95%)')
        axes[0, 0].axvline(x=-2, color='red', linestyle='--')
        axes[0, 0].axvline(x=3, color='darkred', linestyle='--', label='|z|=3 (99.7%)')
        axes[0, 0].axvline(x=-3, color='darkred', linestyle='--')
        axes[0, 0].set_xlabel('Z-Score (PageRank)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Z-Score Distribution (PageRank)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Top 20 packages by z-score
        print("  - Top 20 packages bar chart...")
        top20 = comparison_df.head(20)
        y_pos = range(len(top20))
        axes[0, 1].barh(y_pos, top20['z_score_pagerank'], color='steelblue')
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(top20['package'], fontsize=8)
        axes[0, 1].set_xlabel('Z-Score')
        axes[0, 1].set_title('Top 20 Packages by Z-Score (PageRank)')
        axes[0, 1].invert_yaxis()
        axes[0, 1].axvline(x=3, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # 3. Real vs Baseline PageRank scatter
        print("  - Real vs Baseline scatter plot...")
        axes[1, 0].scatter(comparison_df['baseline_mean_pagerank'], 
                          comparison_df['real_pagerank'],
                          alpha=0.5, s=10)
        # Add diagonal line
        max_val = max(comparison_df['baseline_mean_pagerank'].max(), 
                     comparison_df['real_pagerank'].max())
        axes[1, 0].plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='x=y')
        axes[1, 0].set_xlabel('Baseline Mean PageRank')
        axes[1, 0].set_ylabel('Real PageRank')
        axes[1, 0].set_title('Real vs Baseline PageRank')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Significance levels pie chart
        print("  - Significance levels pie chart...")
        sig_3 = (z_pr.abs() > 3).sum()
        sig_2_only = ((z_pr.abs() > 2) & (z_pr.abs() <= 3)).sum()
        not_sig = (z_pr.abs() <= 2).sum()
        
        sizes = [sig_3, sig_2_only, not_sig]
        labels = [f'|z| > 3\n({sig_3:,}, {sig_3/len(z_pr)*100:.1f}%)',
                 f'2 < |z| ≤ 3\n({sig_2_only:,}, {sig_2_only/len(z_pr)*100:.1f}%)',
                 f'|z| ≤ 2\n({not_sig:,}, {not_sig/len(z_pr)*100:.1f}%)']
        colors = ['darkred', 'orange', 'lightgray']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Statistical Significance Levels')
        
        plt.tight_layout()
        plot_file = PLOTS_DIR / "baseline_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved visualization to {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"\n⚠ Could not create visualizations: {e}")


def save_results(comparison_df):
    """Save all results to files"""
    print("\n" + "=" * 80)
    print("PHASE 8: SAVING RESULTS")
    print("=" * 80)
    
    # Save comparison dataframe
    comparison_file = RESULTS_DIR / "baseline_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)
    print(f"\n1. ✓ Saved: {comparison_file}")
    
    # Save significant packages
    sig_df = comparison_df[comparison_df['z_score_pagerank'].abs() > 2].copy()
    sig_df['significance'] = sig_df['z_score_pagerank'].apply(
        lambda z: 'more_important' if z > 0 else 'less_important'
    )
    sig_file = RESULTS_DIR / "significant_packages.csv"
    sig_df.to_csv(sig_file, index=False)
    print(f"2. ✓ Saved: {sig_file} ({len(sig_df):,} packages)")
    
    # Create summary report
    summary_file = RESULTS_DIR / "comparison_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OBJECTIVE 5: BASELINE VALIDATION - COMPARISON SUMMARY\n")
        f.write("PyPI Dependency Network - Network Science Project\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("Z-SCORE DISTRIBUTION (PageRank)\n")
        f.write("-" * 80 + "\n")
        z_pr = comparison_df['z_score_pagerank']
        f.write(f"Mean: {z_pr.mean():.2f}\n")
        f.write(f"Median: {z_pr.median():.2f}\n")
        f.write(f"Std Dev: {z_pr.std():.2f}\n")
        f.write(f"Min: {z_pr.min():.2f}\n")
        f.write(f"Max: {z_pr.max():.2f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("SIGNIFICANCE LEVELS\n")
        f.write("-" * 80 + "\n")
        sig_2 = (z_pr.abs() > 2).sum()
        sig_3 = (z_pr.abs() > 3).sum()
        total = len(z_pr)
        f.write(f"Packages with |z| > 2 (95% confidence): {sig_2:,} ({sig_2/total*100:.1f}%)\n")
        f.write(f"Packages with |z| > 3 (99.7% confidence): {sig_3:,} ({sig_3/total*100:.1f}%)\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("TOP 50 PACKAGES BY Z-SCORE\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n{'Rank':<6} {'Package':<35} {'Z-Score':<10} {'Real PR':<14} {'Base PR':<14}\n")
        f.write("-" * 80 + "\n")
        for idx, row in comparison_df.head(50).iterrows():
            f.write(f"{idx+1:<6} {row['package']:<35} {row['z_score_pagerank']:<10.2f} "
                   f"{row['real_pagerank']:<14.10f} {row['baseline_mean_pagerank']:<14.10f}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Betweenness Centrality Computed: {CONFIG['compute_betweenness']}\n")
        f.write(f"Parallel Processing: {CONFIG['use_parallel']}\n")
        if CONFIG['use_parallel']:
            f.write(f"Number of Workers: {CONFIG['n_jobs']}\n")
        
        f.write("\n" + "-" * 80 + "\n")
        f.write("INTERPRETATION\n")
        f.write("-" * 80 + "\n")
        f.write("\nPositive z-scores indicate packages that are MORE important than expected\n")
        f.write("based on degree distribution alone - these have genuine structural significance.\n")
        f.write("\nNegative z-scores indicate packages that are LESS important than expected\n")
        f.write("based on their degree - these may be structurally constrained.\n")
        
        if not CONFIG['compute_betweenness']:
            f.write("\nNote: Betweenness centrality was skipped for performance reasons.\n")
            f.write("To include it, set CONFIG['compute_betweenness'] = True in the script.\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"3. ✓ Saved: {summary_file}")
    print(f"\n✓ All results saved to {RESULTS_DIR}/")


def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("STEP 2: COMPUTE BASELINE CENTRALITY")
    print("Compare Original vs Random Graphs")
    print("=" * 80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nConfiguration:")
    print(f"  - Compute Betweenness: {CONFIG['compute_betweenness']}")
    print(f"  - Parallel Processing: {CONFIG['use_parallel']}")
    if CONFIG['use_parallel']:
        print(f"  - Number of Workers: {CONFIG['n_jobs']}")
    print()
    
    start_time = time.time()
    
    # Phase 1: Load original graph
    G = load_graph()
    
    # Phase 2: Load random graphs
    random_graphs = load_random_graphs()
    
    # Phase 3-4: Compute centrality and baseline
    original_centrality, baseline_stats = compute_baseline_centrality(G, random_graphs)
    
    # Phase 5: Compute z-scores
    z_scores = compute_z_scores(original_centrality, baseline_stats)
    
    # Create comparison dataframe
    comparison_df = create_comparison_dataframe(original_centrality, baseline_stats, z_scores)
    
    # Phase 6: Analyze
    analyze_results(comparison_df)
    
    # Phase 7: Visualizations
    create_visualizations(comparison_df)
    
    # Phase 8: Save results
    save_results(comparison_df)
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("BASELINE CENTRALITY ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"✓ Computed centrality for {G.number_of_nodes():,} packages")
    print(f"✓ Compared against {len(random_graphs)} random graphs")
    print(f"✓ Results saved to: {RESULTS_DIR}/")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
