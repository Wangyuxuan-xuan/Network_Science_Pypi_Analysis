"""
Objective 1: Identify Core Packages Using Centrality Analysis
PyPI Dependency Network - Network Science Project

This script analyzes the PyPI dependency DAG to identify core packages
using various centrality measures, with PageRank as the primary metric.
"""

import pandas as pd
import networkx as nx
import numpy as np
import json
import pickle
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

# Set up paths
DATA_DIR = Path("data")
EDGES_FILE = DATA_DIR / "edges.csv"
NODES_FILE = DATA_DIR / "nodes.csv"
METADATA_FILE = DATA_DIR / "metadata.json"
GRAPH_CACHE_FILE = DATA_DIR / "graph_cache.pkl"

# Output files
RESULTS_DIR = Path("results/centrality")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "centrality_results.csv"
TOP_PACKAGES_FILE = RESULTS_DIR / "top_packages.csv"
SUMMARY_FILE = RESULTS_DIR / "centrality_summary.txt"
PLOTS_DIR = RESULTS_DIR
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def log_message(message, file=None):
    """Print and optionally write message to file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    if file:
        file.write(formatted_msg + "\n")


def load_data():
    """Load dataset files and metadata"""
    print("=" * 80)
    print("PHASE 1: DATA LOADING & GRAPH CONSTRUCTION")
    print("=" * 80)
    
    # Load metadata
    print("\n1. Loading metadata...")
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    print(f"   Dataset: {metadata['description']}")
    print(f"   Timestamp: {metadata['timestamp_utc']}")
    print(f"   Total nodes: {metadata['metrics']['final_node_count']:,}")
    print(f"   Total edges: {metadata['metrics']['final_edge_count']:,}")
    print(f"   Cyclic nodes removed: {metadata['metrics']['removed_cyclic_nodes']:,}")
    
    # Load nodes
    print("\n2. Loading nodes from nodes.csv...")
    nodes_df = pd.read_csv(NODES_FILE)
    print(f"   Loaded {len(nodes_df):,} nodes")
    print(f"   Columns: {list(nodes_df.columns)}")
    
    # Load edges
    print("\n3. Loading edges from edges.csv...")
    edges_df = pd.read_csv(EDGES_FILE)
    print(f"   Loaded {len(edges_df):,} edges")
    print(f"   Columns: {list(edges_df.columns)}")
    print(f"   Edge format: {edges_df.columns[0]} -> {edges_df.columns[1]}")
    print(f"   (Interpretation: source package DEPENDS ON target package)")
    
    # Note about caching
    if GRAPH_CACHE_FILE.exists():
        cache_time = datetime.fromtimestamp(GRAPH_CACHE_FILE.stat().st_mtime)
        print(f"\n   Note: Graph cache exists (created: {cache_time.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"         To rebuild graph, delete: {GRAPH_CACHE_FILE}")
    
    return metadata, nodes_df, edges_df


def build_graph(edges_df):
    """Build NetworkX directed graph from edges with caching"""
    
    # Check if cached graph exists
    if GRAPH_CACHE_FILE.exists():
        print("\n4. Loading graph from cache...")
        try:
            with open(GRAPH_CACHE_FILE, 'rb') as f:
                G = pickle.load(f)
            print(f"   ✓ Graph loaded from cache: {GRAPH_CACHE_FILE}")
            print(f"   - Nodes: {G.number_of_nodes():,}")
            print(f"   - Edges: {G.number_of_edges():,}")
            print(f"   - Density: {nx.density(G):.6f}")
            return G
        except Exception as e:
            print(f"   ⚠ Cache loading failed: {e}")
            print("   Building graph from scratch...")
    else:
        print("\n4. Building NetworkX DiGraph (this may take a while)...")
    
    G = nx.DiGraph()
    
    # Add edges from dataframe with progress bar
    print("   Adding edges to graph...")
    edges = [(row['source'], row['target']) for _, row in tqdm(edges_df.iterrows(), 
                                                                total=len(edges_df),
                                                                desc="   Processing edges")]
    G.add_edges_from(edges)
    
    print(f"   Graph constructed:")
    print(f"   - Nodes: {G.number_of_nodes():,}")
    print(f"   - Edges: {G.number_of_edges():,}")
    print(f"   - Density: {nx.density(G):.6f}")
    
    # Save to cache
    print(f"\n   Saving graph to cache: {GRAPH_CACHE_FILE}")
    try:
        with open(GRAPH_CACHE_FILE, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"   ✓ Graph cached successfully")
    except Exception as e:
        print(f"   ⚠ Cache saving failed: {e}")
    
    return G


def validate_graph(G):
    """Validate graph properties"""
    print("\n5. Validating graph properties...")
    
    # Check if DAG
    is_dag = nx.is_directed_acyclic_graph(G)
    print(f"   Is DAG (acyclic): {is_dag}")
    
    if not is_dag:
        print("   WARNING: Graph contains cycles!")
        cycles = list(nx.simple_cycles(G))
        print(f"   Number of cycles found: {len(cycles)}")
    
    # Connected components
    weak_components = list(nx.weakly_connected_components(G))
    strong_components = list(nx.strongly_connected_components(G))
    
    print(f"   Weakly connected components: {len(weak_components)}")
    print(f"   - Largest WCC size: {len(max(weak_components, key=len)):,} nodes")
    print(f"   Strongly connected components: {len(strong_components)}")
    print(f"   - Largest SCC size: {len(max(strong_components, key=len)):,} nodes")
    
    # Degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    print(f"\n   In-degree statistics:")
    print(f"   - Mean: {np.mean(in_degrees):.2f}")
    print(f"   - Median: {np.median(in_degrees):.2f}")
    print(f"   - Max: {np.max(in_degrees)}")
    print(f"   - Nodes with in-degree = 0 (sources): {sum(1 for d in in_degrees if d == 0):,}")
    
    print(f"\n   Out-degree statistics:")
    print(f"   - Mean: {np.mean(out_degrees):.2f}")
    print(f"   - Median: {np.median(out_degrees):.2f}")
    print(f"   - Max: {np.max(out_degrees)}")
    print(f"   - Nodes with out-degree = 0 (sinks): {sum(1 for d in out_degrees if d == 0):,}")
    
    return is_dag


def compute_centrality_measures(G):
    """Compute various centrality measures"""
    print("\n" + "=" * 80)
    print("PHASE 2: CENTRALITY ANALYSIS")
    print("=" * 80)
    
    centrality_results = {}
    
    # PageRank (PRIMARY METRIC)
    print("\n1. Computing PageRank (primary metric)...")
    print("   Parameters: alpha=0.85, max_iter=100, tol=1e-06")
    print("   Note: Higher PageRank = more important based on transitive dependencies")
    start_time = time.time()
    pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
    elapsed = time.time() - start_time
    centrality_results['pagerank'] = pagerank
    print(f"   ✓ PageRank computed for {len(pagerank):,} nodes in {elapsed:.1f} seconds")
    print(f"   - Mean: {np.mean(list(pagerank.values())):.8f}")
    print(f"   - Max: {np.max(list(pagerank.values())):.8f}")
    print(f"   - Min: {np.min(list(pagerank.values())):.8f}")
    
    # In-degree centrality
    print("\n2. Computing In-Degree Centrality...")
    print("   Note: Measures direct dependents (how many packages depend on this one)")
    in_degree_cent = nx.in_degree_centrality(G)
    centrality_results['in_degree_centrality'] = in_degree_cent
    print(f"   ✓ In-degree centrality computed for {len(in_degree_cent):,} nodes")
    print(f"   - Mean: {np.mean(list(in_degree_cent.values())):.8f}")
    print(f"   - Max: {np.max(list(in_degree_cent.values())):.8f}")
    
    # Out-degree centrality
    print("\n3. Computing Out-Degree Centrality...")
    print("   Note: Measures number of dependencies this package has")
    out_degree_cent = nx.out_degree_centrality(G)
    centrality_results['out_degree_centrality'] = out_degree_cent
    print(f"   ✓ Out-degree centrality computed for {len(out_degree_cent):,} nodes")
    print(f"   - Mean: {np.mean(list(out_degree_cent.values())):.8f}")
    print(f"   - Max: {np.max(list(out_degree_cent.values())):.8f}")
    
    # Eigenvector centrality (may not converge for all nodes)
    print("\n4. Computing Eigenvector Centrality...")
    print("   Note: Similar to PageRank, measures influence in the network")
    try:
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=200, tol=1e-6)
        centrality_results['eigenvector_centrality'] = eigenvector_cent
        print(f"   ✓ Eigenvector centrality computed for {len(eigenvector_cent):,} nodes")
        print(f"   - Mean: {np.mean(list(eigenvector_cent.values())):.8f}")
        print(f"   - Max: {np.max(list(eigenvector_cent.values())):.8f}")
    except nx.PowerIterationFailedConvergence:
        print("   ⚠ Eigenvector centrality did not converge - skipping")
        eigenvector_cent = {node: 0.0 for node in G.nodes()}
        centrality_results['eigenvector_centrality'] = eigenvector_cent
    
    # Betweenness centrality (computationally expensive - using smaller sample)
    print("\n5. Computing Betweenness Centrality (approximate)...")
    print("   Note: Identifies 'bridge' packages - using smaller sample for efficiency")
    
    # Use much smaller k-sample for large graphs (betweenness is O(k*n*m))
    # For ~400K nodes and ~1.8M edges, even k=1000 is expensive
    k_sample = min(5000, G.number_of_nodes())
    
    print(f"   Computing shortest paths for {k_sample:,} sampled nodes...")
    print("   This may take 5-10 minutes for large graphs...")
    
    start_time = time.time()
    
    try:
        # NetworkX doesn't provide progress callbacks, so we just run it
        betweenness_cent = nx.betweenness_centrality(G, k=k_sample, normalized=True, seed=42)
        
        elapsed_time = time.time() - start_time
        centrality_results['betweenness_centrality'] = betweenness_cent
        print(f"   ✓ Betweenness centrality computed for {len(betweenness_cent):,} nodes")
        print(f"   - Computation time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"   - Mean: {np.mean(list(betweenness_cent.values())):.8f}")
        print(f"   - Max: {np.max(list(betweenness_cent.values())):.8f}")
    except Exception as e:
        print(f"   ⚠ Betweenness computation failed or timed out: {e}")
        print("   Skipping betweenness centrality (set all values to 0.0)")
        betweenness_cent = {node: 0.0 for node in G.nodes()}
        centrality_results['betweenness_centrality'] = betweenness_cent
    
    return centrality_results


def create_results_dataframe(G, centrality_results):
    """Create comprehensive results dataframe"""
    print("\n" + "=" * 80)
    print("PHASE 3: RESULTS ANALYSIS")
    print("=" * 80)
    
    print("\n1. Creating results dataframe...")
    
    # Create base dataframe with package names
    results_df = pd.DataFrame({
        'package': list(G.nodes())
    })
    
    # Add all centrality measures
    for measure_name, measure_dict in centrality_results.items():
        results_df[measure_name] = results_df['package'].map(measure_dict)
    
    # Add raw degree counts
    results_df['in_degree'] = results_df['package'].map(dict(G.in_degree()))
    results_df['out_degree'] = results_df['package'].map(dict(G.out_degree()))
    results_df['total_degree'] = results_df['in_degree'] + results_df['out_degree']
    
    # Sort by PageRank (descending)
    results_df = results_df.sort_values('pagerank', ascending=False).reset_index(drop=True)
    
    # Add rank column
    results_df.insert(0, 'rank', range(1, len(results_df) + 1))
    
    print(f"   ✓ Results dataframe created with {len(results_df):,} packages")
    print(f"   Columns: {list(results_df.columns)}")
    
    return results_df


def analyze_results(results_df):
    """Analyze and display key results"""
    print("\n2. Analyzing results...")
    
    # Top packages by PageRank
    print("\n   Top 20 Packages by PageRank:")
    print("   " + "-" * 76)
    print(f"   {'Rank':<6} {'Package':<35} {'PageRank':<12} {'In-Deg':<8} {'Out-Deg':<8}")
    print("   " + "-" * 76)
    for idx, row in results_df.head(20).iterrows():
        print(f"   {row['rank']:<6} {row['package']:<35} {row['pagerank']:<12.8f} {row['in_degree']:<8} {row['out_degree']:<8}")
    
    # Statistics
    print("\n   Centrality Score Distributions:")
    print("   " + "-" * 76)
    
    for metric in ['pagerank', 'in_degree_centrality', 'betweenness_centrality']:
        values = results_df[metric].values
        print(f"\n   {metric.upper()}:")
        print(f"     Mean:   {np.mean(values):.8f}")
        print(f"     Median: {np.median(values):.8f}")
        print(f"     Std:    {np.std(values):.8f}")
        print(f"     Min:    {np.min(values):.8f}")
        print(f"     Max:    {np.max(values):.8f}")
        
        # Percentiles
        percentiles = [50, 75, 90, 95, 99]
        print(f"     Percentiles:", end="")
        for p in percentiles:
            print(f" {p}th={np.percentile(values, p):.8f}", end="")
        print()
    
    # Correlation analysis
    print("\n   Correlation between centrality measures:")
    print("   " + "-" * 76)
    metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality', 
               'eigenvector_centrality', 'betweenness_centrality']
    corr_matrix = results_df[metrics].corr()
    print(corr_matrix.to_string())
    
    return


def save_results(results_df, metadata, G, is_dag, centrality_results):
    """Save results to files"""
    print("\n" + "=" * 80)
    print("PHASE 4: SAVING RESULTS")
    print("=" * 80)
    
    # Save full results
    print(f"\n1. Saving full results to {RESULTS_FILE}...")
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"   ✓ Saved {len(results_df):,} packages with centrality scores")
    
    # Save top packages
    print(f"\n2. Saving top 100 packages to {TOP_PACKAGES_FILE}...")
    top_df = results_df.head(100)
    top_df.to_csv(TOP_PACKAGES_FILE, index=False)
    print(f"   ✓ Saved top 100 packages")
    
    # Create comprehensive summary
    print(f"\n3. Creating comprehensive summary in {SUMMARY_FILE}...")
    with open(SUMMARY_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OBJECTIVE 1: CORE PACKAGES IDENTIFICATION - CENTRALITY ANALYSIS\n")
        f.write("PyPI Dependency Network - Network Science Project\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {metadata['description']}\n")
        f.write(f"Dataset Timestamp: {metadata['timestamp_utc']}\n")
        
        # Dataset information
        f.write("\n" + "-" * 80 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total nodes in dataset: {metadata['metrics']['final_node_count']:,}\n")
        f.write(f"Total edges in dataset: {metadata['metrics']['final_edge_count']:,}\n")
        f.write(f"Cyclic nodes removed: {metadata['metrics']['removed_cyclic_nodes']:,}\n")
        f.write(f"\nActive nodes in graph: {G.number_of_nodes():,}\n")
        f.write(f"Active edges in graph: {G.number_of_edges():,}\n")
        f.write(f"Graph density: {nx.density(G):.6f}\n")
        f.write(f"Is DAG (acyclic): {is_dag}\n")
        
        # Connected components
        weak_components = list(nx.weakly_connected_components(G))
        strong_components = list(nx.strongly_connected_components(G))
        f.write(f"\nWeakly connected components: {len(weak_components)}\n")
        f.write(f"  Largest WCC size: {len(max(weak_components, key=len)):,} nodes\n")
        f.write(f"Strongly connected components: {len(strong_components)}\n")
        f.write(f"  Largest SCC size: {len(max(strong_components, key=len)):,} nodes\n")
        
        # Degree statistics
        in_degrees = [d for n, d in G.in_degree()]
        out_degrees = [d for n, d in G.out_degree()]
        f.write(f"\nIn-degree statistics:\n")
        f.write(f"  Mean: {np.mean(in_degrees):.2f}\n")
        f.write(f"  Median: {np.median(in_degrees):.2f}\n")
        f.write(f"  Max: {np.max(in_degrees)}\n")
        f.write(f"  Nodes with in-degree = 0 (sources): {sum(1 for d in in_degrees if d == 0):,}\n")
        f.write(f"\nOut-degree statistics:\n")
        f.write(f"  Mean: {np.mean(out_degrees):.2f}\n")
        f.write(f"  Median: {np.median(out_degrees):.2f}\n")
        f.write(f"  Max: {np.max(out_degrees)}\n")
        f.write(f"  Nodes with out-degree = 0 (sinks): {sum(1 for d in out_degrees if d == 0):,}\n")
        
        # Centrality analysis details
        f.write("\n" + "-" * 80 + "\n")
        f.write("CENTRALITY ANALYSIS DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write("\nCentrality Measures Computed:\n")
        f.write("1. PageRank (PRIMARY METRIC)\n")
        f.write("   - Parameters: alpha=0.85, max_iter=100, tol=1e-06\n")
        f.write("   - Interpretation: Transitive importance in dependency graph\n")
        f.write("   - Higher score = more fundamental/critical package\n")
        f.write("\n2. In-Degree Centrality\n")
        f.write("   - Measures: Number of direct dependents (normalized)\n")
        f.write("   - Interpretation: Immediate ecosystem impact\n")
        f.write("\n3. Out-Degree Centrality\n")
        f.write("   - Measures: Number of dependencies (normalized)\n")
        f.write("   - Interpretation: Package complexity/coupling\n")
        f.write("\n4. Eigenvector Centrality\n")
        f.write("   - Measures: Influence based on neighbor importance\n")
        f.write("   - Similar to PageRank but undamped\n")
        f.write("\n5. Betweenness Centrality (approximate, k=5000)\n")
        f.write("   - Measures: How often package lies on shortest paths\n")
        f.write("   - Interpretation: Bridge packages connecting communities\n")
        
        # Statistics for each measure
        f.write("\n" + "-" * 80 + "\n")
        f.write("CENTRALITY SCORE DISTRIBUTIONS\n")
        f.write("-" * 80 + "\n")
        
        for metric in ['pagerank', 'in_degree_centrality', 'out_degree_centrality',
                       'eigenvector_centrality', 'betweenness_centrality']:
            values = results_df[metric].values
            f.write(f"\n{metric.upper().replace('_', ' ')}:\n")
            f.write(f"  Mean:       {np.mean(values):.10f}\n")
            f.write(f"  Median:     {np.median(values):.10f}\n")
            f.write(f"  Std Dev:    {np.std(values):.10f}\n")
            f.write(f"  Min:        {np.min(values):.10f}\n")
            f.write(f"  Max:        {np.max(values):.10f}\n")
            f.write(f"  Percentiles:\n")
            for p in [25, 50, 75, 90, 95, 99]:
                f.write(f"    {p}th: {np.percentile(values, p):.10f}\n")
        
        # Top packages
        f.write("\n" + "-" * 80 + "\n")
        f.write("TOP 50 CORE PACKAGES (by PageRank)\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n{'Rank':<6} {'Package':<35} {'PageRank':<14} {'In-Deg':<8} {'Out-Deg':<9} {'In-Deg-Cent':<13}\n")
        f.write("-" * 80 + "\n")
        for idx, row in results_df.head(50).iterrows():
            f.write(f"{row['rank']:<6} {row['package']:<35} {row['pagerank']:<14.10f} "
                   f"{row['in_degree']:<8} {row['out_degree']:<9} {row['in_degree_centrality']:<13.10f}\n")
        
        # Correlation matrix
        f.write("\n" + "-" * 80 + "\n")
        f.write("CORRELATION BETWEEN CENTRALITY MEASURES\n")
        f.write("-" * 80 + "\n")
        metrics = ['pagerank', 'in_degree_centrality', 'out_degree_centrality',
                   'eigenvector_centrality', 'betweenness_centrality']
        corr_matrix = results_df[metrics].corr()
        f.write("\n" + corr_matrix.to_string() + "\n")
        
        # Key findings
        f.write("\n" + "-" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Most central package
        top_package = results_df.iloc[0]
        f.write(f"\n1. Most Central Package (by PageRank):\n")
        f.write(f"   Package: {top_package['package']}\n")
        f.write(f"   PageRank: {top_package['pagerank']:.10f}\n")
        f.write(f"   In-degree: {top_package['in_degree']} packages depend on it\n")
        f.write(f"   Out-degree: {top_package['out_degree']} dependencies\n")
        
        # Highest in-degree
        highest_indeg = results_df.loc[results_df['in_degree'].idxmax()]
        f.write(f"\n2. Most Depended-Upon Package (by in-degree):\n")
        f.write(f"   Package: {highest_indeg['package']}\n")
        f.write(f"   In-degree: {highest_indeg['in_degree']} direct dependents\n")
        f.write(f"   PageRank: {highest_indeg['pagerank']:.10f} (Rank: {highest_indeg['rank']})\n")
        
        # Highest betweenness
        highest_between = results_df.loc[results_df['betweenness_centrality'].idxmax()]
        f.write(f"\n3. Most Critical Bridge Package (by betweenness):\n")
        f.write(f"   Package: {highest_between['package']}\n")
        f.write(f"   Betweenness: {highest_between['betweenness_centrality']:.10f}\n")
        f.write(f"   PageRank: {highest_between['pagerank']:.10f} (Rank: {highest_between['rank']})\n")
        
        # Distribution insights
        top10_avg_pr = results_df.head(10)['pagerank'].mean()
        top100_avg_pr = results_df.head(100)['pagerank'].mean()
        f.write(f"\n4. PageRank Concentration:\n")
        f.write(f"   Top 10 packages avg PageRank: {top10_avg_pr:.10f}\n")
        f.write(f"   Top 100 packages avg PageRank: {top100_avg_pr:.10f}\n")
        f.write(f"   Overall avg PageRank: {results_df['pagerank'].mean():.10f}\n")
        f.write(f"   Ratio (Top 10 / Overall): {top10_avg_pr / results_df['pagerank'].mean():.2f}x\n")
        
        # File outputs
        f.write("\n" + "-" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n1. {RESULTS_FILE}\n")
        f.write(f"   - Complete results for all {len(results_df):,} packages\n")
        f.write(f"   - All centrality measures included\n")
        f.write(f"\n2. {TOP_PACKAGES_FILE}\n")
        f.write(f"   - Top 100 packages by PageRank\n")
        f.write(f"\n3. {SUMMARY_FILE.name} (this file)\n")
        f.write(f"   - Comprehensive analysis summary\n")
        f.write(f"\n4. {PLOTS_DIR.name}/ directory\n")
        f.write(f"   - Visualization plots (if generated)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Comprehensive summary saved")
    print(f"\n   Summary includes:")
    print(f"   - Dataset information and graph properties")
    print(f"   - Centrality analysis methodology")
    print(f"   - Distribution statistics for all measures")
    print(f"   - Top 50 core packages with detailed metrics")
    print(f"   - Correlation analysis between measures")
    print(f"   - Key findings and insights")


def create_visualizations(results_df):
    """Create visualization plots"""
    print("\n4. Creating visualizations...")
    
    try:
        # Set style
        sns.set_style("whitegrid")
        
        # 1. PageRank distribution (log scale)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PageRank distribution
        axes[0, 0].hist(np.log10(results_df['pagerank']), bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('log10(PageRank)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('PageRank Distribution (log scale)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top 20 packages bar chart
        top20 = results_df.head(20)
        axes[0, 1].barh(range(20), top20['pagerank'], color='steelblue')
        axes[0, 1].set_yticks(range(20))
        axes[0, 1].set_yticklabels(top20['package'], fontsize=8)
        axes[0, 1].set_xlabel('PageRank')
        axes[0, 1].set_title('Top 20 Packages by PageRank')
        axes[0, 1].invert_yaxis()
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # In-degree vs PageRank scatter
        axes[1, 0].scatter(results_df['in_degree'], results_df['pagerank'], 
                          alpha=0.5, s=10)
        axes[1, 0].set_xlabel('In-Degree')
        axes[1, 0].set_ylabel('PageRank')
        axes[1, 0].set_title('PageRank vs In-Degree')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Centrality measures comparison (top 50)
        top50 = results_df.head(50)
        x = range(len(top50))
        axes[1, 1].plot(x, top50['pagerank'] / top50['pagerank'].max(), 
                       label='PageRank', marker='o', markersize=3)
        axes[1, 1].plot(x, top50['in_degree_centrality'] / top50['in_degree_centrality'].max(), 
                       label='In-Degree Cent.', marker='s', markersize=3)
        axes[1, 1].plot(x, top50['betweenness_centrality'] / top50['betweenness_centrality'].max(), 
                       label='Betweenness Cent.', marker='^', markersize=3)
        axes[1, 1].set_xlabel('Rank')
        axes[1, 1].set_ylabel('Normalized Centrality')
        axes[1, 1].set_title('Centrality Measures Comparison (Top 50, Normalized)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = PLOTS_DIR / "centrality_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved visualization to {plot_file}")
        plt.close()
        
    except Exception as e:
        print(f"   ⚠ Could not create visualizations: {e}")


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("OBJECTIVE 1: IDENTIFY CORE PACKAGES USING CENTRALITY ANALYSIS")
    print("PyPI Dependency Network - Network Science Project")
    print("=" * 80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Phase 1: Load data and build graph
    metadata, nodes_df, edges_df = load_data()
    G = build_graph(edges_df)
    is_dag = validate_graph(G)
    
    # Phase 2: Compute centrality measures
    centrality_results = compute_centrality_measures(G)
    
    # Phase 3: Create and analyze results
    results_df = create_results_dataframe(G, centrality_results)
    analyze_results(results_df)
    
    # Phase 4: Save results
    save_results(results_df, metadata, G, is_dag, centrality_results)
    create_visualizations(results_df)
    
    # Final summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nExecution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ Analyzed {G.number_of_nodes():,} packages with {G.number_of_edges():,} dependencies")
    print(f"✓ Computed 5 centrality measures")
    print(f"✓ Results saved to: {RESULTS_FILE}, {TOP_PACKAGES_FILE}, {SUMMARY_FILE}")
    print(f"\nTop 3 Core Packages by PageRank:")
    for idx, row in results_df.head(3).iterrows():
        print(f"  {row['rank']}. {row['package']} (PageRank: {row['pagerank']:.8f})")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
