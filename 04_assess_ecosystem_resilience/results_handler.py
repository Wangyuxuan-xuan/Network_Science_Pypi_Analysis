"""
Results handling module for resilience analysis.
Handles result analysis, summary generation, and file saving.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

from config import (
    CONFIG,
    RESULTS_FILE,
    SUMMARY_FILE,
    RESULTS_DIR,
    PLOTS_DIR,
    METADATA_FILE
)


def analyze_results(results_df, baseline_metrics, centrality_df):
    """Analyze and display key dataset"""
    print("\n" + "=" * 80)
    print("PHASE 4: ANALYZING RESULTS")
    print("=" * 80)
    
    print("\n1. Network Fragmentation Analysis (with Cascade Failures):")
    print("   " + "-" * 100)
    print(f"   {'k':<6} {'Failed':<12} {'Cascade':<12} {'LWCC Size':<15} {'LWCC %':<10} {'Retention %':<12} {'Components':<12}")
    print("   " + "-" * 100)
    
    for _, row in results_df.iterrows():
        print(f"   {row['k']:<6} {row['nodes_failed_total']:<12,} {row['cascade_failures']:<12,} "
              f"{row['lwcc_size']:<15,} {row['lwcc_fraction']:<10.2%} "
              f"{row['lwcc_size_retention']:<12.2%} {row['weak_component_count']:<12,}")
    
    # Find critical thresholds
    print("\n2. Critical Thresholds:")
    
    # Find k where LWCC retention drops below 50%
    threshold_50 = results_df[results_df['lwcc_size_retention'] < 0.50]
    if len(threshold_50) > 0:
        k_50 = threshold_50.iloc[0]['k']
        print(f"   - 50% LWCC retention lost at k = {k_50}")
    else:
        print(f"   - 50% threshold not reached (network is resilient)")
    
    # Find k where LWCC retention drops below 10%
    threshold_10 = results_df[results_df['lwcc_size_retention'] < 0.10]
    if len(threshold_10) > 0:
        k_10 = threshold_10.iloc[0]['k']
        print(f"   - 10% LWCC retention lost at k = {k_10}")
    else:
        print(f"   - 10% threshold not reached")
    
    # Find k where network fragments significantly (component count > 100)
    threshold_frag = results_df[results_df['weak_component_count'] > 100]
    if len(threshold_frag) > 0:
        k_frag = threshold_frag.iloc[0]['k']
        print(f"   - Significant fragmentation (>100 components) at k = {k_frag}")
    else:
        print(f"   - Network remains relatively connected")
    
    # Top removed packages
    print("\n3. Most Critical Packages (Top 20 by removal impact):")
    print("   " + "-" * 76)
    metric = CONFIG['centrality_metric']
    top_20 = centrality_df.head(20)
    print(f"   {'Rank':<6} {'Package':<35} {'PageRank':<12} {'In-Degree':<10}")
    print("   " + "-" * 76)
    for idx, row in top_20.iterrows():
        print(f"   {idx+1:<6} {row['package']:<35} {row[metric]:<12.8f} {row['in_degree']:<10,}")


def save_results(results_df, baseline_metrics, centrality_df, metadata, null_model_df=None):
    """Save all dataset to files"""
    print("\n" + "=" * 80)
    print("PHASE 6: SAVING RESULTS")
    print("=" * 80)
    
    # Save dataset dataframe
    print(f"\n1. Saving dataset to {RESULTS_FILE}...")
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"   ✓ Saved {len(results_df)} experiment dataset")
    
    # Save null model dataset if available
    if null_model_df is not None:
        null_model_file = RESULTS_DIR / "null_model_results.csv"
        print(f"\n2. Saving null model dataset to {null_model_file}...")
        null_model_df.to_csv(null_model_file, index=False)
        print(f"   ✓ Saved averaged dataset from {CONFIG['num_random_graphs']} randomized graphs")
    
    # Create comprehensive summary
    summary_num = 3 if null_model_df is not None else 2
    print(f"\n{summary_num}. Creating summary report in {SUMMARY_FILE}...")
    with open(SUMMARY_FILE, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("OBJECTIVE 4: ECOSYSTEM RESILIENCE ASSESSMENT\n")
        f.write("PyPI Dependency Network - Network Science Project\n")
        f.write("=" * 80 + "\n")
        f.write(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {metadata['description']}\n")
        f.write(f"Dataset Timestamp: {metadata['timestamp_utc']}\n")
        
        # Methodology
        f.write("\n" + "-" * 80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("\nTargeted Node Removal Experiments with Cascade Failure Simulation:\n")
        f.write("- Removed top-k nodes identified by centrality analysis (Objective 1)\n")
        f.write(f"- Primary ranking metric: {CONFIG['centrality_metric']}\n")
        f.write(f"- Removal values tested: k = {CONFIG['removal_k_values']}\n")
        f.write("\nCascade Failure Model:\n")
        f.write("- When a package is removed, all packages that depend on it (directly or indirectly) also fail\n")
        f.write("- This simulates real-world scenarios where removing a critical package (e.g., numpy)\n")
        f.write("  causes all dependent packages to become non-functional\n")
        f.write("- Cascade failures are computed by finding all reachable nodes in the reverse dependency graph\n")
        f.write("\nNetwork Integrity Metrics:\n")
        f.write("- Total failed nodes (initial removal + cascade failures)\n")
        f.write("- Cascade failure count (additional failures beyond initial removal)\n")
        f.write("- Largest Weakly Connected Component (LWCC) size\n")
        f.write("- LWCC fraction of remaining nodes\n")
        f.write("- Number of weakly connected components\n")
        f.write("- Number of strongly connected components\n")
        f.write("- Network density\n")
        f.write("- Average shortest path length (when applicable)\n")
        
        # Baseline metrics
        f.write("\n" + "-" * 80 + "\n")
        f.write("BASELINE NETWORK METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total nodes: {baseline_metrics['nodes_remaining']:,}\n")
        f.write(f"Total edges: {baseline_metrics['edges_remaining']:,}\n")
        f.write(f"LWCC size: {baseline_metrics['lwcc_size']:,}\n")
        f.write(f"LWCC fraction: {baseline_metrics['lwcc_fraction']:.2%}\n")
        f.write(f"Weak components: {baseline_metrics['weak_component_count']}\n")
        f.write(f"Strong components: {baseline_metrics['strong_component_count']}\n")
        f.write(f"Density: {baseline_metrics['density']:.6f}\n")
        
        # Results summary
        f.write("\n" + "-" * 80 + "\n")
        f.write("REMOVAL EXPERIMENT RESULTS (WITH CASCADE FAILURES)\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n{'k':<6} {'Total Failed':<15} {'Cascade':<12} {'LWCC Size':<15} {'LWCC %':<10} {'Retention %':<12} {'Components':<12}\n")
        f.write("-" * 80 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['k']:<6} {row['nodes_failed_total']:<15,} {row['cascade_failures']:<12,} "
                   f"{row['lwcc_size']:<15,} {row['lwcc_fraction']:<10.2%} "
                   f"{row['lwcc_size_retention']:<12.2%} {row['weak_component_count']:<12,}\n")
        
        # Critical thresholds
        f.write("\n" + "-" * 80 + "\n")
        f.write("CRITICAL THRESHOLDS\n")
        f.write("-" * 80 + "\n")
        
        threshold_50 = results_df[results_df['lwcc_size_retention'] < 0.50]
        if len(threshold_50) > 0:
            k_50 = threshold_50.iloc[0]['k']
            row_50 = results_df[results_df['k'] == k_50].iloc[0]
            f.write(f"\n50% LWCC retention lost at k = {k_50}\n")
            f.write(f"  This means removing the top {k_50} most central packages\n")
            f.write(f"  (causing {row_50['nodes_failed_total']:,} total failures including {row_50['cascade_failures']:,} cascade failures)\n")
            f.write(f"  reduces the largest connected component to less than 50% of its original size.\n")
        else:
            f.write("\n50% threshold not reached - network shows high resilience\n")
        
        threshold_10 = results_df[results_df['lwcc_size_retention'] < 0.10]
        if len(threshold_10) > 0:
            k_10 = threshold_10.iloc[0]['k']
            row_10 = results_df[results_df['k'] == k_10].iloc[0]
            f.write(f"\n10% LWCC retention lost at k = {k_10}\n")
            f.write(f"  Severe fragmentation occurs after removing top {k_10} packages.\n")
            f.write(f"  Total failures: {row_10['nodes_failed_total']:,} (including {row_10['cascade_failures']:,} cascade failures)\n")
        else:
            f.write("\n10% threshold not reached\n")
        
        threshold_frag = results_df[results_df['weak_component_count'] > 100]
        if len(threshold_frag) > 0:
            k_frag = threshold_frag.iloc[0]['k']
            f.write(f"\nSignificant fragmentation (>100 components) at k = {k_frag}\n")
        else:
            f.write("\nNetwork remains relatively connected throughout experiments\n")
        
        # Top critical packages
        f.write("\n" + "-" * 80 + "\n")
        f.write("TOP 20 MOST CRITICAL PACKAGES\n")
        f.write("-" * 80 + "\n")
        metric = CONFIG['centrality_metric']
        top_20 = centrality_df.head(20)
        f.write(f"\n{'Rank':<6} {'Package':<35} {'PageRank':<14} {'In-Degree':<10}\n")
        f.write("-" * 80 + "\n")
        for idx, row in top_20.iterrows():
            f.write(f"{idx+1:<6} {row['package']:<35} {row[metric]:<14.10f} {row['in_degree']:<10,}\n")
        
        # Key findings
        f.write("\n" + "-" * 80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        # Calculate resilience score (area under retention curve)
        if len(results_df) > 1:
            # Approximate area under curve using trapezoidal rule
            x = results_df['k'].values
            y = results_df['lwcc_size_retention'].values
            area = np.trapz(y, x) / (x[-1] - x[0])
            f.write(f"\n1. Overall Resilience Score: {area:.3f}\n")
            f.write(f"   (Area under retention curve, normalized; higher = more resilient)\n")
        
        # Find most impactful single removal
        if len(results_df) > 0:
            first_removal = results_df.iloc[0]
            f.write(f"\n2. Impact of Removing Top Package (with Cascade Failures):\n")
            f.write(f"   - Initial removal: 1 package\n")
            f.write(f"   - Total failed (cascade): {first_removal['nodes_failed_total']:,} packages\n")
            f.write(f"   - Cascade failures: {first_removal['cascade_failures']:,} additional packages\n")
            f.write(f"   - LWCC retention: {first_removal['lwcc_size_retention']:.2%}\n")
            f.write(f"   - Components created: {first_removal['weak_component_count']}\n")
            top_package = centrality_df.iloc[0]['package']
            f.write(f"   - Package: {top_package}\n")
            f.write(f"   - Note: Removing {top_package} causes {first_removal['cascade_failures']:,} dependent packages to fail\n")
        
        # Fragmentation pattern
        f.write(f"\n3. Fragmentation Pattern:\n")
        if len(results_df) > 1:
            # Check if fragmentation is gradual or sudden
            retention_drops = results_df['lwcc_size_retention'].diff().abs()
            max_drop = retention_drops.max()
            max_drop_k = results_df.loc[retention_drops.idxmax(), 'k']
            f.write(f"   - Maximum single-step retention drop: {max_drop:.2%} at k = {max_drop_k}\n")
            if max_drop > 0.10:
                f.write(f"   - Pattern: Sudden fragmentation (cascade failure risk)\n")
            else:
                f.write(f"   - Pattern: Gradual degradation (more resilient)\n")
        
        # Null model comparison (if available)
        if null_model_df is not None and len(null_model_df) > 0:
            f.write("\n" + "-" * 80 + "\n")
            f.write("NULL MODEL COMPARISON (OBJECTIVE 5 BASELINE)\n")
            f.write("-" * 80 + "\n")
            f.write(f"\nTo validate that resilience findings are non-trivial, we compared the original network\n")
            f.write(f"against {CONFIG['num_random_graphs']} randomized null models.\n")
            f.write(f"\nRandomized Graph Properties:\n")
            f.write(f"- Preserve: degree sequence (in-degree and out-degree)\n")
            f.write(f"- Preserve: DAG property (no cycles)\n")
            f.write(f"- Preserve: topological order (depth/level hierarchy)\n")
            f.write(f"- Randomize: edge connections (order-preserving edge swaps)\n")
            f.write(f"\nComparison Methodology:\n")
            f.write(f"- Same removal strategy: top-k nodes by PageRank centrality\n")
            f.write(f"- Same cascade failure model: all dependent packages fail\n")
            f.write(f"- Same metrics: LWCC relative size (normalized by original graph size)\n")
            f.write(f"- Averaged dataset across {CONFIG['num_random_graphs']} randomized graphs\n")
            f.write(f"\nKey Comparison Results:\n")
            if len(null_model_df) > 0:
                # Compare first few k values
                for k in [1, 5, 10, 20]:
                    orig_row = results_df[results_df['k'] == k]
                    null_row = null_model_df[null_model_df['k'] == k]
                    if len(orig_row) > 0 and len(null_row) > 0:
                        orig_lwcc = orig_row.iloc[0]['lwcc_relative_size'] * 100
                        null_lwcc = null_row.iloc[0]['lwcc_relative_size'] * 100
                        diff = orig_lwcc - null_lwcc
                        f.write(f"  - k={k}: Original={orig_lwcc:.2f}%, Null Model={null_lwcc:.2f}%, Difference={diff:+.2f}%\n")
            f.write(f"\nInterpretation:\n")
            f.write(f"- If original network shows similar resilience to null model: findings may be\n")
            f.write(f"  explained by general network topology (degree distribution, DAG structure)\n")
            f.write(f"- If original network shows different resilience: findings reflect specific\n")
            f.write(f"  structural properties of the PyPI dependency ecosystem\n")
        
        # File outputs
        f.write("\n" + "-" * 80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("-" * 80 + "\n")
        f.write(f"\n1. {RESULTS_FILE}\n")
        f.write(f"   - Complete dataset for all removal experiments\n")
        if null_model_df is not None:
            f.write(f"\n2. {RESULTS_DIR.name}/null_model_results.csv\n")
            f.write(f"   - Averaged dataset from randomized graphs\n")
        f.write(f"\n{summary_num if null_model_df is None else summary_num + 1}. {SUMMARY_FILE.name} (this file)\n")
        f.write(f"   - Comprehensive analysis summary\n")
        f.write(f"\n{summary_num + 1 if null_model_df is None else summary_num + 2}. {PLOTS_DIR.name}/resilience_analysis.png\n")
        f.write(f"   - Visualization plots\n")
        if null_model_df is not None:
            f.write(f"\n{summary_num + 2}. {PLOTS_DIR.name}/resilience_null_model_comparison.png\n")
            f.write(f"   - Null model comparison plot (original vs randomized networks)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"   ✓ Comprehensive summary saved")
    print(f"\n   Summary includes:")
    print(f"   - Methodology and metrics")
    print(f"   - Baseline network properties")
    print(f"   - Detailed removal experiment dataset")
    print(f"   - Critical thresholds and key findings")

