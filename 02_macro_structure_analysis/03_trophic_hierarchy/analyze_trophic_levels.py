#!/usr/bin/env python3
"""
Objective 2 (Alternative): Trophic Level Analysis (The Hierarchy Pyramid)

This script analyzes the "Macro-Structure" of the software ecosystem by calculating
the vertical "Height" (Trophic Level) of every package in the Directed Acyclic Graph (DAG).

Outputs:
1. trophic_levels.json: Detailed stats, package lists per level, and basal fractions.
2. level_representatives.json: A mapping of Level -> Representative Package.
3. trophic_pyramid.png: A corrected visualization of the ecosystem's shape.
4. trophic_summary.csv: Tabular summary.

Inputs: pypi_dag/ (nodes.csv, edges.csv)
"""

import argparse
import json
import csv
import sys
import os
from pathlib import Path
from typing import Dict, List, Any

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trophic Level Analysis for Dependency DAGs")
    
    default_graph_dir = Path("pypi_dag")
    default_output_dir = Path("objective2_outputs")

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=default_graph_dir,
        help="Directory containing the DAG (nodes.csv, edges.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save plots and JSON reports"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# Graph Loading
# ---------------------------------------------------------------------------

def load_dag(graph_dir: Path) -> nx.DiGraph:
    """Loads the graph specifically from the cleaned DAG folder."""
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"

    if not nodes_path.exists() or not edges_path.exists():
        print(f"[ERROR] Could not find graph files in {graph_dir}")
        print("Please ensure you run the DAG conversion script first.")
        sys.exit(1)

    print(f"[INFO] Loading DAG from {graph_dir}...")
    
    # Load Edges
    df_edges = pd.read_csv(edges_path)
    if "source" not in df_edges.columns or "target" not in df_edges.columns:
        raise ValueError("edges.csv missing 'source' or 'target' columns")
    
    G = nx.DiGraph()
    # Add Edges (Source depends on Target)
    for _, row in df_edges.iterrows():
        src, dst = str(row['source']).strip(), str(row['target']).strip()
        if src and dst:
            G.add_edge(src, dst)
        
    # Load Nodes (to ensure isolated Level 0 nodes are included)
    df_nodes = pd.read_csv(nodes_path)
    all_nodes = set(df_nodes['package'].astype(str))
    G.add_nodes_from(all_nodes)
    
    print(f"[INFO] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # Verify DAG property
    if not nx.is_directed_acyclic_graph(G):
        print("[CRITICAL ERROR] The input graph is NOT a DAG. Trophic analysis requires a cycle-free graph.")
        print("Please check your cycle removal step.")
        sys.exit(1)
        
    return G

# ---------------------------------------------------------------------------
# Trophic Analysis Logic
# ---------------------------------------------------------------------------

def compute_trophic_levels(G: nx.DiGraph) -> Dict[str, int]:
    """
    Calculates the Trophic Level (Height) for every node.
    Level 0 = No outgoing edges (Depends on nothing).
    Level N = 1 + Max(Level of dependencies).
    """
    levels = {}
    
    print("[INFO] Computing Trophic Levels...")
    
    # Process in Reverse Topological Order (Dependencies first, then Dependents)
    try:
        processing_order = reversed(list(nx.topological_sort(G)))
    except nx.NetworkXUnfeasible:
        print("[ERROR] Cycle detected during topological sort. Graph is not a DAG.")
        sys.exit(1)

    for node in processing_order:
        dependencies = list(G.successors(node))
        
        if not dependencies:
            # Base Case: Depends on nothing (Level 0)
            levels[node] = 0
        else:
            # Recursive Step: 1 + Max height of dependencies
            levels[node] = 1 + max(levels[dep] for dep in dependencies)
            
    return levels

def get_representative_packages(G: nx.DiGraph, levels: Dict[str, int]) -> Dict[int, str]:
    """
    Identifies one 'Representative' package for each level.
    Heuristic: The node in that level with the highest In-Degree.
    """
    representatives = {}
    nodes_by_level = {}
    
    for node, level in levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
        
    for level, nodes in nodes_by_level.items():
        best_node = None
        max_degree = -1
        
        for node in nodes:
            in_degree = G.in_degree(node)
            if in_degree > max_degree:
                max_degree = in_degree
                best_node = node
        
        representatives[level] = best_node
        
    return representatives

# ---------------------------------------------------------------------------
# Visualizations & Outputs
# ---------------------------------------------------------------------------

def save_outputs(G: nx.DiGraph, levels: Dict[str, int], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    total_nodes = len(levels)
    max_depth = max(levels.values()) if levels else 0
    
    # 1. Prepare Data Structures
    counts = {}
    node_lists = {}
    for node, lvl in levels.items():
        counts[lvl] = counts.get(lvl, 0) + 1
        if lvl not in node_lists: 
            node_lists[lvl] = []
        node_lists[lvl].append(node)
        
    representatives = get_representative_packages(G, levels)
    
    # 2. Generate Main JSON Report (Comprehensive)
    report_data = {
        "summary": {
            "total_packages": total_nodes,
            "max_trophic_height": max_depth,
            "ecosystem_shape": "Pyramid" if counts.get(0,0) > counts.get(max_depth,0) else "Inverted/Column"
        },
        "level_statistics": []
    }
    
    for lvl in range(max_depth + 1):
        count = counts.get(lvl, 0)
        pct = (count / total_nodes) * 100
        rep_pkg = representatives.get(lvl, "None")
        
        level_info = {
            "level_id": lvl,
            "count": count,
            "percentage": pct,  # Raw float for precision
            "formatted_percentage": f"{pct:.2f}%",
            "representative_package": rep_pkg,
            "packages": sorted(node_lists.get(lvl, [])) # Full list of packages
        }
        report_data["level_statistics"].append(level_info)
        
    json_path = output_dir / "trophic_levels.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2)
    print(f"[INFO] Saved comprehensive JSON report to: {json_path}")

    # 3. Generate Representatives JSON (Specific Request)
    rep_data = {
        "description": "Representative package (highest in-degree) for each trophic level",
        "representatives": representatives
    }
    rep_path = output_dir / "level_representatives.json"
    with open(rep_path, 'w', encoding='utf-8') as f:
        json.dump(rep_data, f, indent=2)
    print(f"[INFO] Saved representatives JSON to: {rep_path}")
    
    # 4. Generate Pyramid Plot (Corrected)
    print(f"[INFO] Generating Trophic Pyramid Plot...")
    
    # Figure Setup: Increase size to handle text
    fig, ax = plt.subplots(figsize=(14, max(8, max_depth * 0.5)))
    
    levels_sorted = list(range(max_depth + 1))
    counts_sorted = [counts.get(l, 0) for l in levels_sorted]
    reps_sorted = [representatives.get(l, "") for l in levels_sorted]
    
    # Horizontal Bar Chart
    bars = ax.barh(levels_sorted, counts_sorted, color='#1f77b4', alpha=0.8, height=0.6)
    
    ax.set_title(f"Trophic Pyramid: Shape of the PyPI Dependency Network (Max Depth: {max_depth})", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Number of Packages (Log Scale)", fontsize=14)
    ax.set_ylabel("Trophic Level (Height)", fontsize=14)
    ax.set_xscale('log')
    
    # Fix Y-axis ticks to show integers only
    ax.set_yticks(levels_sorted)
    
    # --- TEXT LABELING CORRECTIONS ---
    
    # 1. Determine max width for Plot limits
    # Since it's log scale, we need a multiplicative buffer for text.
    max_count = max(counts_sorted) if counts_sorted else 1
    # Expand X-limit significantly to fit text on the right
    ax.set_xlim(right=max_count * 50) 
    
    for i, bar in enumerate(bars):
        count = counts_sorted[i]
        rep = reps_sorted[i]
        
        # Calculate precise percentage
        pct = (count / total_nodes) * 100
        
        # Format Percentage: Handle tiny values gracefully
        if pct == 0:
            pct_str = "0%"
        elif pct < 0.01:
            pct_str = "<0.01%"
        else:
            pct_str = f"{pct:.2f}%"
            
        # Label Text
        label_text = f" N={count} ({pct_str}) | Ex: {rep}"
        
        # Position: Place text slightly to the right of the bar end
        # On log scale, simple addition doesn't work well visually for small vs large bars,
        # but matplotlib handles 'x' coordinate in data units.
        # We add a small visual offset relative to the bar's width.
        text_x = max(count, 0.8) # Ensure text isn't off-screen left for count=0/1
        
        ax.text(
            text_x * 1.1,  # Place slightly to the right (multiplicative for log)
            i, 
            label_text, 
            va='center', 
            fontsize=10, 
            fontweight='bold', 
            color='#333333'
        )
        
    # Annotations for Context
    ax.text(0.5, -0.8, "Basal / Foundations", color='black', style='italic', ha='left')
    ax.text(0.5, max_depth + 0.2, "Top-Level Applications", color='black', style='italic', ha='left')

    plt.tight_layout()
    plot_path = output_dir / "trophic_pyramid.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved Corrected Pyramid Plot to: {plot_path}")
    
    # 5. Console Summary
    print("\n" + "="*70)
    print(" TROPHIC LEVEL ANALYSIS RESULTS")
    print("="*70)
    print(f"{'LEVEL':<6} | {'COUNT':<8} | {'%':<8} | {'REPRESENTATIVE'}")
    print("-" * 70)
    for lvl in range(min(max_depth + 1, 20)): 
        stats = report_data["level_statistics"][lvl]
        print(f"{lvl:<6} | {stats['count']:<8} | {stats['formatted_percentage']:<8} | {stats['representative_package']}")
    if max_depth > 20:
        print(f"... and {max_depth - 20} more levels.")
    print("="*70 + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    # 1. Load Graph
    try:
        G = load_dag(args.graph_dir)
        
        # 2. Compute Levels
        levels = compute_trophic_levels(G)
        
        # 3. Generate Outputs
        save_outputs(G, levels, args.output_dir)
        
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()