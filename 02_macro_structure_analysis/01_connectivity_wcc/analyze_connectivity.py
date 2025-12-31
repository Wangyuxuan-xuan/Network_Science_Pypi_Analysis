#!/usr/bin/env python3
"""
Objective 2 Extension: Analyze Weakly Connected Components (Connectivity).

This script complements the SCC and Trophic analysis by examining the 
fragmentation of the ecosystem. It treats the dependency graph as 
undirected to identify:
  1) The Giant Connected Component (GCC) - The "Continent".
  2) Isolated components (Islands).
  3) The size distribution of these components (checking for Power Laws).

Outputs:
  - CSV summary of all connected components.
  - JSON report of global connectivity metrics (fragmentation).
  - Plots:
      * Rank-Size Distribution (Log-Log) - "The Zipf Plot"
      * Component Size Histogram (Log-Log)
"""

import argparse
import json
import collections
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# I/O Helpers (Consistent with your previous script)
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Objective 2: Weakly Connected Component Analysis of the PyPI graph."
    )
    default_graph_dir = Path("pypi_clean_graph")
    default_output_dir = Path("objective2_connectivity_outputs")

    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=default_graph_dir,
        help="Directory containing nodes.csv, edges.csv (default: pypi_clean_graph)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to write CSVs, JSONs, and figures (default: objective2_connectivity_outputs)",
    )
    return parser.parse_args()


def load_graph(graph_dir: Path) -> nx.DiGraph:
    """
    Load the directed dependency graph.
    """
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv not found at {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv not found at {edges_path}")

    # Load Nodes
    print(f"[INFO] Loading nodes from {nodes_path}...")
    df_nodes = pd.read_csv(nodes_path)
    if "package" not in df_nodes.columns:
        raise ValueError("nodes.csv must have a 'package' column.")

    # Load Edges
    print(f"[INFO] Loading edges from {edges_path}...")
    df_edges = pd.read_csv(edges_path)
    required_cols = {"source", "target"}
    if not required_cols.issubset(df_edges.columns):
        raise ValueError(f"edges.csv must have {required_cols} columns.")

    G = nx.DiGraph()
    # Add all nodes to track isolated ones
    packages = df_nodes["package"].astype(str).tolist()
    G.add_nodes_from(packages)

    # Add edges
    for row in df_edges.itertuples(index=False):
        src = str(row.source).strip()
        dst = str(row.target).strip()
        if src and dst:
            G.add_edge(src, dst)

    return G


# ---------------------------------------------------------------------------
# Analysis Class
# ---------------------------------------------------------------------------

class ConnectivityAnalysis:
    def __init__(self, G: nx.DiGraph):
        self.total_nodes = G.number_of_nodes()
        self.total_edges = G.number_of_edges()
        
        # Compute Weakly Connected Components (WCC)
        # sort by size (largest first)
        self.wccs = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
        self.sizes = [len(c) for c in self.wccs]
        
        # Giant Component (First one in sorted list)
        self.gcc_nodes = self.wccs[0] if self.wccs else set()
        self.gcc_size = len(self.gcc_nodes)
        
        # Metrics
        self.num_components = len(self.wccs)
        self.num_isolated = sum(1 for s in self.sizes if s == 1)
        
        # Calculate fragmentation stats
        self.percent_in_gcc = (self.gcc_size / self.total_nodes * 100) if self.total_nodes else 0
        self.percent_isolated = (self.num_isolated / self.total_nodes * 100) if self.total_nodes else 0


# ---------------------------------------------------------------------------
# Output Functions
# ---------------------------------------------------------------------------

def save_metrics_report(analysis: ConnectivityAnalysis, output_dir: Path):
    """Saves global connectivity statistics."""
    report = {
        "global_structure": {
            "total_nodes": analysis.total_nodes,
            "total_edges": analysis.total_edges,
            "total_connected_components": analysis.num_components
        },
        "giant_connected_component": {
            "size": analysis.gcc_size,
            "percentage_of_ecosystem": round(analysis.percent_in_gcc, 4),
            "status": "Dominant" if analysis.percent_in_gcc > 50 else "Fragmented"
        },
        "fragmentation": {
            "isolated_packages_count": analysis.num_isolated,
            "isolated_packages_percentage": round(analysis.percent_isolated, 4),
            "second_largest_component_size": analysis.sizes[1] if len(analysis.sizes) > 1 else 0
        }
    }
    
    out_path = output_dir / "connectivity_metrics.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[INFO] Saved metrics report to {out_path}")


def save_component_csv(analysis: ConnectivityAnalysis, output_dir: Path):
    """Saves a summary of the top 1000 components (or all if small)."""
    data = []
    for rank, comp in enumerate(analysis.wccs, 1):
        size = len(comp)
        # Store just size and rank for all, maybe first node as ID
        # To keep CSV small, we might not want to list EVERY node for the giant component
        example_node = next(iter(comp))
        data.append({
            "rank_id": rank,
            "size": size,
            "percentage": (size / analysis.total_nodes) * 100,
            "example_node": example_node
        })
    
    df = pd.DataFrame(data)
    out_path = output_dir / "connected_components_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved components CSV summary to {out_path}")


def plot_connectivity_charts(analysis: ConnectivityAnalysis, output_dir: Path):
    """
    Generates:
    1. Rank-Size Plot (Zipf): Shows the steep drop-off from the Giant Component.
    2. Component Size Distribution: Histogram of sizes.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')

    sizes = analysis.sizes
    ranks = range(1, len(sizes) + 1)

    # --- Plot 1: Rank-Size Distribution (Log-Log) ---
    # This is the "Zipf" plot requested for distribution analysis
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.loglog(ranks, sizes, marker='.', linestyle='none', color='#1f77b4', alpha=0.5, markersize=8)
    
    # Highlight the Giant Component
    ax1.scatter([1], [sizes[0]], color='red', s=100, label='Giant Connected Component (GCC)', zorder=5)
    
    ax1.set_xlabel("Component Rank (Log Scale)", fontsize=12)
    ax1.set_ylabel("Component Size (Nodes, Log Scale)", fontsize=12)
    ax1.set_title(f"Connectivity Rank-Size Distribution\n(GCC contains {analysis.percent_in_gcc:.1f}% of all packages)", fontsize=14)
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / "wcc_rank_size_distribution.png", dpi=300)
    plt.close(fig1)

    # --- Plot 2: Size Frequency Histogram (Log-Log) ---
    # How many components have size X?
    size_counts = collections.Counter(sizes)
    xs = sorted(size_counts.keys())
    ys = [size_counts[x] for x in xs]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.loglog(xs, ys, marker='o', linestyle='-', color='#d62728', alpha=0.7)
    
    ax2.set_xlabel("Component Size (Number of Packages)", fontsize=12)
    ax2.set_ylabel("Frequency (Number of Components)", fontsize=12)
    ax2.set_title("Distribution of Connected Component Sizes", fontsize=14)
    
    # Annotation for isolated nodes
    if 1 in size_counts:
        ax2.annotate(f'Isolated Nodes\n(Size=1, N={size_counts[1]})', 
                     xy=(1, size_counts[1]), 
                     xytext=(1.5, size_counts[1]/2),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(output_dir / "wcc_size_frequency.png", dpi=300)
    plt.close(fig2)
    
    print(f"[INFO] Generated plots in {output_dir}")


def print_console_summary(analysis: ConnectivityAnalysis):
    print("\n" + "="*60)
    print("WEAKLY CONNECTED COMPONENT (WCC) ANALYSIS")
    print("="*60)
    print(f"Total Packages (Nodes):     {analysis.total_nodes:,}")
    print(f"Total Components:           {analysis.num_components:,}")
    print("-" * 30)
    print(f"Giant Connected Component (GCC):")
    print(f"  - Size: {analysis.gcc_size:,} packages")
    print(f"  - Coverage: {analysis.percent_in_gcc:.2f}% of ecosystem")
    print("-" * 30)
    print(f"Fragmentation:")
    print(f"  - Isolated Packages (Size=1): {analysis.num_isolated:,} ({analysis.percent_isolated:.2f}%)")
    if len(analysis.sizes) > 1:
        print(f"  - 2nd Largest Component:      {analysis.sizes[1]:,}")
    print("="*60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    
    try:
        G = load_graph(args.graph_dir)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    print("[INFO] Computing Weakly Connected Components (WCC)...")
    analysis = ConnectivityAnalysis(G)
    
    print_console_summary(analysis)
    
    print(f"[INFO] Writing outputs to {args.output_dir}...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    save_metrics_report(analysis, args.output_dir)
    save_component_csv(analysis, args.output_dir)
    plot_connectivity_charts(analysis, args.output_dir)
    
    print("[SUCCESS] Analysis Complete.")

if __name__ == "__main__":
    main()