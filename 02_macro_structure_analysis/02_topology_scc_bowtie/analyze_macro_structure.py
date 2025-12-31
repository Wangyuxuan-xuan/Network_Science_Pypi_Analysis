#!/usr/bin/env python3
"""
Objective 2: Analyze Macro-Structure of the PyPI DS/ML dependency network.
Expanded to include detailed analysis of Circular Dependencies and Self-Loops.

This script:
  1) Loads the directed dependency graph.
  2) Computes Strongly Connected Components (SCCs).
  3) Performs Bow-Tie Analysis (Core, In, Out, Tendrils, Disconnected).
  4) Identifies and logs specific "Design Flaws":
     - Complex Cycles (SCC size > 1)
     - Self-Loops (SCC size == 1 with self-edge)
  5) Outputs:
     - CSV summaries of SCCs and Bow-tie membership.
     - JSON reports for programmatic processing of cycles.
     - Visualization plots:
         * SCC size distribution (log-log)
         * Bow-tie region sizes (bar chart)
         * Node Health Distribution (Log-scale Bar Chart)

Usage:
    python analyze_macro_structure.py --graph-dir pypi_clean_graph --output-dir objective2_outputs
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Any

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Objective 2: SCC / Bow-tie and Cycle Analysis of the PyPI DS/ML graph."
    )
    # Default paths set based on user prompt structure
    default_graph_dir = Path("pypi_clean_graph")
    default_output_dir = Path("objective2_outputs")

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
        help="Directory to write CSVs, JSONs, and figures (default: objective2_outputs)",
    )
    return parser.parse_args()


def load_graph(graph_dir: Path) -> nx.DiGraph:
    """
    Load the directed dependency graph from nodes.csv and edges.csv.
    """
    nodes_path = graph_dir / "nodes.csv"
    edges_path = graph_dir / "edges.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv not found at {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv not found at {edges_path}")

    # Load Nodes
    df_nodes = pd.read_csv(nodes_path)
    if "package" not in df_nodes.columns:
        raise ValueError(f"nodes.csv must have a 'package' column.")

    # Load Edges
    df_edges = pd.read_csv(edges_path)
    required_cols = {"source", "target"}
    if not required_cols.issubset(df_edges.columns):
        raise ValueError(f"edges.csv must have {required_cols} columns.")

    # Build directed graph
    G = nx.DiGraph()
    # Add all packages to ensure isolated nodes are tracked
    packages = df_nodes["package"].astype(str).tolist()
    G.add_nodes_from(packages)

    # Add edges
    for row in df_edges.itertuples(index=False):
        src = str(row.source).strip()
        dst = str(row.target).strip()
        if src and dst:
            G.add_edge(src, dst)

    return G


def load_metadata(graph_dir: Path) -> Dict:
    meta_path = graph_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Analysis Classes & Functions
# ---------------------------------------------------------------------------

class SCCAnalysis:
    def __init__(
        self,
        sccs: List[Set[str]],
        node_to_scc: Dict[str, int],
        scc_to_region: Dict[int, str],
        node_to_region: Dict[str, str],
        core_index: int,
        complex_cycles: List[Dict[str, Any]],
        self_loops: List[str],
        healthy_nodes_count: int
    ):
        self.sccs = sccs
        self.node_to_scc = node_to_scc
        self.scc_to_region = scc_to_region
        self.node_to_region = node_to_region
        self.core_index = core_index
        
        # New attributes for Cycle Analysis
        self.complex_cycles = complex_cycles  # List of dicts describing >1 node SCCs
        self.self_loops = self_loops          # List of node names
        self.healthy_nodes_count = healthy_nodes_count

    @property
    def num_scc(self) -> int:
        return len(self.sccs)

    @property
    def scc_sizes(self) -> List[int]:
        return [len(c) for c in self.sccs]

    @property
    def region_counts(self) -> Counter:
        return Counter(self.node_to_region.values())

    @property
    def scc_region_counts(self) -> Counter:
        return Counter(self.scc_to_region.values())


def analyze_structure(G: nx.DiGraph) -> SCCAnalysis:
    """
    Perform SCC detection, Bow-Tie Decomposition, and Cycle Analysis.
    """
    # 1. SCC Detection
    sccs = list(nx.strongly_connected_components(G))
    num_scc = len(sccs)

    # Map each node -> SCC index
    node_to_scc: Dict[str, int] = {}
    for idx, comp in enumerate(sccs):
        for node in comp:
            node_to_scc[node] = idx

    # 2. Cycle Detection (Case A & Case B)
    complex_cycles_info = []
    self_loop_nodes = []
    
    complex_nodes_set = set()
    self_loop_nodes_set = set()

    for idx, comp in enumerate(sccs):
        size = len(comp)
        if size > 1:
            # Case A: Complex Cycle
            comp_nodes = list(comp)
            complex_nodes_set.update(comp)
            complex_cycles_info.append({
                "scc_id": idx,
                "node_count": size,
                "nodes": comp_nodes
            })
        elif size == 1:
            # Case B: Check for self-loop
            node = next(iter(comp))
            if G.has_edge(node, node):
                self_loop_nodes.append(node)
                self_loop_nodes_set.add(node)
    
    total_bad_nodes = len(complex_nodes_set) + len(self_loop_nodes_set)
    total_nodes = G.number_of_nodes()
    healthy_count = total_nodes - total_bad_nodes

    # 3. Bow-Tie Decomposition (on Condensation Graph)
    H = nx.DiGraph()
    H.add_nodes_from(range(num_scc))

    for u, v in G.edges():
        cu = node_to_scc[u]
        cv = node_to_scc[v]
        if cu != cv:
            H.add_edge(cu, cv)

    # Identify CORE (Largest SCC)
    if sccs:
        core_index = max(range(num_scc), key=lambda i: len(sccs[i]))
    else:
        core_index = -1 # Empty graph handling

    # Ancestors/descendants on condensation DAG
    if core_index != -1:
        ancestors_core = nx.ancestors(H, core_index)
        descendants_core = nx.descendants(H, core_index)
        core_set = {core_index}
    else:
        ancestors_core = set()
        descendants_core = set()
        core_set = set()

    in_scc = ancestors_core
    out_scc = descendants_core
    base = core_set | in_scc | out_scc

    # Tendrils
    tendrils_from_in: Set[int] = set()
    for i in in_scc:
        tendrils_from_in |= nx.descendants(H, i)

    tendrils_to_out: Set[int] = set()
    for j in out_scc:
        tendrils_to_out |= nx.ancestors(H, j)

    tendrils = (tendrils_from_in | tendrils_to_out) - base

    # Disconnected
    all_scc_indices = set(H.nodes())
    disconnected = all_scc_indices - base - tendrils

    # Mappings
    scc_to_region: Dict[int, str] = {}
    for i in all_scc_indices:
        if i == core_index:
            scc_to_region[i] = "CORE"
        elif i in in_scc:
            scc_to_region[i] = "IN"
        elif i in out_scc:
            scc_to_region[i] = "OUT"
        elif i in tendrils:
            scc_to_region[i] = "TENDRILS"
        elif i in disconnected:
            scc_to_region[i] = "DISCONNECTED"
        else:
            scc_to_region[i] = "UNKNOWN"

    node_to_region: Dict[str, str] = {
        node: scc_to_region[node_to_scc[node]] for node in G.nodes()
    }

    return SCCAnalysis(
        sccs=sccs,
        node_to_scc=node_to_scc,
        scc_to_region=scc_to_region,
        node_to_region=node_to_region,
        core_index=core_index,
        complex_cycles=complex_cycles_info,
        self_loops=self_loop_nodes,
        healthy_nodes_count=healthy_count
    )


# ---------------------------------------------------------------------------
# Output Functions (JSON, CSV, Plots)
# ---------------------------------------------------------------------------

def save_json_reports(analysis: SCCAnalysis, output_dir: Path) -> None:
    """
    Saves the two specific JSON files requested for programmatic processing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Circular Dependencies (Case A)
    # Aggregating total nodes involved
    total_complex_nodes = sum(item["node_count"] for item in analysis.complex_cycles)
    
    circular_data = {
        "summary": {
            "total_non_trivial_sccs": len(analysis.complex_cycles),
            "total_nodes_involved": total_complex_nodes
        },
        "sccs": analysis.complex_cycles
    }
    
    path_circular = output_dir / "circular_dependencies.json"
    with open(path_circular, "w", encoding="utf-8") as f:
        json.dump(circular_data, f, indent=2)
    print(f"[INFO] Saved Circular Dependencies report to: {path_circular}")

    # 2. Self Loops (Case B)
    self_loop_data = {
        "summary": {
            "total_self_loop_nodes": len(analysis.self_loops)
        },
        "nodes": analysis.self_loops
    }
    
    path_selfloops = output_dir / "self_loops.json"
    with open(path_selfloops, "w", encoding="utf-8") as f:
        json.dump(self_loop_data, f, indent=2)
    print(f"[INFO] Saved Self-Loop report to: {path_selfloops}")


def save_csv_summaries(G: nx.DiGraph, analysis: SCCAnalysis, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # SCC Summary
    scc_rows = []
    for scc_id, comp in enumerate(analysis.sccs):
        region = analysis.scc_to_region.get(scc_id, "UNKNOWN")
        scc_rows.append({
            "scc_id": scc_id,
            "size": len(comp),
            "region": region,
            "is_core": int(scc_id == analysis.core_index),
        })
    df_scc = pd.DataFrame(scc_rows).sort_values("size", ascending=False)
    df_scc.to_csv(output_dir / "scc_summary.csv", index=False)

    # Node Membership
    node_rows = []
    for node in G.nodes():
        node_rows.append({
            "package": node,
            "region": analysis.node_to_region[node],
            "scc_id": analysis.node_to_scc[node],
        })
    df_nodes = pd.DataFrame(node_rows).sort_values("package")
    df_nodes.to_csv(output_dir / "node_bowtie_membership.csv", index=False)
    print(f"[INFO] Saved CSV summaries to {output_dir}")


def plot_visualizations(analysis: SCCAnalysis, output_dir: Path) -> None:
    """
    Generates:
    1. SCC size distribution (Log-Log)
    2. Bow-Tie region sizes (Bar)
    3. Node Health Distribution (Log-Scale Bar Chart)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use('ggplot')  # Use a nice style if available

    # --- Plot 1: SCC Size Distribution (Log-Log) ---
    sizes = sorted(analysis.scc_sizes, reverse=True)
    ranks = range(1, len(sizes) + 1)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.loglog(ranks, sizes, marker="o", linestyle="none", alpha=0.6, markersize=5, color="#1f77b4")
    ax1.set_xlabel("SCC Rank (Log Scale)", fontsize=12)
    ax1.set_ylabel("SCC Size (Number of Packages, Log Scale)", fontsize=12)
    ax1.set_title("Distribution of Strongly Connected Component Sizes", fontsize=14)
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    fig1.tight_layout()
    fig1.savefig(output_dir / "scc_size_distribution.png", dpi=300)
    plt.close(fig1)

    # --- Plot 2: Bow-Tie Region Sizes ---
    region_order = ["CORE", "IN", "OUT", "TENDRILS", "DISCONNECTED"]
    counts = analysis.region_counts
    values = [counts.get(r, 0) for r in region_order]
    
    # Colors for bow-tie
    colors_bt = ['#d62728', '#ff7f0e', '#2ca02c', '#9467bd', '#7f7f7f']

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    bars = ax2.bar(region_order, values, color=colors_bt, alpha=0.8)
    ax2.set_ylabel("Number of Packages", fontsize=12)
    ax2.set_title("Bow-Tie Structure of the PyPI Dependency Network", fontsize=14)
    
    # Add counts on top of bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig2.tight_layout()
    fig2.savefig(output_dir / "bowtie_region_sizes.png", dpi=300)
    plt.close(fig2)

    # --- Plot 3: Node Health / Design Flaw Distribution (Log-Scale Bar Chart) ---
    # REPLACED PIE CHART WITH LOG-SCALE BAR CHART TO HANDLE 99.8% IMBALANCE
    
    # Metrics
    count_complex = sum(item["node_count"] for item in analysis.complex_cycles)
    count_selfloops = len(analysis.self_loops)
    count_healthy = analysis.healthy_nodes_count
    
    # Setup data
    categories = ['Healthy Nodes', 'Complex Cycles\n(SCC size > 1)', 'Self-Loops\n(SCC=1 + Loop)']
    values_health = [count_healthy, count_complex, count_selfloops]
    colors_health = ['#2ca02c', '#d62728', '#ff7f0e'] # Green (Good), Red (Complex), Orange (Simple)
    
    total = sum(values_health)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Draw bars
    bars = ax3.bar(categories, values_health, color=colors_health, alpha=0.8)
    
    # Set Y-axis to Log Scale to make small bars visible
    ax3.set_yscale('log')
    ax3.set_ylabel("Number of Nodes (Log Scale)", fontsize=12)
    ax3.set_title("Distribution of Structural Design Flaws", fontsize=14, fontweight='bold')
    
    # Add detailed annotations on top of bars
    for bar, val in zip(bars, values_health):
        height = bar.get_height()
        if height > 0:
            percentage = (val / total) * 100
            # Place label slightly above bar
            ax3.text(
                bar.get_x() + bar.get_width()/2., 
                height * 1.1, # Shift up slightly (multiplicative for log scale)
                f"N={val}\n({percentage:.2f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
            )
        else:
            # Handle 0 case
            ax3.text(
                bar.get_x() + bar.get_width()/2., 
                1, # floor for log scale
                "N=0",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='black'
            )

    # Add a note about the scale
    plt.figtext(0.99, 0.01, "* Y-axis is Logarithmic", ha="right", fontsize=8, style='italic')

    fig3.tight_layout()
    fig3.savefig(output_dir / "node_health_distribution.png", dpi=300)
    plt.close(fig3)
    
    print(f"[INFO] Generated 3 plots in {output_dir}")


def print_console_summary(analysis: SCCAnalysis, metadata: Dict) -> None:
    print("\n" + "=" * 72)
    print("MACRO-STRUCTURE & CYCLE ANALYSIS RESULTS")
    print("=" * 72)
    
    # Metadata info
    if metadata:
        print(f"Dataset Timestamp: {metadata.get('timestamp_utc', 'N/A')}")
        print(f"Seed Packages: {len(metadata.get('seed_packages', []))}")
        print("-" * 30)

    # Bow Tie
    print(f"Largest SCC (CORE) Size: {len(analysis.sccs[analysis.core_index])} packages")
    print("\nBow-Tie Regions:")
    total_nodes = sum(analysis.region_counts.values())
    for region in ["CORE", "IN", "OUT", "TENDRILS", "DISCONNECTED"]:
        c = analysis.region_counts.get(region, 0)
        pct = (c / total_nodes * 100) if total_nodes else 0
        print(f"  {region:<15}: {c:>6} ({pct:>5.2f}%)")

    # Cycle Analysis
    print("-" * 30)
    print("Structural Flaws Detected:")
    
    total_complex_nodes = sum(c["node_count"] for c in analysis.complex_cycles)
    print(f"  1. Complex Circular Dependencies (SCC size > 1):")
    print(f"     - Number of Groups (SCCs): {len(analysis.complex_cycles)}")
    print(f"     - Total Packages Involved: {total_complex_nodes}")
    
    print(f"  2. Self-Loops (Trivial SCC with self-edge):")
    print(f"     - Total Packages Involved: {len(analysis.self_loops)}")
    
    print(f"  3. Healthy Packages (Valid DAG nodes):")
    print(f"     - Total: {analysis.healthy_nodes_count}")
    
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    
    print(f"[INFO] Loading graph from: {args.graph_dir}")
    try:
        G = load_graph(args.graph_dir)
        metadata = load_metadata(args.graph_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load graph: {e}")
        return

    print("[INFO] Computing SCCs and analyzing structure...")
    analysis = analyze_structure(G)

    print_console_summary(analysis, metadata)

    print("[INFO] Saving Results...")
    save_csv_summaries(G, analysis, args.output_dir)
    save_json_reports(analysis, args.output_dir)
    
    print("[INFO] Generating Visualizations...")
    plot_visualizations(analysis, args.output_dir)
    
    print(f"[SUCCESS] All operations complete. Outputs in: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
