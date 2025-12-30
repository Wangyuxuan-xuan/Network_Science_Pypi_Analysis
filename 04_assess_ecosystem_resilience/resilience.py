"""
Objective 4: Assess Ecosystem Resilience
PyPI Dependency Network - Network Science Project

This script performs targeted node removal experiments to assess the resilience
of the PyPI dependency network. It removes top-k nodes identified by centrality
analysis and measures the impact on network integrity.

EXTENDED WITH NULL MODEL COMPARISON (Objective 5 Baseline):
- Runs the same removal experiments on randomized graphs (depth-preserving DAG randomization)
- Compares original network resilience against null models to validate findings
- Randomized graphs preserve: degree sequence, DAG property, topological order
- Same methodology: PageRank centrality, cascade failures, LWCC metrics
- Generates comparison plot: original network vs averaged randomized networks
"""

import json
import time
from datetime import datetime

from config import CONFIG, METADATA_FILE
from data_loader import load_graph, load_centrality_results
from experiments import perform_removal_experiments, run_null_model_comparison
from results_handler import analyze_results, save_results
from visualization import create_visualizations


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("OBJECTIVE 4: ASSESS ECOSYSTEM RESILIENCE")
    print("PyPI Dependency Network - Network Science Project")
    print("=" * 80)
    print(f"\nExecution started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    start_time = time.time()
    
    # Phase 1: Load graph
    G = load_graph()
    original_num_nodes = G.number_of_nodes()
    
    # Phase 2: Load centrality dataset
    centrality_df = load_centrality_results()
    
    # Phase 3: Perform removal experiments on original graph
    results_df, baseline_metrics = perform_removal_experiments(G, centrality_df)
    
    # Phase 3B: Null Model Comparison (Objective 5 baseline)
    # Run the same experiments on randomized graphs and compute averaged dataset
    null_model_df = run_null_model_comparison(G, original_num_nodes, centrality_df)
    
    # Phase 4: Analyze dataset
    analyze_results(results_df, baseline_metrics, centrality_df)
    
    # Phase 5: Create visualizations (including null model comparison if available)
    create_visualizations(results_df, baseline_metrics, null_model_df=null_model_df)
    
    # Phase 6: Save dataset
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    save_results(results_df, baseline_metrics, centrality_df, metadata, null_model_df=null_model_df)
    
    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("RESILIENCE ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nTotal execution time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"✓ Performed {len(results_df)} removal experiments on original network")
    if null_model_df is not None:
        print(f"✓ Completed null model comparison using {CONFIG['num_random_graphs']} randomized graphs")
    print(f"✓ Results saved to: results/")
    print(f"\nKey Finding: Network resilience assessed through targeted node removal with CASCADE FAILURES")
    print(f"  - Baseline LWCC: {baseline_metrics['lwcc_size']:,} nodes")
    print(f"  - Tested removal of top-k nodes for k = {CONFIG['removal_k_values']}")
    print(f"  - Cascade failure model: All dependent packages (direct/indirect) also fail")
    print(f"  - This simulates real-world scenarios where removing critical packages causes chain reactions")
    if null_model_df is not None:
        print(f"\nNull Model Comparison (Objective 5):")
        print(f"  - Compared original network resilience against {CONFIG['num_random_graphs']} randomized null models")
        print(f"  - Randomized graphs preserve: degree sequence, DAG property, topological order")
        print(f"  - Comparison plot saved: resilience_null_model_comparison.png")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
