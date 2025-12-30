"""
Visualization module for resilience analysis.
Handles creation of plots and charts.
"""

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def create_visualizations(results_df, baseline_metrics, null_model_df=None):
    """
    Create visualization plots with optional null model comparison.
    
    Args:
        results_df: Results from original graph removal experiments
        baseline_metrics: Baseline metrics from original graph
        null_model_df: Optional DataFrame with averaged dataset from randomized graphs
    """
    print("\n" + "=" * 80)
    print("PHASE 5: CREATING VISUALIZATIONS")
    print("=" * 80)
    
    print("\nGenerating resilience plots...")
    
    try:
        # Extract data from results_df
        k_values = results_df['k'].values
        lwcc_retention = results_df['lwcc_size_retention'].values
        
        # Filter out any negative k values (safety check)
        valid_mask = k_values >= 0
        k_values = k_values[valid_mask]
        lwcc_retention = lwcc_retention[valid_mask]
        
        # Add k=0 baseline point if not present (retention = 1.0)
        if len(k_values) == 0 or k_values[0] != 0:
            k_values = np.concatenate([[0], k_values])
            lwcc_retention = np.concatenate([[1.0], lwcc_retention])
        
        # Fix symlog connection at k=0 (use small positive value)
        k_plot = k_values.copy()
        k_plot[0] = 1e-6
        
        # Ensure all k_plot values are non-negative (safety check)
        k_plot = np.maximum(k_plot, 1e-6)
        
        # Create first plot: Original Network only
        print("  - Creating Original Network plot...")
        create_original_network_plot(k_plot, lwcc_retention, k_values)
        
        # Create second plot: Original Network vs Null Model (if available)
        if null_model_df is not None and len(null_model_df) > 0:
            print("  - Creating Original Network vs Null Model comparison plot...")
            null_k_values = null_model_df['k'].values
            null_retention = null_model_df['lwcc_size_retention'].values
            
            # Filter out any negative k values (safety check)
            valid_mask = null_k_values >= 0
            null_k_values = null_k_values[valid_mask]
            null_retention = null_retention[valid_mask]
            
            # Add k=0 baseline point if not present (retention = 1.0)
            if len(null_k_values) == 0 or null_k_values[0] != 0:
                null_k_values = np.concatenate([[0], null_k_values])
                null_retention = np.concatenate([[1.0], null_retention])
            
            # Fix symlog for null model k values (use small positive value)
            null_k_plot = null_k_values.copy()
            null_k_plot[0] = 1e-6
            
            # Ensure all null_k_plot values are non-negative (safety check)
            null_k_plot = np.maximum(null_k_plot, 1e-6)
            
            create_comparison_plot(k_plot, lwcc_retention, k_values, 
                                 null_k_plot, null_retention, null_k_values)
        
    except Exception as e:
        print(f"\n⚠ Could not create visualizations: {e}")
        import traceback
        traceback.print_exc()


def create_original_network_plot(k_plot, lwcc_retention, k_values):
    """
    Create plot showing only Original Network resilience curve.
    
    Args:
        k_plot: k values for plotting (with k=0 fixed to 1e-6 for symlog)
        lwcc_retention: LWCC retention values
        k_values: Original k values (for annotations)
    """
    plt.figure(figsize=(9, 6))
    
    # Plot Original Network
    plt.plot(
        k_plot,
        lwcc_retention,
        marker="o",
        linewidth=2.5,
        label="Targeted Removal"
    )
    
    # Thresholds
    plt.axhline(0.5, linestyle="--", linewidth=1, color='gray')
    plt.axhline(0.1, linestyle="--", linewidth=1, color='gray')
    
    # Find key points for annotations
    # Find k where retention drops below 50%
    below_50 = np.where(lwcc_retention < 0.5)[0]
    if len(below_50) > 0:
        k_50_idx = below_50[0]
        k_50 = k_values[k_50_idx]
        retention_50 = lwcc_retention[k_50_idx]
        
        plt.annotate(
            f"50% collapse\n(k = {k_50})",
            xy=(k_plot[k_50_idx], retention_50),
            xytext=(k_plot[k_50_idx] * 2, retention_50 + 0.15),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )
    
    # Find k where retention drops below 10%
    below_10 = np.where(lwcc_retention < 0.1)[0]
    if len(below_10) > 0:
        k_10_idx = below_10[0]
        k_10 = k_values[k_10_idx]
        retention_10 = lwcc_retention[k_10_idx]
        
        plt.annotate(
            f"Severe fragmentation\n(<10%, k = {k_10})",
            xy=(k_plot[k_10_idx], retention_10),
            xytext=(k_plot[k_10_idx] * 2, retention_10 + 0.15),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )
    
    # Axes (关键：分段刻度)
    plt.xscale("symlog", linthresh=10, base=10)
    # Ensure x-axis starts at 0 (no negative values)
    plt.xlim(left=0, right=None)
    
    # Set ticks - use standard values that cover the range
    max_k = int(k_values.max())
    # Standard tick values
    standard_ticks = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    # Filter to only include ticks up to max_k
    ticks = [t for t in standard_ticks if t <= max_k]
    # Add max_k if not already in ticks
    if max_k not in ticks:
        ticks.append(max_k)
    plt.xticks(ticks, ticks)
    
    plt.ylim(0, 1.05)
    
    plt.xlabel("Number of removed packages (k)", fontsize=12)
    plt.ylabel("LWCC Retention (fraction of original network)", fontsize=12)
    
    plt.title(
        "Ecosystem Fragility under Targeted Package Removal",
        fontsize=14
    )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = PLOTS_DIR / "resilience_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved Original Network plot to {plot_file}")
    plt.close()


def create_comparison_plot(k_plot, lwcc_retention, k_values, 
                          null_k_plot, null_retention, null_k_values):
    """
    Create comparison plot showing Original Network vs Null Model.
    
    Args:
        k_plot: k values for original network (with k=0 fixed to 1e-6 for symlog)
        lwcc_retention: LWCC retention values for original network
        k_values: Original k values (for annotations)
        null_k_plot: k values for null model (with k=0 fixed to 1e-6 for symlog)
        null_retention: LWCC retention values for null model
        null_k_values: Original null model k values
    """
    plt.figure(figsize=(9, 6))
    
    # Plot Original Network
    plt.plot(
        k_plot,
        lwcc_retention,
        marker="o",
        linewidth=2.5,
        label="Targeted Removal (Original PyPI)"
    )
    
    # Plot Null Model
    plt.plot(
        null_k_plot,
        null_retention,
        marker="s",
        linestyle="--",
        linewidth=2,
        label="Targeted Removal (Randomized Null Model)"
    )
    
    # Thresholds
    plt.axhline(0.5, linestyle="--", linewidth=1, color='gray')
    plt.axhline(0.1, linestyle="--", linewidth=1, color='gray')
    
    # Find key points for annotations (using original network)
    # Find k where retention drops below 50%
    below_50 = np.where(lwcc_retention < 0.5)[0]
    if len(below_50) > 0:
        k_50_idx = below_50[0]
        k_50 = k_values[k_50_idx]
        retention_50 = lwcc_retention[k_50_idx]
        
        plt.annotate(
            f"50% collapse\n(k = {k_50})",
            xy=(k_plot[k_50_idx], retention_50),
            xytext=(k_plot[k_50_idx] * 2, retention_50 + 0.15),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )
    
    # Find k where retention drops below 10%
    below_10 = np.where(lwcc_retention < 0.1)[0]
    if len(below_10) > 0:
        k_10_idx = below_10[0]
        k_10 = k_values[k_10_idx]
        retention_10 = lwcc_retention[k_10_idx]
        
        plt.annotate(
            f"Severe fragmentation\n(<10%, k = {k_10})",
            xy=(k_plot[k_10_idx], retention_10),
            xytext=(k_plot[k_10_idx] * 2, retention_10 + 0.15),
            arrowprops=dict(arrowstyle="->"),
            fontsize=11
        )
    
    # Axes (关键：分段刻度)
    plt.xscale("symlog", linthresh=10, base=10)
    # Ensure x-axis starts at 0 (no negative values)
    plt.xlim(left=0, right=None)
    
    # Set ticks - use standard values that cover the range
    max_k = int(max(k_values.max(), null_k_values.max()))
    # Standard tick values
    standard_ticks = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    # Filter to only include ticks up to max_k
    ticks = [t for t in standard_ticks if t <= max_k]
    # Add max_k if not already in ticks
    if max_k not in ticks:
        ticks.append(max_k)
    plt.xticks(ticks, ticks)
    
    plt.ylim(0, 1.05)
    
    plt.xlabel("Number of removed packages (k)", fontsize=12)
    plt.ylabel("LWCC Retention (fraction of original network)", fontsize=12)
    
    plt.title(
        "Ecosystem Fragility under Targeted Package Removal\nOriginal Network vs Randomized Null Model",
        fontsize=14
    )
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = PLOTS_DIR / "resilience_null_model_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved comparison plot to {plot_file}")
    plt.close()
