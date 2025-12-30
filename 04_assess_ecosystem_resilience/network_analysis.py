"""
Network analysis module for resilience analysis.
Handles network metrics computation and cascade failure analysis.
"""

import networkx as nx
import numpy as np

from config import CONFIG


def find_cascade_failures(G, removed_nodes):
    """
    Find all nodes that will fail due to cascade effect.
    
    In a dependency graph where edge (A, B) means A depends on B:
    - If B is removed, all packages that depend on B (directly or indirectly) will fail.
    - This is found by finding all nodes reachable from removed nodes in the reverse graph.
    
    Args:
        G: Directed graph where edge (A, B) means A depends on B
        removed_nodes: List of initially removed nodes
        
    Returns:
        set: All nodes that fail (including cascade failures)
    """
    if not removed_nodes:
        return set()
    
    # Build reverse graph: edge (A, B) in G means A depends on B
    # In reverse graph G_reverse, edge (B, A) means B is depended on by A
    # So from a removed node B in G_reverse, we can reach all packages that depend on B
    G_reverse = G.reverse()
    
    # Find all nodes that are reachable from any removed node in the reverse graph
    # These are all packages that depend (directly or indirectly) on the removed packages
    failed_nodes = set(removed_nodes)
    
    for removed_node in removed_nodes:
        if removed_node in G_reverse:
            # Find all nodes reachable from this removed node in reverse graph
            # These are all packages that depend on this removed package (directly or indirectly)
            try:
                # Use descendants to find all nodes reachable from removed_node
                # In the reverse graph, this gives us all packages that depend on removed_node
                reachable = nx.descendants(G_reverse, removed_node)
                failed_nodes.update(reachable)
            except Exception as e:
                # If node doesn't exist or other error, skip
                # This can happen if the node is isolated or doesn't exist in the graph
                pass
    
    return failed_nodes


def compute_network_metrics(G, removed_nodes=None, enable_cascade=True):
    """
    Compute network integrity metrics after node removal with cascade failure simulation.
    
    Args:
        G: Original graph
        removed_nodes: Initially removed nodes (top-k by centrality)
        enable_cascade: If True, simulate cascade failures (all dependent packages also fail)
    
    Returns:
        dict: Network metrics including LWCC size, component count, etc.
    """
    if removed_nodes is None:
        removed_nodes = []
    
    # Find all nodes that fail (including cascade failures)
    if enable_cascade and CONFIG['enable_cascade_failures']:
        all_failed_nodes = find_cascade_failures(G, removed_nodes)
    else:
        all_failed_nodes = set(removed_nodes)
    
    # Create a copy and remove all failed nodes
    G_copy = G.copy()
    G_copy.remove_nodes_from(all_failed_nodes)
    
    # Compute weakly connected components
    weak_components = list(nx.weakly_connected_components(G_copy))
    weak_component_sizes = [len(comp) for comp in weak_components]
    
    if len(weak_component_sizes) == 0:
        lwcc_size = 0
        lwcc_fraction = 0.0
    else:
        lwcc_size = max(weak_component_sizes)
        lwcc_fraction = lwcc_size / G_copy.number_of_nodes() if G_copy.number_of_nodes() > 0 else 0.0
    
    # Compute strongly connected components
    strong_components = list(nx.strongly_connected_components(G_copy))
    strong_component_sizes = [len(comp) for comp in strong_components]
    
    if len(strong_component_sizes) == 0:
        lscc_size = 0
    else:
        lscc_size = max(strong_component_sizes)
    
    metrics = {
        'nodes_remaining': G_copy.number_of_nodes(),
        'edges_remaining': G_copy.number_of_edges(),
        'nodes_removed': len(removed_nodes),  # Initial removal count
        'nodes_failed_total': len(all_failed_nodes),  # Total failed (including cascade)
        'cascade_failures': len(all_failed_nodes) - len(removed_nodes),  # Additional cascade failures
        'weak_component_count': len(weak_components),
        'lwcc_size': lwcc_size,
        'lwcc_fraction': lwcc_fraction,
        'strong_component_count': len(strong_components),
        'lscc_size': lscc_size,
        'density': nx.density(G_copy) if G_copy.number_of_nodes() > 0 else 0.0,
    }
    
    # Compute additional metrics if graph is not too fragmented
    if lwcc_size > 100:  # Only compute for reasonably sized components
        try:
            lwcc_subgraph = G_copy.subgraph(max(weak_components, key=len))
            
            # Average shortest path length (only for connected nodes)
            if lwcc_subgraph.number_of_nodes() > 1:
                try:
                    # For large graphs, sample nodes for path length
                    if lwcc_subgraph.number_of_nodes() > 1000:
                        sample_nodes = np.random.choice(
                            list(lwcc_subgraph.nodes()), 
                            size=min(100, lwcc_subgraph.number_of_nodes()),
                            replace=False
                        )
                        path_lengths = []
                        for u in sample_nodes:
                            paths = nx.single_source_shortest_path_length(lwcc_subgraph, u, cutoff=10)
                            path_lengths.extend([d for d in paths.values() if d > 0])
                        avg_path_length = np.mean(path_lengths) if path_lengths else np.nan
                    else:
                        avg_path_length = nx.average_shortest_path_length(lwcc_subgraph)
                except:
                    avg_path_length = np.nan
            else:
                avg_path_length = np.nan
            
            metrics['avg_path_length'] = avg_path_length
        except:
            metrics['avg_path_length'] = np.nan
    else:
        metrics['avg_path_length'] = np.nan
    
    return metrics

