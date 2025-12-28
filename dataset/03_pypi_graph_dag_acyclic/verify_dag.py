import networkx as nx
import csv
import os
import sys

def verify_dag_integrity(directory):
    nodes_path = os.path.join(directory, "nodes.csv")
    edges_path = os.path.join(directory, "edges.csv")

    print(f"[INFO] Verifying DAG integrity for: {directory}")

    # 1. Load the Graph
    G = nx.DiGraph()
    
    print("[INFO] Loading nodes...")
    try:
        with open(nodes_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row:
                    G.add_node(row[0])
    except FileNotFoundError:
        print(f"[ERROR] Could not find {nodes_path}")
        return

    print("[INFO] Loading edges...")
    try:
        with open(edges_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 2:
                    G.add_edge(row[0], row[1])
    except FileNotFoundError:
        print(f"[ERROR] Could not find {edges_path}")
        return

    print(f"[INFO] Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    # 2. Check for Self Loops (Simplest Cycle)
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        print(f"[FAIL] Found {len(self_loops)} self-loops! This is NOT a DAG.")
        print(f"       Sample: {self_loops[:3]}")
        return False

    # 3. Check for Cycles (DAG Property)
    print("[INFO] Running topological sort to verify DAG property...")
    try:
        is_dag = nx.is_directed_acyclic_graph(G)
    except Exception as e:
        print(f"[ERROR] Algorithm failed: {e}")
        return False

    if is_dag:
        print("-" * 40)
        print("[SUCCESS] The graph is a strict Directed Acyclic Graph (DAG).")
        print("-" * 40)
        
        # Optional: Print longest path depth (expensive on huge graphs, simplified here)
        # We just confirm it's acyclic.
        return True
    else:
        print("-" * 40)
        print("[FAIL] The graph contains cycles.")
        print("-" * 40)
        try:
            cycle = nx.find_cycle(G, orientation="original")
            print(f"Sample cycle found: {cycle}")
        except:
            print("Could not extract sample cycle.")
        return False

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "pypi_dag"
    verify_dag_integrity(target_dir)
