import json
import csv
import os
import sys
import time
import networkx as nx

def convert_to_dag(input_dir, output_dir):
    """
    Converts a directed graph into a Directed Acyclic Graph (DAG) by identifying
    and removing nodes involved in cycles (Strongly Connected Components).
    
    Logic:
    1. Case A: SCC has > 1 node -> All nodes are cyclic.
    2. Case B: SCC has 1 node AND a self-loop -> Node is cyclic.
    """
    
    # 1. Setup paths
    input_nodes = os.path.join(input_dir, "nodes.csv")
    input_edges = os.path.join(input_dir, "edges.csv")
    input_adj = os.path.join(input_dir, "adjacency.jsonl")
    
    os.makedirs(output_dir, exist_ok=True)
    out_nodes = os.path.join(output_dir, "nodes.csv")
    out_edges = os.path.join(output_dir, "edges.csv")
    out_adj = os.path.join(output_dir, "adjacency.jsonl")
    out_meta = os.path.join(output_dir, "metadata.json")
    out_log = os.path.join(output_dir, "cyclic_analysis.json")

    print(f"[INFO] Loading graph from {input_dir}...")
    
    # 2. Load Graph into NetworkX
    # We need the full structure to calculate SCCs efficiently.
    G = nx.DiGraph()
    
    # Load Nodes (to ensure isolated nodes are tracked)
    try:
        with open(input_nodes, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if row:
                    G.add_node(row[0])
    except FileNotFoundError:
        print(f"[ERROR] Could not find {input_nodes}")
        sys.exit(1)

    # Load Edges
    try:
        with open(input_edges, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # Skip header
            for row in reader:
                if len(row) >= 2:
                    G.add_edge(row[0], row[1])
    except FileNotFoundError:
        print(f"[ERROR] Could not find {input_edges}")
        sys.exit(1)

    print(f"[INFO] Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    print("[INFO] Computing Strongly Connected Components (SCCs)...")

    # 3. Identify Cyclic Nodes (The Logic)
    sccs = list(nx.strongly_connected_components(G))
    
    case_a_nodes = set() # SCC size > 1
    case_b_nodes = set() # SCC size == 1 with self-loop
    
    # Structure for the detailed log
    log_data = {
        "summary": {},
        "cyclic_sccs_detected": [], # List of lists (for Case A)
        "self_loops_detected": []   # List of strings (for Case B)
    }

    for component in sccs:
        if len(component) > 1:
            # Case A: Complex Cycle
            comp_list = list(component)
            case_a_nodes.update(component)
            log_data["cyclic_sccs_detected"].append(comp_list)
        elif len(component) == 1:
            # Case B: Check for self-loop
            node = list(component)[0]
            if G.has_edge(node, node):
                case_b_nodes.add(node)
                log_data["self_loops_detected"].append(node)

    bad_nodes = case_a_nodes.union(case_b_nodes)
    valid_dag_nodes = set(G.nodes()) - bad_nodes
    
    print(f"[INFO] Cycle detection complete.")
    print(f"  - Case A (Complex Cycles): {len(case_a_nodes)} nodes")
    print(f"  - Case B (Self Loops):     {len(case_b_nodes)} nodes")
    print(f"  - Total Nodes to Remove:   {len(bad_nodes)}")
    print(f"  - Remaining DAG Nodes:     {len(valid_dag_nodes)}")

    # Update Log Summary
    log_data["summary"] = {
        "total_nodes_scanned": G.number_of_nodes(),
        "total_removed_nodes": len(bad_nodes),
        "count_case_a_nodes": len(case_a_nodes),
        "count_case_b_nodes": len(case_b_nodes),
        "count_cyclic_sccs": len(log_data["cyclic_sccs_detected"]),
        "count_self_loops": len(log_data["self_loops_detected"])
    }

    # 4. Stream and Filter Output
    
    # Write nodes.csv
    print("[INFO] Writing DAG nodes.csv...")
    with open(out_nodes, "w", encoding="utf-8") as f:
        f.write("package\n")
        for node in sorted(valid_dag_nodes):
            f.write(f"{node}\n")

    # Write edges.csv (Filter on fly)
    print("[INFO] Writing DAG edges.csv...")
    kept_edges_count = 0
    with open(input_edges, "r", encoding="utf-8") as fin, \
         open(out_edges, "w", encoding="utf-8") as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        header = next(reader, None)
        if header: writer.writerow(header)
        
        for row in reader:
            if len(row) < 2: continue
            src, dst = row[0], row[1]
            if src in valid_dag_nodes and dst in valid_dag_nodes:
                writer.writerow([src, dst])
                kept_edges_count += 1

    # Write adjacency.jsonl (Filter on fly)
    print("[INFO] Writing DAG adjacency.jsonl...")
    with open(input_adj, "r", encoding="utf-8") as fin, \
         open(out_adj, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip(): continue
            try:
                record = json.loads(line)
                pkg = record["package"]
                
                # Only keep record if the package itself is valid
                if pkg in valid_dag_nodes:
                    # Filter dependencies to only include valid nodes
                    new_deps = [d for d in record["dependencies"] if d in valid_dag_nodes]
                    record["dependencies"] = new_deps
                    fout.write(json.dumps(record) + "\n")
            except json.JSONDecodeError:
                continue

    # 5. Save Logs and Metadata
    print(f"[INFO] Saving analysis log to {out_log}...")
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "PyPI DAG (Ghosts Removed + Cycles Removed)",
        "source_dataset": input_dir,
        "metrics": {
            "final_node_count": len(valid_dag_nodes),
            "final_edge_count": kept_edges_count,
            "removed_cyclic_nodes": len(bad_nodes)
        }
    }
    
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 50)
    print(f"DONE. DAG dataset saved to: {output_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_dag.py <input_dir> <output_dir>")
        sys.exit(1)
        
    convert_to_dag(sys.argv[1], sys.argv[2])
