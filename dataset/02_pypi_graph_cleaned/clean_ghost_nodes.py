import json
import csv
import os
import sys
import time

def clean_graph(input_dir, output_dir):
    """
    Cleans the PyPI dependency graph by removing 'Ghost Nodes'.
    
    Ghost Nodes are defined as packages that appear as dependencies (targets)
    but do not exist as primary entries (sources) in the adjacency list.
    """
    
    # 1. Setup paths
    input_adj = os.path.join(input_dir, "adjacency.jsonl")
    input_edges = os.path.join(input_dir, "edges.csv")
    
    os.makedirs(output_dir, exist_ok=True)
    out_nodes = os.path.join(output_dir, "nodes.csv")
    out_edges = os.path.join(output_dir, "edges.csv")
    out_adj = os.path.join(output_dir, "adjacency.jsonl")
    out_meta = os.path.join(output_dir, "metadata.json")
    out_log = os.path.join(output_dir, "removed_ghosts.txt")

    print(f"[INFO] Reading source of truth from {input_adj}...")

    # 2. Build the "Real List" (Valid Nodes)
    # These are packages that strictly exist in the crawl data as keys.
    valid_nodes = set()
    try:
        with open(input_adj, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                valid_nodes.add(record["package"])
    except FileNotFoundError:
        print(f"[ERROR] Could not find {input_adj}")
        sys.exit(1)
        
    print(f"[INFO] Found {len(valid_nodes):,} valid 'Real' packages.")

    # 3. Filter Adjacency & Identify Ghosts
    print("[INFO] Cleaning adjacency list and identifying ghosts...")
    ghosts_found = set()
    edge_count = 0
    
    with open(input_adj, "r", encoding="utf-8") as fin, \
         open(out_adj, "w", encoding="utf-8") as fout:
        
        for line in fin:
            if not line.strip(): continue
            record = json.loads(line)
            pkg = record["package"]
            
            # Filter dependencies
            clean_deps = []
            for dep in record["dependencies"]:
                if dep in valid_nodes:
                    clean_deps.append(dep)
                    edge_count += 1 # Count valid edges for metadata
                else:
                    ghosts_found.add(dep)
            
            # Write cleaned record
            record["dependencies"] = clean_deps
            fout.write(json.dumps(record) + "\n")

    # 4. Write cleaned edges.csv
    # We regenerate this from the cleaned adjacency logic or filter the old file.
    # Filtering the old file is safer to ensure strict consistency with the input format.
    print("[INFO] Cleaning edges.csv...")
    dropped_edges_count = 0
    
    with open(input_edges, "r", encoding="utf-8") as fin, \
         open(out_edges, "w", encoding="utf-8") as fout:
        
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        
        # Header
        header = next(reader, None)
        if header:
            writer.writerow(header)
            
        for row in reader:
            if len(row) < 2: continue
            src, dst = row[0], row[1]
            
            # Logic: Keep edge only if BOTH source and target are valid.
            # (Source should always be valid if it came from adjacency keys, but we check target primarily)
            if src in valid_nodes and dst in valid_nodes:
                writer.writerow([src, dst])
            else:
                dropped_edges_count += 1

    # 5. Write cleaned nodes.csv
    print("[INFO] Writing clean nodes.csv...")
    with open(out_nodes, "w", encoding="utf-8") as f:
        f.write("package\n")
        for node in sorted(valid_nodes):
            f.write(f"{node}\n")

    # 6. Write Ghost Log
    print(f"[INFO] Writing log of {len(ghosts_found):,} unique ghost packages...")
    with open(out_log, "w", encoding="utf-8") as f:
        f.write(f"Total Unique Ghosts: {len(ghosts_found)}\n")
        f.write("--------------------------------------------------\n")
        for g in sorted(ghosts_found):
            f.write(f"{g}\n")

    # 7. Write Metadata
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "description": "Cleaned PyPI Graph (Ghosts Removed)",
        "num_projects": len(valid_nodes),
        "num_edges": edge_count,
        "num_ghosts_removed": len(ghosts_found),
        "source_dataset": input_dir
    }
    
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("-" * 50)
    print(f"DONE. Clean dataset saved to: {output_dir}")
    print(f"  - Valid Nodes:   {len(valid_nodes):,}")
    print(f"  - Valid Edges:   {edge_count:,}")
    print(f"  - Ghosts Removed: {len(ghosts_found):,}")
    print(f"  - Dropped Edges:  {dropped_edges_count:,}")
    print(f"  - Ghost Log:      {out_log}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_ghost_nodes.py <input_dir> <output_dir>")
        sys.exit(1)
        
    clean_graph(sys.argv[1], sys.argv[2])
