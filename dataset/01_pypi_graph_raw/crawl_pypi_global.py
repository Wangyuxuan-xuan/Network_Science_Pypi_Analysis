#!/usr/bin/env python3
"""
Global PyPI Dependency Crawler (JSON API + Simple Index)

This script builds a *global* directed dependency graph for all projects on PyPI.

Graph definition:
    - Nodes: Python packages (projects on PyPI).
    - Directed edge (A, B): project A lists project B as a dependency,
      according to the 'requires_dist' field in the PyPI JSON API for A.

Abstraction (same as your original DS/ML script):
    1. Ignore optional dependencies (those whose marker uses `extra == ...`).
    2. Strip all version specifiers and extras; keep only the core package name.
       e.g., "numpy>=1.23" -> "numpy".

Data source:
    - Project list from the PyPI Simple Index: https://pypi.org/simple/
    - Metadata from PyPI JSON API: https://pypi.org/pypi/<project>/json

Outputs (same format as original script):
    - nodes.csv:       all unique package names (one per line, header 'package')
    - edges.csv:       directed edges (CSV with header 'source,target')
    - adjacency.jsonl: one JSON record per package:
                       {"package": ..., "dependencies": [...]}
    - metadata.json:   crawl metadata (timestamp, counts, file paths)

CLI:
    - --output-dir:   directory where outputs are written (default: pypi_global_graph)
    - --max-projects: optional limit on how many projects to process
                      (useful for debugging/testing)
    - --sleep:        delay between requests (default: 0.2s)
    - --user-agent:   optional custom User-Agent string

Example debug run (process only 50 projects):
    python crawl_pypi_global.py --max-projects 50 --output-dir debug_global_graph
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Set, Tuple

# --- Third-party imports -----------------------------------------------------

try:
    import requests
except ImportError:
    print(
        "This script requires the 'requests' library.\n"
        "Install it with:\n\n    pip install requests\n",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from packaging.requirements import Requirement
except ImportError:
    print(
        "This script requires the 'packaging' library.\n"
        "Install it with:\n\n    pip install packaging\n",
        file=sys.stderr,
    )
    sys.exit(1)


# --- Name normalization (PEP 503-style) -------------------------------------

NAME_NORMALIZATION_RE = re.compile(r"[-_.]+")


def normalize_name(name: str) -> str:
    """
    Normalize package names according to PEP 503:

    - lower-case
    - replace runs of -, _ and . with a single -

    This ensures consistent node naming:
        'Scikit-Learn', 'scikit_learn', 'scikit.learn' -> 'scikit-learn'
    """
    return NAME_NORMALIZATION_RE.sub("-", name).lower()


# --- PyPI Simple Index project listing --------------------------------------

def fetch_all_project_names(session) -> List[str]:
    """
    Fetch all project names from the PyPI Simple Index.

    URL: https://pypi.org/simple/

    The page is an HTML with many <a> tags, each containing the normalized
    project name. We extract the anchor text and normalize it again to be safe.
    """
    url = "https://pypi.org/simple/"
    print(f"[INFO] Fetching project list from {url} ...", file=sys.stderr)

    try:
        resp = session.get(url, timeout=60)
    except requests.RequestException as exc:
        print(f"[ERROR] Failed to fetch project list from {url}: {exc}", file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        print(
            f"[ERROR] Unexpected HTTP {resp.status_code} when fetching project list.",
            file=sys.stderr,
        )
        sys.exit(1)

    html = resp.text

    # Simple regex to extract anchor text from <a ...>name</a>
    names = re.findall(r"<a [^>]*>([^<]+)</a>", html)

    normalized = sorted({normalize_name(n.strip()) for n in names if n.strip()})

    print(
        f"[INFO] Found {len(normalized)} projects in the Simple Index.",
        file=sys.stderr,
    )

    return normalized


# --- PyPI JSON API helpers ---------------------------------------------------

def fetch_package_metadata(name: str, session, max_retries: int = 3) -> dict | None:
    """
    Fetch JSON metadata for a package from the PyPI JSON API.

    Returns:
        dict: JSON metadata on success.
        None: if the package does not exist or repeatedly fails.
    """
    url = f"https://pypi.org/pypi/{name}/json"

    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, timeout=10)
        except requests.RequestException as exc:
            wait = 2**attempt
            print(
                f"[WARN] Request error for {name} (attempt {attempt}/{max_retries}): "
                f"{exc}. Retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except ValueError:
                print(
                    f"[WARN] Invalid JSON for package {name}, skipping.",
                    file=sys.stderr,
                )
                return None

        if resp.status_code == 404:
            # Not found on PyPI; could be stale or removed.
            print(f"[WARN] Package not found on PyPI: {name}", file=sys.stderr)
            return None

        if resp.status_code == 429 or 500 <= resp.status_code < 600:
            # Rate limit or server error – exponential backoff
            wait = 2**attempt
            print(
                f"[WARN] HTTP {resp.status_code} for {name} "
                f"(attempt {attempt}/{max_retries}). Retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

        # Other 4xx codes: treat as fatal for this package
        print(
            f"[WARN] HTTP {resp.status_code} for {name}, skipping this package.",
            file=sys.stderr,
        )
        return None

    print(
        f"[ERROR] Failed to fetch {name} after {max_retries} retries, skipping.",
        file=sys.stderr,
    )
    return None


def extract_dependencies_from_metadata(metadata: dict) -> Set[str]:
    """
    Extract a set of dependency package names from PyPI JSON metadata.

    - Uses info['requires_dist'].
    - Ignores OPTIONAL dependencies (marker uses `extra == ...`).
    - Strips version specifiers and extras, keeping only the core package name.
    """
    info = metadata.get("info", {})
    requires_dist = info.get("requires_dist") or []
    dependencies: Set[str] = set()

    for entry in requires_dist:
        # entry is a requirement string like:
        #   "numpy>=1.23"
        #   "pytest; extra == 'test'"
        #   "importlib-metadata; python_version < '3.8'"
        try:
            req = Requirement(entry)
        except Exception:
            # Some packages have malformed requirement strings; skip them.
            continue

        # Ignore optional dependencies: any marker that uses `extra == ...`
        if req.marker is not None and "extra ==" in str(req.marker):
            continue

        dep_name = normalize_name(req.name)
        if dep_name:
            dependencies.add(dep_name)

    return dependencies


# --- Global graph building ---------------------------------------------------

def build_global_dependency_graph(
    project_names: Iterable[str],
    max_projects: int | None = None,
    sleep_between: float = 0.2,
    user_agent: str | None = None,
) -> Tuple[Dict[str, Set[str]], Set[Tuple[str, str]]]:
    """
    Build a global dependency graph for the given list of project names.

    Args:
        project_names: iterable of normalized project names from the Simple Index.
        max_projects:  optional hard cap on # of projects to process
                       (useful for debugging).
        sleep_between: seconds to sleep between successful HTTP requests.
        user_agent:    optional User-Agent string for the HTTP session.

    Returns:
        adjacency: dict mapping package -> set of dependency package names.
        edges:     set of (source, target) tuples representing directed edges.
    """
    adjacency: Dict[str, Set[str]] = {}
    edges: Set[Tuple[str, str]] = set()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": user_agent
            or "pypi-global-dependency-crawler/0.1 (https://pypi.org/)"
        }
    )

    processed = 0
    total_projects = len(list(project_names))  # we may need the list twice
    # Rebuild iterator after length check
    project_list = list(project_names)

    print(
        f"[INFO] Starting global crawl over {total_projects} projects.",
        file=sys.stderr,
    )

    for name in project_list:
        if max_projects is not None and processed >= max_projects:
            print(
                f"[INFO] Reached max_projects={max_projects}, stopping crawl.",
                file=sys.stderr,
            )
            break

        pkg = normalize_name(name)
        processed += 1

        print(
            f"[INFO] ({processed}/{total_projects}) Fetching {pkg}...",
            file=sys.stderr,
        )

        metadata = fetch_package_metadata(pkg, session=session)

        if metadata is None:
            # Treat as node with no outgoing deps
            adjacency[pkg] = set()
            # Be polite anyway, even if request failed quickly
            if sleep_between > 0:
                time.sleep(sleep_between)
            continue

        deps = extract_dependencies_from_metadata(metadata)
        adjacency[pkg] = deps

        for dep in deps:
            edges.add((pkg, dep))

        # Be polite to PyPI
        if sleep_between > 0:
            time.sleep(sleep_between)

    print(
        f"[INFO] Crawl finished. "
        f"Processed {len(adjacency)} projects, {len(edges)} edges.",
        file=sys.stderr,
    )

    return adjacency, edges


# --- Saving results ----------------------------------------------------------

def save_graph_data(
    adjacency: Dict[str, Set[str]],
    edges: Set[Tuple[str, str]],
    output_dir: str,
) -> None:
    """
    Save graph data to disk in several convenient formats:

    - nodes.csv:      one package per line (column: package)
    - edges.csv:      directed edges (columns: source,target)
    - adjacency.jsonl: JSON lines with {"package": ..., "dependencies": [...]}
    - metadata.json:  crawl metadata (timestamp, counts, file paths)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all nodes: keys of adjacency + any dependency targets
    all_nodes: Set[str] = set(adjacency.keys())
    for deps in adjacency.values():
        all_nodes.update(deps)

    nodes_path = os.path.join(output_dir, "nodes.csv")
    edges_path = os.path.join(output_dir, "edges.csv")
    adjacency_path = os.path.join(output_dir, "adjacency.jsonl")
    metadata_path = os.path.join(output_dir, "metadata.json")

    # Save nodes.csv
    with open(nodes_path, "w", encoding="utf-8") as f_nodes:
        f_nodes.write("package\n")
        for node in sorted(all_nodes):
            f_nodes.write(f"{node}\n")

    # Save edges.csv
    with open(edges_path, "w", encoding="utf-8") as f_edges:
        f_edges.write("source,target\n")
        for src, dst in sorted(edges):
            f_edges.write(f"{src},{dst}\n")

    # Save adjacency.jsonl
    with open(adjacency_path, "w", encoding="utf-8") as f_adj:
        for pkg in sorted(adjacency.keys()):
            deps = sorted(adjacency[pkg])
            record = {"package": pkg, "dependencies": deps}
            f_adj.write(json.dumps(record) + "\n")

    # Save metadata.json
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "num_projects_processed": len(adjacency),
        "num_edges": len(edges),
        "output_files": {
            "nodes": os.path.abspath(nodes_path),
            "edges": os.path.abspath(edges_path),
            "adjacency": os.path.abspath(adjacency_path),
            "metadata": os.path.abspath(metadata_path),
        },
    }

    with open(metadata_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)

    print(
        f"[INFO] Saved graph data to '{output_dir}'.\n"
        f"  - nodes:      {nodes_path}\n"
        f"  - edges:      {edges_path}\n"
        f"  - adjacency:  {adjacency_path}\n"
        f"  - metadata:   {metadata_path}",
        file=sys.stderr,
    )


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl the PyPI JSON API for *all* projects (via Simple Index) to "
            "build a global directed dependency graph."
        )
    )

    parser.add_argument(
        "--output-dir",
        default="pypi_global_graph",
        help="Directory to save the graph dataset (default: pypi_global_graph).",
    )

    parser.add_argument(
        "--max-projects",
        type=int,
        default=None,
        help=(
            "Optional limit on how many projects to process (for testing/debugging). "
            "If omitted, the script attempts to process all projects from the "
            "Simple Index."
        ),
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help=(
            "Seconds to sleep between successful HTTP requests to avoid "
            "overloading PyPI (default: 0.2)."
        ),
    )

    parser.add_argument(
        "--user-agent",
        type=str,
        default=None,
        help=(
            "Optional custom User-Agent string for HTTP requests. "
            "If not set, a generic crawler UA is used."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Prepare session to fetch project list
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent
            or "pypi-global-dependency-crawler/0.1 (https://pypi.org/)"
        }
    )

    project_names = fetch_all_project_names(session)

    adjacency, edges = build_global_dependency_graph(
        project_names=project_names,
        max_projects=args.max_projects,
        sleep_between=args.sleep,
        user_agent=args.user_agent,
    )

    save_graph_data(
        adjacency=adjacency,
        edges=edges,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

