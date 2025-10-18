#!/usr/bin/env python3
"""
Parse allocator logs to check if base_allocations nodes are subset of available nodes.
"""

import re
import ast

def parse_log(log_file):
    """Parse log file and extract Pollux optimize inputs."""

    with open(log_file, 'r') as f:
        content = f.read()

    # Find all Pollux optimize input blocks
    pattern = r"INFO:adaptdl_sched\.policy\.pollux:Pollux optimize inputs \| nodes=(\{.*?\})\s*INFO:adaptdl_sched\.policy\.pollux:Pollux optimize inputs \| base_allocations=(\{.*?\})"

    matches = re.findall(pattern, content, re.DOTALL)

    print(f"Found {len(matches)} Pollux optimize input blocks\n")

    violations = []

    for i, (nodes_str, base_alloc_str) in enumerate(matches, 1):
        # Parse the dictionaries
        try:
            nodes_dict = ast.literal_eval(nodes_str)
            base_alloc_dict = ast.literal_eval(base_alloc_str)
        except Exception as e:
            print(f"Error parsing block {i}: {e}")
            continue

        # Get all nodes from base_allocations
        alloc_nodes = set()
        for job_key, node_list in base_alloc_dict.items():
            alloc_nodes.update(node_list)

        # Get all available nodes
        available_nodes = set(nodes_dict.keys())

        # Check if alloc_nodes is subset of available_nodes
        is_subset = alloc_nodes.issubset(available_nodes)

        print(f"Block {i}:")
        print(f"  Available nodes: {len(available_nodes)}")
        print(f"  Nodes in base_allocations: {len(alloc_nodes)}")
        print(f"  Is subset: {is_subset}")

        if not is_subset:
            missing = alloc_nodes - available_nodes
            print(f"  VIOLATION: Missing nodes: {missing}")
            violations.append({
                'block': i,
                'missing_nodes': missing,
                'alloc_nodes': alloc_nodes,
                'available_nodes': available_nodes
            })

        print()

    if violations:
        print(f"\n{'='*80}")
        print(f"SUMMARY: Found {len(violations)} violations")
        print(f"{'='*80}")
        for v in violations:
            print(f"\nBlock {v['block']}:")
            print(f"  Missing nodes: {v['missing_nodes']}")
    else:
        print(f"\n{'='*80}")
        print("SUMMARY: No violations found - all base_allocations nodes are in available nodes")
        print(f"{'='*80}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python test.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]
    parse_log(log_file)
