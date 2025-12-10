#!/usr/bin/env python3
"""
Analyze scale-down events from allocator and monitor logs.

This script processes allocator logs to identify scale-down cycles and
monitor logs to track when reallocations occur and idle nodes are removed.
"""

import re
import json
from datetime import datetime
from typing import Dict, List, Tuple, Set
import sys


def parse_allocator_log(allocator_path: str) -> List[Dict]:
    """
    Parse allocator log to find scale-down cycles.

    Returns list of dicts with:
    - start_time: timestamp when cycle starts
    - end_time: timestamp when cycle ends (start of next - 60s)
    - current_nodes: number of current nodes
    - desired_nodes: number of desired nodes
    - active_nodes: list of active node names
    - allocations: dict of job -> node list
    """
    scaledown_cycles = []

    with open(allocator_path, 'r') as f:
        lines = f.readlines()

    # Find all cycle boundaries
    cycle_starts = []
    for i, line in enumerate(lines):
        timestamp_match = re.search(r'\[TIMESTAMP: ([\d.]+)\]', line)
        if timestamp_match:
            cycle_starts.append((i, float(timestamp_match.group(1))))

    # Process each cycle
    for cycle_idx in range(len(cycle_starts)):
        start_line, start_time = cycle_starts[cycle_idx]

        # Determine cycle end line
        if cycle_idx < len(cycle_starts) - 1:
            end_line = cycle_starts[cycle_idx + 1][0]
        else:
            end_line = len(lines)

        # Look for autoscaling in this cycle
        autoscale_found = False
        for i in range(start_line, end_line):
            autoscale_match = re.search(
                r'Allocator side: Starting autoscaling: current=(\d+), desired=(\d+)',
                lines[i]
            )

            if autoscale_match:
                current = int(autoscale_match.group(1))
                desired = int(autoscale_match.group(2))

                # Only process scale-down (current > desired)
                if current > desired:
                    cycle_info = {
                        'start_time': start_time,
                        'cycle_idx': cycle_idx,  # Store index to find next cycle
                        'current_nodes': current,
                        'desired_nodes': desired,
                    }

                    # Find active nodes (should be shortly after autoscaling line)
                    for j in range(i, min(i + 10, end_line)):
                        active_match = re.search(r"Active nodes: (\[.*?\])", lines[j])
                        if active_match:
                            active_nodes_str = active_match.group(1)
                            # Parse the list
                            active_nodes = eval(active_nodes_str)
                            cycle_info['active_nodes'] = active_nodes
                            break

                    # Find allocations (should be shortly after active nodes)
                    for j in range(i, min(i + 20, end_line)):
                        alloc_match = re.search(r"Allocations \(in [\d.]+ sec\): (\{.*\})", lines[j])
                        if alloc_match:
                            alloc_str = alloc_match.group(1)
                            # Parse the allocation dict
                            allocations = eval(alloc_str)
                            # Convert tuple keys to strings
                            cycle_info['allocations'] = {
                                f"{k[0]}/{k[1]}": v for k, v in allocations.items()
                            }
                            break

                    scaledown_cycles.append(cycle_info)
                    autoscale_found = True
                break

    # Calculate end times using the next allocation cycle (not next scaledown cycle)
    for cycle in scaledown_cycles:
        cycle_idx = cycle['cycle_idx']
        # Find the next allocation cycle start time
        if cycle_idx < len(cycle_starts) - 1:
            next_cycle_start = cycle_starts[cycle_idx + 1][1]
            cycle['end_time'] = next_cycle_start - 60
        else:
            # For the last cycle, estimate end time as start + 60
            cycle['end_time'] = cycle['start_time'] + 60

        # Remove the temporary cycle_idx field
        del cycle['cycle_idx']

    return scaledown_cycles


def parse_monitor_log(monitor_path: str) -> List[Dict]:
    """
    Parse monitor log to get job allocations and node states over time.

    Returns list of dicts with timestamp, jobs, and ready_nodes.
    """
    monitor_data = []

    with open(monitor_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                monitor_data.append({
                    'timestamp': data['timestamp'],
                    'jobs': {
                        job['name']: job['allocation']
                        for job in data.get('submitted_jobs', [])
                        if 'allocation' in job
                    },
                    'ready_nodes': set(data['cluster_nodes'].get('ready_node_names', []))
                })
            except json.JSONDecodeError:
                continue

    return monitor_data


def find_reallocation_times(cycle: Dict, monitor_data: List[Dict]) -> Dict:
    """
    Find reallocation timestamps for a scale-down cycle.

    Returns dict with:
    - first_realloc_time: when first job matches computed allocation
    - all_realloc_time: when all jobs match computed allocation
    - idle_node_info: dict of node -> {stop_used_time, removed_time}
    """
    cycle_end = cycle['end_time']
    target_allocations = cycle['allocations']
    active_nodes = set(cycle['active_nodes'])

    # Find idle nodes (will be in ready nodes but not in active nodes initially)
    # We need to look at monitor data around cycle end to find which nodes are ready
    idle_nodes = set()
    for entry in monitor_data:
        if abs(entry['timestamp'] - cycle_end) < 5:  # Within 5 seconds of cycle end
            ready_nodes = entry['ready_nodes']
            idle_nodes = ready_nodes - active_nodes
            break

    results = {
        'first_realloc_time': None,
        'all_realloc_time': None,
        'idle_node_info': {}
    }

    # Look for reallocations after cycle end
    timeout_time = cycle_end + 120
    matched_jobs = set()

    for entry in monitor_data:
        if entry['timestamp'] < cycle_end:
            continue
        if entry['timestamp'] > timeout_time:
            break

        current_jobs = entry['jobs']

        # Check which jobs match target allocation
        for job_name, target_alloc in target_allocations.items():
            # Extract just the job name (remove namespace)
            job_short_name = job_name.split('/')[-1]

            if job_short_name in current_jobs:
                current_alloc = current_jobs[job_short_name]

                # Compare allocations (as sorted lists)
                if sorted(current_alloc) == sorted(target_alloc):
                    if job_short_name not in matched_jobs:
                        matched_jobs.add(job_short_name)

                        # First reallocation
                        if results['first_realloc_time'] is None:
                            results['first_realloc_time'] = entry['timestamp']

        # Check if all jobs matched
        if len(matched_jobs) == len(target_allocations):
            if results['all_realloc_time'] is None:
                results['all_realloc_time'] = entry['timestamp']

    # Track idle nodes
    for node in idle_nodes:
        node_info = {
            'stop_used_time': None,
            'removed_time': None
        }

        # Find when node stops being used in allocations
        for entry in monitor_data:
            if entry['timestamp'] < cycle_end:
                continue

            # Check if node is in any job allocation
            node_in_use = False
            for job_name, alloc in entry['jobs'].items():
                if node in alloc:
                    node_in_use = True
                    break

            if not node_in_use and node_info['stop_used_time'] is None:
                node_info['stop_used_time'] = entry['timestamp']

            # Find when node disappears from ready_nodes
            if node not in entry['ready_nodes'] and node_info['removed_time'] is None:
                node_info['removed_time'] = entry['timestamp']
                break

        results['idle_node_info'][node] = node_info

    # Check for timeout
    if results['all_realloc_time'] is None:
        if results['first_realloc_time'] is not None:
            print(f"WARNING: Not all jobs reallocated within 120s for cycle at {cycle['start_time']}")
        else:
            print(f"ERROR: No reallocations detected within 120s for cycle at {cycle['start_time']}")

    return results


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python analyze_scaledown.py <allocator_log> <monitor_log> [output_file]")
        sys.exit(1)

    allocator_path = sys.argv[1]
    monitor_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) == 4 else None

    print("Parsing allocator log...")
    scaledown_cycles = parse_allocator_log(allocator_path)
    print(f"Found {len(scaledown_cycles)} scale-down cycles")

    print("\nParsing monitor log...")
    monitor_data = parse_monitor_log(monitor_path)
    print(f"Loaded {len(monitor_data)} monitor entries")

    # Redirect output to file if specified
    import sys as sys_module
    original_stdout = sys_module.stdout
    if output_file:
        sys_module.stdout = open(output_file, 'w')

    print("\n" + "="*80)
    print("SCALE-DOWN ANALYSIS")
    print("="*80)

    for i, cycle in enumerate(scaledown_cycles):
        print(f"\n--- Scale-down Cycle {i+1} ---")
        print(f"Start time: {cycle['start_time']} ({datetime.fromtimestamp(cycle['start_time'])})")
        print(f"End time: {cycle['end_time']} ({datetime.fromtimestamp(cycle['end_time'])})")
        print(f"Scaling: {cycle['current_nodes']} -> {cycle['desired_nodes']} nodes")
        print(f"Active nodes ({len(cycle['active_nodes'])}): {cycle['active_nodes']}")

        # Analyze reallocations
        realloc_info = find_reallocation_times(cycle, monitor_data)

        if realloc_info['first_realloc_time']:
            print(f"\nFirst reallocation: {realloc_info['first_realloc_time']} "
                  f"({datetime.fromtimestamp(realloc_info['first_realloc_time'])})")
            print(f"  Time after cycle end: {realloc_info['first_realloc_time'] - cycle['end_time']:.2f}s")
        else:
            print("\nFirst reallocation: NOT DETECTED")

        if realloc_info['all_realloc_time']:
            print(f"All reallocations complete: {realloc_info['all_realloc_time']} "
                  f"({datetime.fromtimestamp(realloc_info['all_realloc_time'])})")
            print(f"  Time after cycle end: {realloc_info['all_realloc_time'] - cycle['end_time']:.2f}s")
        else:
            print("All reallocations complete: NOT DETECTED (timeout)")

        print(f"\nIdle nodes ({len(realloc_info['idle_node_info'])} nodes):")
        for node, info in realloc_info['idle_node_info'].items():
            print(f"  {node}:")
            if info['stop_used_time']:
                print(f"    Stopped being used: {info['stop_used_time']} "
                      f"({datetime.fromtimestamp(info['stop_used_time'])})")
                print(f"      Time after cycle end: {info['stop_used_time'] - cycle['end_time']:.2f}s")
            else:
                print(f"    Stopped being used: NOT DETECTED")

            if info['removed_time']:
                print(f"    Removed from ready nodes: {info['removed_time']} "
                      f"({datetime.fromtimestamp(info['removed_time'])})")
                print(f"      Time after cycle end: {info['removed_time'] - cycle['end_time']:.2f}s")
            else:
                print(f"    Removed from ready nodes: NOT DETECTED")

        print()

    # Restore stdout and close file if needed
    if output_file:
        sys_module.stdout.close()
        sys_module.stdout = original_stdout
        print(f"\nResults written to {output_file}")


if __name__ == '__main__':
    main()
