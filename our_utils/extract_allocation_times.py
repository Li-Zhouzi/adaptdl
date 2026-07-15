#!/usr/bin/env python3
"""
Extract allocation times from AdaptDL scheduler logs and calculate statistics.
"""

import re
import sys
from pathlib import Path

# Threshold for categorizing allocation times (in seconds)
THRESHOLD = 10  # 10 milliseconds


def extract_allocation_times(log_file):
    """Extract allocation times from log file."""
    allocation_pattern = r"Allocations \(in ([\d.]+) sec\)"
    times = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(allocation_pattern, line)
            if match:
                time = float(match.group(1))
                times.append(time)
                print(f"Found: {time} sec")

    return times


def calculate_stats(times):
    """Calculate statistics for allocation times."""
    if not times:
        print("No allocation times found!")
        return

    # Split times based on threshold
    below_threshold = [t for t in times if t < THRESHOLD]
    above_threshold = [t for t in times if t >= THRESHOLD]

    # Overall statistics
    mean = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "="*50)
    print("ALLOCATION TIME STATISTICS")
    print("="*50)
    print(f"Threshold:         {THRESHOLD:.6f} sec ({THRESHOLD*1000:.3f} ms)")
    print(f"Total allocations: {len(times)}")
    print(f"Overall mean:      {mean:.6f} sec ({mean*1000:.3f} ms)")
    print(f"Min time:          {min_time:.6f} sec ({min_time*1000:.3f} ms)")
    print(f"Max time:          {max_time:.6f} sec ({max_time*1000:.3f} ms)")

    print("\n" + "-"*50)
    print(f"BELOW THRESHOLD (< {THRESHOLD:.6f} sec)")
    print("-"*50)
    if below_threshold:
        mean_below = sum(below_threshold) / len(below_threshold)
        print(f"Count:             {len(below_threshold)}")
        print(f"Mean:              {mean_below:.6f} sec ({mean_below*1000:.3f} ms)")
    else:
        print("No allocations below threshold")

    print("\n" + "-"*50)
    print(f"ABOVE THRESHOLD (>= {THRESHOLD:.6f} sec)")
    print("-"*50)
    if above_threshold:
        mean_above = sum(above_threshold) / len(above_threshold)
        print(f"Count:             {len(above_threshold)}")
        print(f"Mean:              {mean_above:.6f} sec ({mean_above*1000:.3f} ms)")
    else:
        print("No allocations above threshold")

    print("="*50)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_allocation_times.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]

    if not Path(log_file).exists():
        print(f"Error: File '{log_file}' not found!")
        sys.exit(1)

    print(f"Processing log file: {log_file}\n")
    times = extract_allocation_times(log_file)
    calculate_stats(times)


if __name__ == "__main__":
    main()
