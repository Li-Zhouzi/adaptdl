#!/usr/bin/env python3

import re

# Read the file
with open('scaledown_analysis_results.txt', 'r') as f:
    content = f.read()

# Find all instances of "Removed from ready nodes" that are NOT "NOT DETECTED"
# and capture the time after cycle end from the following line
pattern = r'Removed from ready nodes: (?!NOT DETECTED).*?\n\s+Time after cycle end: ([\d.]+)s'
matches = re.findall(pattern, content)

# Convert to floats and calculate average
times = [float(time) for time in matches]

if times:
    avg_time = sum(times) / len(times)
    print(f"Found {len(times)} nodes removed from ready nodes (excluding NOT DETECTED)")
    print(f"Average time after cycle end: {avg_time:.2f}s")
    print(f"\nAll times: {sorted(times)}")
else:
    print("No matching entries found")
