import json
import sys
import matplotlib.pyplot as plt
from datetime import datetime


def plot_actual_vs_theoretical(dataset_file_path, output_path=None):
    """
    Parse a whitespace-separated dataset file where each line is:
        <job_name> <actual> <theoretical> <ratio>

    Produces a grouped bar chart with jobs on the x-axis, grouped by job type
    (prefix before '-') and a single bar per job: Actual/Theoretical ratio.
    """
    # Read and parse lines
    records = []  # (job_name, job_type, job_idx, ratio)
    with open(dataset_file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            # Expect at least 3 tokens: name, actual, theoretical (ratio optional)
            if len(parts) < 3:
                continue
            job_name = parts[0]
            try:
                actual = float(parts[1])
                theoretical = float(parts[2])
            except ValueError:
                # Skip headers or malformed rows
                continue

            # Prefer explicit ratio if provided, else compute
            ratio = None
            if len(parts) >= 4:
                try:
                    ratio = float(parts[3])
                except ValueError:
                    ratio = None
            if ratio is None:
                ratio = (actual / theoretical) if theoretical != 0 else None
            if ratio is None:
                continue

            if '-' in job_name:
                job_type, _, suffix = job_name.partition('-')
                # Extract numeric index if possible for stable sorting within type
                try:
                    job_idx = int(suffix)
                except ValueError:
                    job_idx = 0
            else:
                job_type = job_name
                job_idx = 0

            records.append((job_name, job_type, job_idx, ratio))

    if not records:
        print(f"No valid records parsed from: {dataset_file_path}")
        return

    # Group by type; stable sort by (job_type, job_idx)
    records.sort(key=lambda r: (r[1], r[2], r[0]))

    # Build plotting positions with gaps between groups
    positions = []
    job_labels = []
    ratio_values = []

    group_gap = 1.0  # extra spacing added between groups
    bar_width = 0.6

    x_cursor = 0.0
    current_group = None

    for job_name, job_type, job_idx, ratio in records:
        if current_group is None:
            current_group = job_type
        elif job_type != current_group:
            # Add a gap between different job types
            x_cursor += group_gap
            current_group = job_type

        positions.append(x_cursor)
        job_labels.append(job_name)
        ratio_values.append(ratio)

        # Advance x position for the next job
        x_cursor += 1.0

    # Plot
    import numpy as np
    x = np.array(positions)

    plt.figure(figsize=(16, 6))
    plt.bar(x, ratio_values, width=bar_width, color='#1f77b4', label='Actual/Theoretical')

    # Reference line at 1.0
    plt.axhline(1.0, color='gray', linestyle='--', linewidth=1)

    # X-axis setup
    plt.xticks(x, job_labels, rotation=60, ha='right')
    plt.xlabel('Jobs')
    plt.ylabel('Actual/Theoretical ratio')
    plt.title('Ratio (Actual/Theoretical) by Job (grouped by type)')
    plt.legend()
    plt.tight_layout()

    # Save/show
    out_file = output_path or 'ratio_by_job.png'
    plt.savefig(out_file, dpi=150)
    print(f"Saved plot to: {out_file}")
    plt.show()


def extract_job_logs(log_file, job_name):
    """
    Extract all log entries related to a specific job.

    Args:
        log_file: Path to the JSON log file
        job_name: Name of the job to extract (e.g., "cifar10-0")

    Returns:
        Dictionary where keys are timestamps and values are job info
    """
    job_logs = {}

    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                log_entry = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping line {line_num} - Invalid JSON: {e}", file=sys.stderr)
                continue

            timestamp = log_entry.get('timestamp')
            submitted_jobs = log_entry.get('submitted_jobs', [])

            # Find the specific job in this log entry
            for job in submitted_jobs:
                if job.get('name') == job_name:
                    job_logs[timestamp] = job
                    break

    return job_logs


def print_job_logs(job_logs, job_name):
    """Pretty print the extracted job logs."""
    if not job_logs:
        print(f"No log entries found for job: {job_name}")
        return

    print(f"Found {len(job_logs)} log entries for job: {job_name}\n")
    print("="*80)

    for timestamp, job_info in sorted(job_logs.items()):
        print(f"\nTimestamp: {timestamp}")
        print(f"Job Info:")
        for key, value in job_info.items():
            print(f"  {key}: {value}")
        print("-"*80)


if __name__ == "__main__":
    # Dataset plotting mode: read whitespace table and plot Actual/Theoretical ratio
    if len(sys.argv) >= 2 and sys.argv[1] == "--dataset":
        dataset_path = sys.argv[2] if len(sys.argv) >= 3 else \
            "/Users/lizhouzi/Documents/GitHub/adaptdl/test.py"
        plot_actual_vs_theoretical(dataset_path)
        sys.exit(0)

    print("Usage:")
    print("  python t5est2.py --dataset [data_file]")
    sys.exit(1)