import json
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_two_effective_gpu_usage(gpu_usage_dict1, gpu_usage_dict2, label1='Experiment 1', label2='Experiment 2', output_filename=None):
    """
    Plot two effective GPU usage time series in one figure for comparison.

    Args:
        gpu_usage_dict1: Dictionary mapping timestamps to GPU counts for first experiment
        gpu_usage_dict2: Dictionary mapping timestamps to GPU counts for second experiment
        label1: Label for first experiment
        label2: Label for second experiment
        output_filename: Optional filename to save the plot
    """
    if not gpu_usage_dict1 and not gpu_usage_dict2:
        print("No effective GPU usage data to plot.")
        return

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot first dataset if available
    if gpu_usage_dict1:
        sorted_items1 = sorted(gpu_usage_dict1.items())
        timestamps1 = [t for t, _ in sorted_items1]
        gpu_counts1 = [count for _, count in sorted_items1]

        start_time1 = timestamps1[0]
        relative_times1 = [(t - start_time1) for t in timestamps1]

        ax.plot(relative_times1, gpu_counts1, linewidth=1.5, color='steelblue', label=label1, alpha=0.8)
        ax.fill_between(relative_times1, gpu_counts1, alpha=0.2, color='steelblue')

    # Plot second dataset if available
    if gpu_usage_dict2:
        sorted_items2 = sorted(gpu_usage_dict2.items())
        timestamps2 = [t for t, _ in sorted_items2]
        gpu_counts2 = [count for _, count in sorted_items2]

        start_time2 = timestamps2[0]
        relative_times2 = [(t - start_time2) for t in timestamps2]

        ax.plot(relative_times2, gpu_counts2, linewidth=1.5, color='green', label=label2, alpha=0.8)
        ax.fill_between(relative_times2, gpu_counts2, alpha=0.2, color='green')

    ax.set_xlabel('Time (seconds)', fontsize=24)
    ax.set_ylabel('Number of GPUs', fontsize=24)
    # ax.set_title('Effective GPU Usage Over Time - Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=22)  # Make axis numbers (ticks) larger

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {output_filename}")
    else:
        plt.show()

    plt.close()


def load_gpu_usage_from_file(filename):
    """Load GPU usage dictionary from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    # Convert string keys back to floats (JSON keys must be strings)
    return {float(k): v for k, v in data.items()}


def main():
    """
    Main function to plot two GPU usage datasets from files.

    Usage:
        python plot_gpu_usage.py <file1> <file2> [output_filename]
        python plot_gpu_usage.py <file1> <file2> [label1] [label2] [output_filename]
    """
    if len(sys.argv) < 3:
        print("Usage: python plot_gpu_usage.py <gpu_usage_file1.json> <gpu_usage_file2.json> [label1] [label2] [output_filename]")
        print("   or: python plot_gpu_usage.py <gpu_usage_file1.json> <gpu_usage_file2.json> [output_filename]")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    # python ./our_utils/plot_gpu_usage.py ./experiment_results/monitor_log_effective_gpu_usage-fw48.json ./experiment_results/monitor_log_effective_gpu_usage-pollux0.6.json ./gpu-usage.png

    # Parse optional arguments
    label1 = 'BOA Constrictor'
    label2 = 'Pollux w/ autoscaling'
    output_filename = None

    if len(sys.argv) == 4:
        # Either output filename or label1
        if sys.argv[3].endswith('.png'):
            output_filename = sys.argv[3]
        else:
            label1 = sys.argv[3]
    elif len(sys.argv) == 5:
        # label1 and label2, or label1 and output
        if sys.argv[4].endswith('.png'):
            label1 = sys.argv[3]
            output_filename = sys.argv[4]
        else:
            label1 = sys.argv[3]
            label2 = sys.argv[4]
    elif len(sys.argv) >= 6:
        label1 = sys.argv[3]
        label2 = sys.argv[4]
        output_filename = sys.argv[5]

    print(f"Loading GPU usage data from: {file1}")
    gpu_usage_dict1 = load_gpu_usage_from_file(file1)

    print(f"Loading GPU usage data from: {file2}")
    gpu_usage_dict2 = load_gpu_usage_from_file(file2)

    plot_two_effective_gpu_usage(gpu_usage_dict1, gpu_usage_dict2, label1, label2, output_filename)


if __name__ == "__main__":
    main()
