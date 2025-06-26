from process_monitor import process_log_file
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python log_profile.py <log_file_path>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    result = process_log_file(log_file)

    # Print log metrics
    print("\n" + "="*50)
    print("Log Metrics")
    print("="*50)
    if result['log_metrics']['autoscaling_time'] is not None:
        print(f"Autoscaling Time: {result['log_metrics']['autoscaling_time']:.2f} seconds")
    else:
        print("Autoscaling Time: Not reached")
    if result['log_metrics']['gpu_ready_time'] is not None:
        print(f"GPU Ready Time: {result['log_metrics']['gpu_ready_time']:.2f} seconds")
    else:
        print("GPU Ready Time: Not reached")

    # Print rescaling times for each job
    print("\n" + "="*50)
    print("Rescaling Times by Job and Epoch")
    print("="*50)
    for job_name, job_info in result['jobs'].items():
        print(f"\nJob: {job_name}")
        print(f"Status: {'Completed' if job_info['is_completed'] else 'Running'}")
        print("Epoch  |  Rescaling Time (seconds)")
        print("-" * 35)
        for epoch, epoch_info in sorted(job_info['epoch_data'].items()):
            if epoch_info['rescaling_time'] > 0:  # Only print epochs with rescaling
                print(f"{epoch:5d} | {epoch_info['rescaling_time']:20.2f}")

    print("\n" + "="*50)
    print("Full job information:")
    for job_name, job_info in result['jobs'].items():
        print(f"\nJob: {job_name}")
        for epoch, epoch_info in sorted(job_info['epoch_data'].items()):
            print(f"  Epoch {epoch}: {epoch_info}")
    # print(result['jobs'])

if __name__ == "__main__":
    main()