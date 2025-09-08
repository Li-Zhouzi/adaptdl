import json
import sys
import csv
from datetime import datetime
import numpy as np

NUM_GPU_PER_NODE = 4


def process_log_file(log_file_path):
    """Process log file and extract simplified job metrics."""
    jobs = {}
    total_gpu_hours = 0
    effective_gpu_hours = 0
    waiting_for_ready_hours = 0
    wasted_capacity_hours = 0
    theoretical_rescaling_hours = 0
    high_waste_examples = []
    last_job_arrival_time = None
    previous_allocations = {}  # Track previous GPU allocations for each job
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        log_entry = json.loads(line.strip())
        timestamp = log_entry['timestamp']
        
        # Calculate GPU hours for this time step
        if i > 0:
            prev_log = json.loads(lines[i-1].strip())
            time_diff = timestamp - prev_log['timestamp']
            
            # Extract cluster node information
            cluster_nodes = prev_log.get('cluster_nodes', {})
            total_nodes = cluster_nodes.get('total', 0)
            ready_nodes = cluster_nodes.get('ready', 0)
            
            # Calculate GPU metrics
            prev_total_gpus = total_nodes * NUM_GPU_PER_NODE
            ready_gpus = ready_nodes * NUM_GPU_PER_NODE
            total_gpu_hours += prev_total_gpus * time_diff / 3600  # Convert to hours
            
            # Count effective GPUs (only those actually allocated)
            prev_effective_gpus = sum(len(job.get('allocation', [])) for job in prev_log['submitted_jobs'] if job.get('allocation'))
            effective_gpu_hours += prev_effective_gpus * time_diff / 3600
            
            # Calculate decomposition metrics
            # WaitingForReady: total - ready
            waiting_for_ready_gpus = prev_total_gpus - ready_gpus
            waiting_for_ready_hours += waiting_for_ready_gpus * time_diff / 3600
            
            # Wasted Capacity: ready - effective (since effective includes all allocated)
            wasted_capacity_gpus = ready_gpus - prev_effective_gpus
            wasted_capacity_hours += wasted_capacity_gpus * time_diff / 3600
            
            # Track examples of high waste (>= 50 wasted GPUs)
            if wasted_capacity_gpus >= 50:
                # Get job allocations for this timestamp
                job_info = []
                for job in prev_log['submitted_jobs']:
                    allocation = job.get('allocation', [])
                    if allocation:
                        gpu_count = len(allocation)
                        job_info.append(f"{job['name']} using {set(allocation)} ({gpu_count} GPUs)")
                
                high_waste_examples.append({
                    'timestamp': prev_log['timestamp'],
                    'ready_nodes': ready_nodes,
                    'wasted_gpus': wasted_capacity_gpus,
                    'jobs': job_info
                })
        
        for job in log_entry['submitted_jobs']:
            job_name = job['name']
            epoch = job['epoch']
            allocation = job.get('allocation', [])
            progress = job.get('progress', 0)
            
            # Check for GPU allocation changes and calculate theoretical rescaling hours
            current_gpu_count = len(allocation)
            if job_name in previous_allocations:
                previous_gpu_count = len(previous_allocations[job_name])
                if current_gpu_count != previous_gpu_count and current_gpu_count > 0:
                    # GPU allocation changed, add theoretical rescaling cost
                    theoretical_rescaling_hours += (current_gpu_count * 50) / 3600
            
            # Update previous allocation for this job
            if current_gpu_count > 0:
                previous_allocations[job_name] = allocation
            
            # Initialize job if first time seeing it
            if job_name not in jobs:
                jobs[job_name] = {
                    'first_seen': timestamp,
                    'epochs': {}
                }
                # Track the latest job arrival time
                last_job_arrival_time = timestamp
            
            # Initialize epoch if first time seeing it
            if epoch not in jobs[job_name]['epochs']:
                jobs[job_name]['epochs'][epoch] = {
                    'first_seen': timestamp,
                    'last_seen': timestamp,
                    'gpu_allocations': [],
                    'progress_history': [],
                    'wasted_time': 0,
                    'last_progress': None,
                    'stuck_start_time': None
                }
            
            epoch_info = jobs[job_name]['epochs'][epoch]
            epoch_info['last_seen'] = timestamp
            
            # Track GPU allocations
            gpu_count = len(allocation)
            if gpu_count not in epoch_info['gpu_allocations']:
                epoch_info['gpu_allocations'].append(gpu_count)
            
            # Track progress and calculate wasted time
            # Wasted time occurs when: progress is null OR progress is stuck at same value
            is_wasted = False
            
            if progress is None:
                # Null progress is always wasted time
                is_wasted = True
            elif epoch_info['last_progress'] is not None and progress == epoch_info['last_progress']:
                # Progress stuck at same value
                # For epoch 0, only count as wasted if progress > 0 (startup time at 0 is normal)
                if epoch != 0 or progress > 0:
                    is_wasted = True
            
            if is_wasted:
                if epoch_info['stuck_start_time'] is None:
                    epoch_info['stuck_start_time'] = timestamp
            else:
                # Progress is making progress, end any wasted time period
                if epoch_info['stuck_start_time'] is not None:
                    epoch_info['wasted_time'] += timestamp - epoch_info['stuck_start_time']
                    epoch_info['stuck_start_time'] = None
            
            epoch_info['last_progress'] = progress
            epoch_info['progress_history'].append((timestamp, progress))
    
    # Calculate final epoch durations and handle any remaining stuck time
    for job_name, job_info in jobs.items():
        for epoch_num, epoch_info in job_info['epochs'].items():
            epoch_info['duration'] = epoch_info['last_seen'] - epoch_info['first_seen']
            # Add any remaining stuck time
            if epoch_info['stuck_start_time'] is not None:
                epoch_info['wasted_time'] += epoch_info['last_seen'] - epoch_info['stuck_start_time']
    
    # Calculate rescaling hours from existing wasted time data
    rescaling_hours = 0
    for job_name, job_info in jobs.items():
        for epoch_num, epoch_info in job_info['epochs'].items():
            # Convert wasted time from seconds to hours and multiply by GPU count
            wasted_seconds = epoch_info['wasted_time']
            gpu_allocations = epoch_info['gpu_allocations']
            if gpu_allocations:
                # Use the maximum GPU allocation for this epoch
                max_gpus = max(gpu_allocations)
                rescaling_hours += (wasted_seconds * max_gpus) / 3600
    
    return jobs, total_gpu_hours, effective_gpu_hours, waiting_for_ready_hours, wasted_capacity_hours, rescaling_hours, theoretical_rescaling_hours, high_waste_examples, last_job_arrival_time

def calculate_metrics(total_gpu_hours, effective_gpu_hours, waiting_for_ready_hours, wasted_capacity_hours, rescaling_hours, theoretical_rescaling_hours, last_job_arrival_time, first_job_time):
    """Calculate simplified metrics based on GPU hours and job arrival time."""
    experiment_duration_hours = (last_job_arrival_time - first_job_time) / 3600
    average_gpu_usage = total_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    effective_average_gpu_usage = effective_gpu_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    waiting_for_ready_average = waiting_for_ready_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    wasted_capacity_average = wasted_capacity_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    rescaling_average = rescaling_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    theoretical_rescaling_average = theoretical_rescaling_hours / experiment_duration_hours if experiment_duration_hours > 0 else 0
    
    return {
        'total_gpu_hours': total_gpu_hours,
        'effective_gpu_hours': effective_gpu_hours,
        'waiting_for_ready_hours': waiting_for_ready_hours,
        'wasted_capacity_hours': wasted_capacity_hours,
        'rescaling_hours': rescaling_hours,
        'theoretical_rescaling_hours': theoretical_rescaling_hours,
        'experiment_duration_hours': experiment_duration_hours,
        'average_gpu_usage': average_gpu_usage,
        'effective_average_gpu_usage': effective_average_gpu_usage,
        'waiting_for_ready_average': waiting_for_ready_average,
        'wasted_capacity_average': wasted_capacity_average,
        'rescaling_average': rescaling_average,
        'theoretical_rescaling_average': theoretical_rescaling_average,
        'last_job_arrival_time': last_job_arrival_time
    }

def print_summary(metrics):
    """Print summary of simplified metrics."""
    print("\n" + "="*60)
    print("LOG PROCESSING SUMMARY")
    print("="*60)
    
    print(f"Total GPU Hours: {metrics['total_gpu_hours']:.2f}")
    print(f"Effective GPU Hours: {metrics['effective_gpu_hours']:.2f}")
    print(f"Experiment Duration: {metrics['experiment_duration_hours']:.2f} hours")
    print(f"Average GPU Usage: {metrics['average_gpu_usage']:.2f} GPUs")
    print(f"Effective Average GPU Usage: {metrics['effective_average_gpu_usage']:.2f} GPUs")
    
    print(f"\nDecomposition:")
    # print(f"WaitingForReady Hours: {metrics['waiting_for_ready_hours']:.2f}")
    print(f"WaitingForReady Average: {metrics['waiting_for_ready_average']:.2f} GPUs")
    # print(f"Wasted Capacity Hours: {metrics['wasted_capacity_hours']:.2f}")
    print(f"Wasted Capacity Average: {metrics['wasted_capacity_average']:.2f} GPUs")
    print(f"Rescaling Average: {metrics['rescaling_average']:.2f} GPUs")
    print(f"Theoretical Rescaling Average: {metrics['theoretical_rescaling_average']:.2f} GPUs")
    
    print(f"\nLast Job Arrival Time: {datetime.fromtimestamp(metrics['last_job_arrival_time'])}")

def print_high_waste_examples(high_waste_examples, limit=10):
    """Print examples of timestamps with high wasted capacity."""
    if not high_waste_examples:
        print(f"\nNo high waste examples found (threshold >= 50 GPUs)")
        return
    
    print(f"\n" + "="*80)
    print(f"HIGH WASTE EXAMPLES (showing up to {limit})")
    print("="*80)
    
    # Sort by wasted GPUs descending and take top examples
    sorted_examples = sorted(high_waste_examples, key=lambda x: x['wasted_gpus'], reverse=True)[:limit]
    
    for example in sorted_examples:
        print(f"\nTimestamp: {example['timestamp']}")
        print(f"Ready Nodes: {example['ready_nodes']}, Wasted GPUs: {example['wasted_gpus']:.0f}")
        
        if example['jobs']:
            print("Jobs:")
            for job_info in example['jobs']:
                print(f"  - {job_info}")
        else:
            print("Jobs: None running")

def print_all_jobs_summary(jobs):
    """Print response time and wasted time for all jobs."""
    print(f"\n" + "="*60)
    print("ALL JOBS SUMMARY")
    print("="*60)
    print(f"{'Job Name':<20} {'Response Time(s)':<18} {'Total Wasted(s)':<15}")
    print("-" * 60)
    
    for job_name, job_info in jobs.items():
        # Calculate total response time (from first seen to completion of last epoch)
        if job_info['epochs']:
            last_epoch_end = max(epoch_info['last_seen'] for epoch_info in job_info['epochs'].values())
            response_time = last_epoch_end - job_info['first_seen']
        else:
            response_time = 0
        
        # Calculate total wasted time across all epochs
        total_wasted = sum(epoch_info['wasted_time'] for epoch_info in job_info['epochs'].values())
        
        print(f"{job_name:<20} {response_time:<18.1f} {total_wasted:<15.1f}")

def get_theoretical_duration(application, epoch, num_gpus):
    """Calculate theoretical duration using goodput function."""
    goodput_functions = {
            'cifar10': {
                0: {1: 713.5061271835817, 2: 880.5349482251487, 4: 943.1826910951117, 8: 2084.8023190467325, 12: 4419.635957136727, 16: 3420.4277146058444},
                1: {1: 713.5061271835817, 2: 880.5349482251487, 4: 943.1826910951117, 8: 2084.8023190467325, 12: 4419.635957136727, 16: 3420.4277146058444},
                2: {1: 713.5061271835817, 2: 880.5349482251487, 4: 943.1826910951117, 8: 2084.8023190467325, 12: 4419.635957136727, 16: 3420.4277146058444},
                3: {1: 713.5061271835817, 2: 880.5349482251487, 4: 943.1826910951117, 8: 2084.8023190467325, 12: 4419.635957136727, 16: 3420.4277146058444},
                4: {1: 713.5061271835817, 2: 880.5349482251487, 4: 943.1826910951117, 8: 2084.8023190467325, 12: 4419.635957136727, 16: 3420.4277146058444},
                5: {1: 1013.3285519160074, 2: 1563.3199123933398, 4: 1613.6037759869673, 8: 2185.3812384894795, 12: 2544.3075977660924, 16: 3147.069060538767},
                6: {1: 694.9225418498365, 2: 874.439632502621, 4: 1695.9170945069652, 8: 1576.2912196585182, 12: 1430.0414716745513, 16: 3304.4069778848934},
                7: {1: 696.6850215856377, 2: 1836.3548009745828, 4: 894.2989766451667, 8: 2333.1264117757096, 12: 3076.105684866412, 16: 2252.6396644121673},
                8: {1: 1015.55709075265, 2: 1058.2821351392322, 4: 1573.5721940337858, 8: 2195.128004316584, 12: 3082.877061642209, 16: 2409.2321794763275},
                9: {1: 710.4718714455266, 2: 1121.8567845890025, 4: 863.8039652944686, 8: 2082.1330744222546, 12: 3033.089636387403, 16: 3746.2878353899328},
                10: {1: 756.483900236882, 2: 1363.6705962806313, 4: 1783.4543321235014, 8: 1881.656021530399, 12: 2181.509352337989, 16: 3722.0828302363216},
                11: {1: 931.3691404645041, 2: 925.9894758701895, 4: 1034.010917052346, 8: 2325.0662983904413, 12: 1889.6318208883577, 16: 4071.9089325420773},
                12: {1: 722.0571127402129, 2: 1478.1001225437326, 4: 1370.7782501692268, 8: 2345.707323340159, 12: 3260.0393735090615, 16: 3712.6429325890977},
                13: {1: 992.6451555840993, 2: 888.2978289016203, 4: 1785.7732840975889, 8: 1714.3763375989354, 12: 3281.0473670830543, 16: 2150.7633687406874},
                14: {1: 729.8766188282707, 2: 1473.3644016221936, 4: 926.2303489817891, 8: 2333.736445863275, 12: 3274.1564256384618, 16: 2672.755570577299},
                15: {1: 700.7177676649774, 2: 829.3412856692094, 4: 1583.1045452589278, 8: 2451.1465906168914, 12: 2261.5186557506677, 16: 3756.2855682773816},
                16: {1: 1003.0355865102649, 2: 1586.5322930107461, 4: 880.1495002472969, 8: 2317.7908299743367, 12: 2084.578448174148, 16: 4095.563246575658},
                17: {1: 709.9699765065382, 2: 887.1687007197801, 4: 1589.4671486523603, 8: 1906.1263979235362, 12: 3280.46253390465, 16: 4116.297931399104},
                18: {1: 690.7908232274343, 2: 1581.157503147082, 4: 1146.8717138518675, 8: 2451.786115976884, 12: 3278.702672144429, 16: 4123.193183827393},
                19: {1: 1015.8074435279848, 2: 1255.8345139188507, 4: 1096.1570880283211, 8: 2505.967462639833, 12: 3337.9267750144636, 16: 2426.7721633928522},
                20: {1: 716.6253638574977, 2: 1043.6871138553586, 4: 1479.0199859230352, 8: 1879.9229210375217, 12: 1985.5304799538064, 16: 2876.7422044599275},
                21: {1: 808.5516277351122, 2: 1537.3064853211433, 4: 876.2036094782934, 8: 2363.059074099316, 12: 1820.7399985935217, 16: 4131.4061507115985},
                22: {1: 875.7805956362251, 2: 854.8514808301119, 4: 1590.688588230504, 8: 2354.6142322240353, 12: 3094.438364878578, 16: 4138.4284291834},
                23: {1: 702.8695961582572, 2: 1596.5616893178649, 4: 871.0406259248814, 8: 2530.314060160523, 12: 3299.76699700547, 16: 4547.45469305919},
                24: {1: 993.5573729836547, 2: 869.4395488522014, 4: 1667.9215468161547, 8: 1846.7734868586845, 12: 3295.961950287892, 16: 3990.414034340228},
                25: {1: 738.824115272221, 2: 1581.882406512472, 4: 946.2914302735854, 8: 2540.6314379274536, 12: 1818.2253414515387, 16: 3765.0968182731517},
                26: {1: 680.735944854634, 2: 892.054293695062, 4: 1473.9969757629835, 8: 2355.8873255258795, 12: 2504.6861227538793, 16: 2596.8682379672364},
                27: {1: 1020.6249706354336, 2: 1588.4340208847952, 4: 1585.6397811679312, 8: 2134.592822138087, 12: 3323.9279488684115, 16: 3449.0196843341255},
                28: {1: 702.1776933130361, 2: 1522.055979193602, 4: 894.2553053884606, 8: 1761.7854758666535, 12: 3315.9271109271626, 16: 4593.554135777327},
                29: {1: 703.6985864475046, 2: 896.1556704007412, 4: 1589.9280498635046, 8: 2428.2718823602017, 12: 3312.0584143481174, 16: 4523.036287219119},
                30: {1: 1035.2191930595643, 2: 1534.2635066289681, 4: 891.3906923955769, 8: 2426.0208373954474, 12: 2107.5960870294703, 16: 4062.711396544063},
                31: {1: 710.9518272566653, 2: 893.3602301180275, 4: 1719.3211670840808, 8: 1632.2643057227233, 12: 2436.5100449319375, 16: 5079.811325695178},
                32: {1: 814.3476048221712, 2: 1529.818301542142, 4: 1156.8974603230172, 8: 2401.2661644624673, 12: 3384.3464937484014, 16: 4023.2742109644787},
                33: {1: 871.6599156712491, 2: 857.0138689447705, 4: 1159.3983555604534, 8: 2368.6619617930555, 12: 5761.542084616056, 16: 2856.194752486851},
                34: {1: 704.4198888996386, 2: 1589.2697977044074, 4: 1720.8912727135644, 8: 2165.8889103660463, 12: 6264.108343716254, 16: 3118.892982401557},
                35: {1: 1038.2940286166463, 2: 1063.7081634135786, 4: 892.7509847070368, 8: 1957.6945335907258, 12: 4488.825155188222, 16: 4511.348682982798},
                36: {1: 709.7644618512871, 2: 1162.0580321525301, 4: 1645.9597129346653, 8: 2384.131669470253, 12: 5059.651183446736, 16: 4574.135949564002},
                37: {1: 706.1541561387762, 2: 1534.8586849275412, 4: 949.4344329757879, 8: 2386.4760409937053, 12: 2233.1188345842934, 16: 4484.758582327923},
                38: {1: 1005.2880165276738, 2: 857.2799746227547, 4: 1479.0213523608534, 8: 1774.1271588640507, 12: 3874.442287944899, 16: 4204.801455024527},
                39: {1: 730.0408089690358, 2: 1587.3767747111256, 4: 1591.4328021279914, 8: 2426.274006672175, 12: 5729.001848008714, 16: 4566.176222597607},
                40: {1: 757.3442514356104, 2: 855.7164353601974, 4: 874.7375920072681, 8: 2418.4568042680767, 12: 6701.875647769067, 16: 2757.330628286074},
                41: {1: 962.9248681245541, 2: 1595.3618122179198, 4: 1587.3767747111256, 8: 2347.4283033240054, 12: 5677.1440507301195, 16: 3276.655468775182},
                42: {1: 714.9463647043015, 2: 875.2667173644617, 4: 891.3292113386487, 8: 1812.7891554347725, 12: 6743.003812655955, 16: 4608.939296427511},
                43: {1: 949.6729514957204, 2: 1567.836024342146, 4: 1595.4716659824464, 8: 2401.7211247522496, 12: 5747.98930200485, 16: 4591.843895454643},
                44: {1: 758.0862930740197, 2: 1195.7027789334197, 4: 912.6182054640684, 8: 2413.6312380105487, 12: 6745.551193977344, 16: 4149.940308118155},
                45: {1: 700.2332532623279, 2: 1112.9513426935964, 4: 1692.9983229633065, 8: 1929.6240647993855, 12: 3674.07754576374, 16: 4563.868450597137},
                46: {1: 999.8755943589993, 2: 1556.1745338808514, 4: 1597.8728147482434, 8: 2269.198794998862, 12: 3113.115453906615, 16: 4462.487636751661},
                47: {1: 683.9920883669644, 2: 867.8229467917926, 4: 920.2592469828743, 8: 2142.7161080878504, 12: 4462.018031927138, 16: 2941.2948184130337},
                48: {1: 687.613804008957, 2: 1573.5626421435786, 4: 1562.980407429696, 8: 2546.5116945507093, 12: 5727.463504746049, 16: 2938.5706141425267},
                49: {1: 1022.2760920996299, 2: 863.8242567897391, 4: 866.7117607902453, 8: 1787.1834098133306, 12: 6639.220213779427, 16: 4557.75586021461},
                50: {1: 710.691694761043, 2: 1524.4828188882045, 4: 1582.539149818506, 8: 2412.7670188414586, 12: 6654.656363792485, 16: 5114.5302246950005},
                51: {1: 699.0957606735493, 2: 847.1570291179205, 4: 941.5461043729272, 8: 2365.62807667283, 12: 5699.406291009073, 16: 4567.806901323115},
                52: {1: 1036.6236464268072, 2: 1579.979861952918, 4: 1410.9937051029804, 8: 2108.994892096979, 12: 6674.169922488811, 16: 4553.65406206552},
                53: {1: 1036.6236464268072, 2: 1005.6173733382694, 4: 906.0351365633836, 8: 2517.8803370992105, 12: 3591.5531080875176, 16: 3349.6575642659577},
                54: {1: 1036.6236464268072, 2: 1005.6173733382694, 4: 906.0351365633836, 8: 2517.8803370992105, 12: 3591.5531080875176, 16: 3349.6575642659577},
                55: {1: 1036.6236464268072, 2: 1521.8458784611928, 4: 1556.6027624497374, 8: 2111.1241585749044, 12: 3311.685711242947, 16: 3152.8510787997784},
                56: {1: 1036.6236464268072, 2: 878.6752448260327, 4: 882.5097498792213, 8: 2505.8405930829067, 12: 4916.516935153427, 16: 4086.696459908779},
                57: {1: 1036.6236464268072, 2: 1539.6087097537772, 4: 1509.50450690477, 8: 2391.196037684251, 12: 6278.9371179635755, 16: 4575.186241525011},
                58: {1: 1036.6236464268072, 2: 879.6078453051308, 4: 902.1402903621242, 8: 2137.5317706747333, 12: 6397.301138582161, 16: 4551.34687641046},
                59: {1: 1036.6236464268072, 2: 1558.7016485118834, 4: 1534.5981393544912, 8: 2529.1000898470948, 12: 6638.579417849862, 16: 4889.98608565189},
                60: {1: 1036.6236464268072, 2: 1580.1771507136891, 4: 1553.7497703254562, 8: 2425.8724127088735, 12: 3593.1692925713223, 16: 2934.6576220646525},
                61: {1: 1036.6236464268072, 2: 1580.1771507136891, 4: 1553.7497703254562, 8: 2425.8724127088735, 12: 3593.1692925713223, 16: 2934.6576220646525},
                62: {1: 1036.6236464268072, 2: 1580.1771507136891, 4: 1553.7497703254562, 8: 2425.8724127088735, 12: 3593.1692925713223, 16: 2934.6576220646525},
                63: {1: 1036.6236464268072, 2: 883.4451827831449, 4: 922.3471885935394, 8: 2545.1352852527602, 12: 1805.0852374009964, 16: 3356.706127113265},
                64: {1: 1036.6236464268072, 2: 1561.3746906969309, 4: 1528.4959023572253, 8: 2409.93040919606, 12: 3576.205086516148, 16: 4541.257626708091},
                65: {1: 1036.6236464268072, 2: 857.8569474763999, 4: 883.6140043334481, 8: 2533.137119826264, 12: 3599.308749393508, 16: 5119.899751962865},
                66: {1: 1036.6236464268072, 2: 1593.1464994332016, 4: 1695.2848870526566, 8: 2571.0955389453816, 12: 3546.4098821353155, 16: 4567.572113676256},
                67: {1: 1036.6236464268072, 2: 858.3229466526286, 4: 1393.4130650241734, 8: 2408.092846941643, 12: 3320.4768783864292, 16: 4519.6110502493675},
                68: {1: 1036.6236464268072, 2: 1550.1117503739642, 4: 962.5379830716114, 8: 2560.546804269945, 12: 2104.3444388575713, 16: 5005.91392614733},
                69: {1: 1036.6236464268072, 2: 948.6810563838819, 4: 1655.1207147841706, 8: 2570.581747044701, 12: 2334.235100498011, 16: 2909.601958951142},
                70: {1: 1036.6236464268072, 2: 1376.5237996940846, 4: 875.1841926799408, 8: 2409.820996830815, 12: 3614.312335746624, 16: 2693.626669466705},
                71: {1: 1036.6236464268072, 2: 1534.8958006077, 4: 1597.1452758499886, 8: 2562.8622591319336, 12: 3781.9100099776037, 16: 5133.2995821522845},
                72: {1: 1036.6236464268072, 2: 869.3602455707896, 4: 871.778728914445, 8: 2545.4327728620556, 12: 3300.2137272488435, 16: 4562.206242242554},
                73: {1: 1036.6236464268072, 2: 1586.2325058307783, 4: 1714.4537440346833, 8: 2396.5234028174946, 12: 3340.972406125679, 16: 4451.136536665686},
                74: {1: 1036.6236464268072, 2: 887.4271198290112, 4: 1655.0808225826315, 8: 2241.3739222679887, 12: 2090.5877634814447, 16: 5006.424684730902},
                75: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 930.908698347273, 8: 2492.886931367583, 12: 3346.154702852683, 16: 4526.999533517552},
                76: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1581.3330863103618, 8: 2565.6238901605975, 12: 3700.6434223695605, 16: 4369.2990587603545},
                77: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1581.3330863103618, 8: 2565.6238901605975, 12: 3700.6434223695605, 16: 4369.2990587603545},
                78: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1581.3330863103618, 8: 2565.6238901605975, 12: 3700.6434223695605, 16: 4369.2990587603545},
                79: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2276.987715602459, 12: 2611.4156566469164, 16: 4567.3083048314675},
                80: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2580.507982383434, 12: 3680.7438038721707, 16: 4421.138467104012},
                81: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2580.507982383434, 12: 3680.7438038721707, 16: 4421.138467104012},
                82: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2269.8892804016778, 12: 3298.799072489607, 16: 5026.9362651055735},
                83: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2436.8198568506377, 12: 3689.1052381638265, 16: 3728.5866052242977},
                84: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2551.83453444399, 12: 3571.9364995576875, 16: 2546.306113674966},
                85: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2309.973051166215, 12: 2252.6222188308234, 16: 3673.160730645033},
                86: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2442.156868407819, 12: 2810.3293303761398, 16: 5135.0250884234},
                87: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2567.2803646091643, 12: 3581.759572660274, 16: 4517.143524987923},
                88: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2567.2803646091643, 12: 3581.759572660274, 16: 4517.143524987923},
                89: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2567.2803646091643, 12: 3581.759572660274, 16: 4517.143524987923},
                90: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2503.5256923579263, 12: 3086.582641072249, 16: 5078.107350043366},
                91: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2567.069060157149, 12: 3600.7380960920814, 16: 3157.6562992830454},
                92: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2567.069060157149, 12: 3600.7380960920814, 16: 3157.6562992830454},
                93: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2427.8418851116426, 12: 3572.7933092445364, 16: 4501.670402224184},
                94: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2258.987173943764, 12: 3830.016220230608, 16: 5103.1348618933125},
                95: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2517.2327321108064, 12: 2333.4943257247646, 16: 4951.735997200973},
                96: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2517.2327321108064, 12: 2333.4943257247646, 16: 4951.735997200973},
                97: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2281.9032130774804, 12: 2331.6442238179307, 16: 5000.741591756361},
                98: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2571.0806293745354, 12: 3600.818406530302, 16: 2840.7471731602363},
                99: {1: 1036.6236464268072, 2: 1509.6459637220719, 4: 1216.6210035627705, 8: 2571.0806293745354, 12: 3600.818406530302, 16: 2840.7471731602363},
            },
        }

    assert application in goodput_functions, f"Application {application} not found in goodput functions"
    assert epoch in goodput_functions[application], f"Epoch {epoch} not found in goodput functions for application {application}"
    assert num_gpus in goodput_functions[application][epoch], f"GPU count {num_gpus} not found in goodput functions for application {application} at epoch {epoch}"
    
    if application == 'cifar10':
        return 50000 / goodput_functions[application][epoch][num_gpus]
    else:
        raise ValueError(f"Application {application} not supported")

def export_all_jobs_epoch_details_csv(jobs, csv_filename):
    """Export epoch details for all jobs to CSV file."""
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['job_name', 'epoch', 'allocation', 'duration', 'wasted', 'theoretical']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for job_name, job_info in jobs.items():
            application = job_name.split('-')[0]
            job_epochs = job_info['epochs']
            
            for epoch in sorted(job_epochs.keys()):
                epoch_info = job_epochs[epoch]
                gpu_list = sorted(epoch_info['gpu_allocations'])
                
                # Format allocation string
                allocation_str = str(gpu_list) if len(gpu_list) > 1 else str(gpu_list[0]) if gpu_list else "0"
                
                duration = epoch_info['duration']
                wasted = epoch_info['wasted_time']
                
                # Calculate theoretical duration based on final GPU allocation
                if len(gpu_list) > 1:
                    theoretical = -1
                else:
                    final_gpu_count = gpu_list[0] if gpu_list else 0
                    try:
                        theoretical = get_theoretical_duration(application, epoch, final_gpu_count) if final_gpu_count > 0 else -1
                    except (AssertionError, ValueError):
                        theoretical = -1
                
                writer.writerow({
                    'job_name': job_name,
                    'epoch': epoch,
                    'allocation': allocation_str,
                    'duration': round(duration, 1),
                    'wasted': round(wasted, 1),
                    'theoretical': round(theoretical, 1) if theoretical != -1 else -1
                })

def print_epoch_details(job_name, jobs, goodput_function=None):
    """Print formatted epoch details table."""
    if job_name not in jobs:
        print(f"Job {job_name} not found in results.")
        return
    
    application = job_name.split('-')[0]
    job_epochs = jobs[job_name]['epochs']
    if not job_epochs:
        print(f"No epoch data for job {job_name}.")
        return
    
    print(f"\n" + "="*70)
    print(f"EPOCH DETAILS FOR {job_name}")
    print("="*70)
    print(f"{'Epoch':<8} {'GPUs Used':<15} {'Duration(s)':<12} {'Wasted(s)':<10} {'Theoretical(s)':<15}")
    print("-" * 70)
    
    for epoch in sorted(job_epochs.keys()):
        epoch_info = job_epochs[epoch]
        gpu_list = sorted(epoch_info['gpu_allocations'])

        
        gpu_str = str(gpu_list) if len(gpu_list) > 1 else str(gpu_list[0]) if gpu_list else "0"
        
        duration = epoch_info['duration']
        wasted = epoch_info['wasted_time']
        
        # Calculate theoretical duration based on final GPU allocation
        if len(gpu_list) > 1:
            theoretical = -1
        else:
            final_gpu_count = gpu_list[0]
            theoretical = get_theoretical_duration(application, epoch, final_gpu_count)
        
        print(f"{epoch:<8} {gpu_str:<15} {duration:<12.1f} {wasted:<10.1f} {theoretical:<15.1f}")

    

def main():
    if len(sys.argv) != 2:
        print("Usage: python manage_monitor_log.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    print(f"Processing log file: {log_file_path}")
    
    jobs, total_gpu_hours, effective_gpu_hours, waiting_for_ready_hours, wasted_capacity_hours, rescaling_hours, theoretical_rescaling_hours, high_waste_examples, last_job_arrival_time = process_log_file(log_file_path)
    
    # Find first job time for metrics calculation
    first_job_time = min(job_info['first_seen'] for job_info in jobs.values()) if jobs else 0
    
    # Print response time and wasted time for all jobs
    print_all_jobs_summary(jobs)
    
    # List of jobs to show detailed epoch information for
    # Modify this list to see details for different jobs
    detailed_jobs = ['cifar10-0', 'cifar10-15']  # Add more job names here as needed
    
    # Print detailed epoch information for specified jobs
    for job_name in detailed_jobs:
        if job_name in jobs:
            print_epoch_details(job_name, jobs)
        else:
            available_jobs = list(jobs.keys())
            print(f"\nJob '{job_name}' not found. Available jobs: {available_jobs}")
            # Optionally show details for first available job if target not found
            if available_jobs and job_name == detailed_jobs[0]:  # Only for first job in list
                print_epoch_details(available_jobs[0], jobs)
    
    # Export all jobs epoch details to CSV
    import os
    base_name = os.path.splitext(log_file_path)[0]  # Remove extension properly
    csv_filename = base_name + '_epoch_details.csv'
    
    # Safety check to never overwrite the original file
    if csv_filename == log_file_path:
        csv_filename = log_file_path + '_epoch_details.csv'
    
    export_all_jobs_epoch_details_csv(jobs, csv_filename)
    print(f"\nEpoch details for all jobs exported to: {csv_filename}")
    
    # Print high waste examples before summary
    print_high_waste_examples(high_waste_examples)
    
    metrics = calculate_metrics(total_gpu_hours, effective_gpu_hours, waiting_for_ready_hours, wasted_capacity_hours, rescaling_hours, theoretical_rescaling_hours, last_job_arrival_time, first_job_time)
    print_summary(metrics)

if __name__ == "__main__":
    main()