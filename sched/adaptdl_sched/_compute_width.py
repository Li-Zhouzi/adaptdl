from adaptdl.goodput import GoodputFunction
from adaptdl.global_profile_state import GlobalProfileState
from ._configs import APPLICATIONS, NUM_GPU_PER_NODE, ARRIVAL_RATE
import numpy as np
import cvxpy as cp
import math
import random

def _load_goodput_function(global_profile_state: GlobalProfileState):
    """
    Load goodput functions and create a goodput dictionary using global profile state data.
    
    Args:
        global_profile_state: The global profile state containing profiles, perf_params, and grad_params
        
    Returns:
        dict: A dictionary where goodput_dict[app][epoch][num_replica] contains the optimized goodput
    """
    goodput_dict = {}
    
    # Validate that all applications have required data
    for application in APPLICATIONS.keys():
        # Assert that application has perf_params
        assert application in global_profile_state.global_perf_params, \
            f"Application {application} missing from global_perf_params"
        # Assert that application has profile data
        assert application in global_profile_state.global_profiles, \
            f"Application {application} missing from global_profiles"
        # Assert that application has grad_params
        assert application in global_profile_state.global_grad_params, \
            f"Application {application} missing from global_grad_params"
        
        # Get application configuration
        app_config = APPLICATIONS[application]
        
        # Validate that all epochs for this application have grad_params
        expected_epochs = range(app_config.max_epochs)
        actual_epochs = set(global_profile_state.global_grad_params[application].keys())
        missing_epochs = set(expected_epochs) - actual_epochs
        assert len(missing_epochs) == 0, \
            f"Application {application} missing grad_params for epochs: {missing_epochs}"
        
        perf_params = global_profile_state.global_perf_params[application]
        profile = global_profile_state.global_profiles[application]
        
        goodput_dict[application] = {}
        
        # For each epoch that has grad_params
        for epoch in global_profile_state.global_grad_params[application].keys():
            grad_params = global_profile_state.global_grad_params[application][epoch]
            
            # Assert that grad_params is not None
            assert grad_params is not None, \
                f"Application {application} epoch {epoch} has None grad_params"
            
            # Create GoodputFunction with the global profile data
            goodput_fn = GoodputFunction(perf_params, grad_params, app_config.init_batch_size)
            
            goodput_dict[application][epoch] = {}
            
            # Calculate optimal goodput for replicas 1-64
            for num_replicas in range(1, 65):
                num_nodes = max(1, (num_replicas + NUM_GPU_PER_NODE - 1) // NUM_GPU_PER_NODE)
                
                
                # Optimize for the best goodput using application-specific config
                optimal_goodput, _, _ = goodput_fn.optimize(
                    num_nodes, num_replicas, 
                    max_batch_size=app_config.max_batch_size,
                    atomic_bsz_range=(app_config.min_local_bsz, app_config.max_local_bsz),
                    accumulation=app_config.gradient_accumulation
                )
                
                goodput_dict[application][epoch][num_replicas] = optimal_goodput
    return goodput_dict

def _get_speedup_and_size(global_profile_state: GlobalProfileState):
    goodput_dict = _load_goodput_function(global_profile_state)
    speedup_dict = dict()
    size_dict = dict()
    for app in goodput_dict.keys():
        speedup_dict[app] = dict()
        size_dict[app] = dict()
        for epoch in goodput_dict[app].keys():
            speedup_dict[app][epoch] = dict()
            for num_replicas, goodput in goodput_dict[app][epoch].items():
                speedup_dict[app][epoch][num_replicas] = goodput / goodput_dict[app][epoch][1]
            size_dict[app][epoch] = global_profile_state.global_size[app][epoch] / goodput_dict[app][epoch][1]
    return speedup_dict, size_dict

def _feasible_speedup(speedup_dict):
    # a speedup dictionary withonly feasible points. 
    # speedup_dict[application][epoch][replicas]=speedup
    feasible_sp_dict = dict()
    for app, list1 in speedup_dict.items():
        feasible_sp_dict[app] = dict()
        for epoch, list2 in list1.items():
            cleaned = dict()
            current = speedup_dict[app][epoch]
            
            cleaned[1] = 1
            idx = 1 # 1-idx has been cleaned
            while idx < len(current.keys()):
                highest_slope = 0
                highest_slope_idx = None
                # find the next point with highest slope
                for j in range(idx + 1, len(current.keys()) + 1):
                    slope = (current[j] - current[idx]) / (j - idx)
                    if slope > highest_slope:
                        highest_slope = slope
                        highest_slope_idx = j
                if not highest_slope_idx:
                    # this means, idx is the largest speedup
                    break
                cleaned[highest_slope_idx] = current[highest_slope_idx]
                idx = highest_slope_idx
                
            feasible_sp_dict[app][epoch] = cleaned
            
    return feasible_sp_dict

def _continuous_inv_speedup(speed_dict, application, epoch, k):
    data_dict = speed_dict[application][epoch]
    x_data = np.array(list(data_dict.keys()), dtype=float)
    y_data = np.array(list(data_dict.values()), dtype=float)

    def _pwl_function(x):
        slopes = (x_data[1:] - x_data[:-1]) / (y_data[1:] - y_data[:-1])
        intercepts = x_data[:-1] - slopes * y_data[:-1]
        expressions = slopes * x + intercepts
        return np.maximum.reduce(expressions)
    return _pwl_function(k)

def _cons_term(speed_dict, application, epoch, z):
    # implement the inverse of the speedup function
    # 1 >= z >= 1 / y_data[-1]
    data_dict = speed_dict[application][epoch]
    if not data_dict:
        raise ValueError(f"Empty data dictionary for application {application} at epoch {epoch}")
        
    x_data = np.array(list(data_dict.keys()), dtype=float)
    y_data = np.array(list(data_dict.values()), dtype=float)
    
    if len(x_data) < 2 or len(y_data) < 2:
        return 1
        # raise ValueError(f"Insufficient data points for application {application} at epoch {epoch}")
        
    sort_indices = np.argsort(x_data)
    x_data = x_data[sort_indices]
    y_data = y_data[sort_indices]
    # print(application, epoch)
    # print(x_data)
    # print(y_data)
    def pwl_function(y):
        slopes = (x_data[1:] - x_data[:-1]) / (y_data[1:] - y_data[:-1])
        intercepts = x_data[:-1] - slopes * y_data[:-1]
        if len(slopes) == 1:
            return slopes[0] + intercepts[0]*y
        expressions = slopes + intercepts * y
        return cp.maximum(*expressions)
    return pwl_function(z)
    # return z * z    

def _compute_width(arrival_dict, mean_size, speedup_dict, b):
    # preparation for optimization
    rho_dict = {name: [a * rate for i, a in enumerate(mean_size[name])]
    for name, rate in arrival_dict.items()}
    index_map = [(name, i) for name in rho_dict for i in range(len(rho_dict[name]))]
    # print("rho dict: ")
    # for idx, (name, i) in enumerate(index_map):
    #         print(name, i, rho_dict[name][i])
    fs_speed_dict = _feasible_speedup(speedup_dict)
    rho_array = np.array([rho_dict[name][i] for name, i in index_map])
    # print("total rho: ", np.sum(rho_array))
    
    # do the optimization problem
    z = cp.Variable(len(index_map))
    # Vectorized objective function
    const_term = cp.hstack([_cons_term(fs_speed_dict, name, i, z[idx])
        for idx, (name, i) in enumerate(index_map)])
    # speedups = cp.hstack([continuous_inv_speedup(fs_speed_dict, name, i, cp.inv_pos(z[idx])) * z[idx]
    #     for idx, (name, i) in enumerate(index_map)])
    objective_terms = cp.multiply(rho_array, z)
    sum_arrival_rates = sum(arrival_dict.values())
    objective = cp.Minimize(cp.sum(objective_terms) / sum_arrival_rates)
    # Vectorized constraints
    constraints = [
    z <= 1,
    cp.sum(cp.multiply(rho_array, const_term)) <= b
    ]
    constraints += [z[idx] >= (1 / list(fs_speed_dict[name][i].values())[-1]) for idx, (name, i) in enumerate(index_map)]
    
    
    problem = cp.Problem(objective, constraints)
    # print('start optimization cvx')
    problem.solve()
    # problem.solve(solver=cp.ECOS)
    # print(problem.status)
    # print("average jct by solver: ", problem.value)
    if problem.status == cp.OPTIMAL:
        optimized_z = z.value
        k_dict = {}
        for idx, (name, i) in enumerate(index_map):
            if name not in k_dict:
                k_dict[name] = dict()
            ki = _continuous_inv_speedup(fs_speed_dict, name, i, 1 / optimized_z[idx])
            # print(name, i, ki)
            # rounding ki to intergers
            def closest_key(D, x):
                # Get the key with the minimum absolute difference from x
                closest = min(D.keys(), key=lambda k: abs(k - x))
                return closest
            k_dict[name][i] = closest_key(fs_speed_dict[name][i], ki)

        return k_dict
    else:
        return None
    

def _compute_things_with_rescale(speedup_dict, mean_size_dict, k_dict, application_rates):
    applications = {job_type: APPLICATIONS[job_type] for job_type in application_rates}
    jct_dict = {}
    total_budget = 0
    rescale_dict = {}
    for job_name in k_dict.keys():
        rescale = applications[job_name.split("-")[0]].rescale_time
        jct_dict[job_name] = []
        rescale_dict[job_name] = []
        for epoch in range(len(k_dict[job_name])):
            # print(job_name, epoch)
            k = k_dict[job_name][epoch]
            speedup = speedup_dict[job_name][epoch][k]
            running_time = mean_size_dict[job_name][epoch] / speedup
            if epoch == 0 or k != k_dict[job_name][epoch-1]:
                running_time += rescale
                rescale_dict[job_name].append(rescale)
            else: 
                rescale_dict[job_name].append(0)
            jct_dict[job_name].append(running_time)
            total_budget += running_time * k * application_rates[job_name]
    s = 0
    for k,l in jct_dict.items():
        s += sum(l)*application_rates[k]
    s /= sum(application_rates.values())

    rescale_time = 0
    for k,l in rescale_dict.items():
        rescale_time += sum(l) * application_rates[k]
    rescale_time /= sum(application_rates.values())
    # print("average theory jct ", s)
    return s, total_budget, rescale_time

def _get_glue_list(arrival_dict, size_dict):
    num_epochs_dict = dict()
    for app_name in arrival_dict.keys():
        num_epochs_dict[app_name] = len(size_dict[app_name])
    print("num_epochs_dict: ", num_epochs_dict)
    # Generate 30 random dictionaries
    glue_list = []
    for _ in range(30):
        glue_dict = {}
        for app_name, k in num_epochs_dict.items():
            max_glue = math.ceil(k / 10)
            glue_dict[app_name] = random.randint(1, max_glue)
        if glue_dict not in glue_list:
            glue_list.append(glue_dict)

    return glue_list

def _get_width_with_rescale(speedup_dict, size_dict, application_rates, b):
    glue_list = _get_glue_list(application_rates, size_dict)
    
    min_jct_over_glue = None
    min_glue_ind = None
    final_k_dict = None

    print("--------------------OPTIMIZING OVER GLUE-----------------------------------")
    for index in range(len(glue_list)):
        # construct the new speedup dictionary and size data
        size_glue = dict()
        speed_glue = dict()
        for name in application_rates.keys():
            glue = glue_list[index][name]

            size_glue[name] = []
            if name not in speed_glue:
                speed_glue[name] = dict()

            for epoch, size in enumerate(size_dict[name]):
                if epoch % glue == 0:
                    size_glue[name].append(0)
                size_glue[name][int(epoch / glue)] += size 

            for epoch, d1 in speedup_dict[name].items():
                if epoch % glue == 0:
                    speed_glue[name][int(epoch / glue)] = dict()
                for k, sp in d1.items():
                    if epoch % glue == 0:
                        speed_glue[name][int(epoch / glue)][k] = 0
                    speed_glue[name][int(epoch / glue)][k] += size_dict[name][epoch] / sp

            
            for epoch, d1 in speed_glue[name].items():
                for k, sp in d1.items():
                    speed_glue[name][epoch][k] = size_glue[name][epoch] / sp
        print("--------FOR GLUE=",glue_list[index],"---------------------")
        k_glue = _compute_width_iter(application_rates, size_glue, speed_glue, b) # compute the glued optimization. 
        if k_glue is None:
            continue
        # regenerate the full k dictionary
        k_glue_dict = dict()

        for name, d1 in speedup_dict.items():
            glue = glue_list[index][name]
            if name not in k_glue_dict:
                k_glue_dict[name] = dict()
            for epoch, d2 in d1.items():
                k_glue_dict[name][epoch] = k_glue[name][int(epoch / glue)]  
        # print("glue param: ", glue, "k_glue: ", k_glue_dict)
        avg_jct, total_b, rt = _compute_things_with_rescale(speedup_dict, size_dict, k_glue_dict, application_rates)
        print("jct: ", avg_jct, "rescaling time: ", rt, "budget: ", total_b)
        # print("here2", len(k_glue_dict["deepspeech2"]))
        if not min_jct_over_glue or avg_jct < min_jct_over_glue:
            min_glue_ind = index
            min_jct_over_glue = avg_jct
            final_k_dict = k_glue_dict
    if final_k_dict is None:
        print("No valid glue found")
        return None, None, None

    print("--------------------OPTIMAL GLUE: ", glue_list[min_glue_ind], "----------------------------")


    return final_k_dict
                
def _compute_width_iter(application_rates, size_data, speedup_dict, b):
    total_budget = None
    running_b = b
    k_dict = None
    while ((total_budget is None) or (total_budget > b)) and running_b > 0:
        # print("running_b: ", running_b, flush=True)
        k_dict = _compute_width(
            application_rates, size_data, speedup_dict, running_b
        )
        if k_dict is None:
            running_b -= 1
            continue
        s, total_budget, rescale_time = _compute_things_with_rescale(speedup_dict, size_data, k_dict, application_rates)
        running_b -= 1
        # print(f"Total budget: {total_budget}, b: {b}")
    if total_budget is None or total_budget > b:
        return None 
    print("Total budget in compute_width_iter", total_budget) 
    return k_dict


def get_width(global_profile_state: GlobalProfileState, b):
    # speedup_dict, size_dict = _get_speedup_and_size(global_profile_state)
    # return _get_width_with_rescale(speedup_dict, size_dict, ARRIVAL_RATE, b)
    width = dict()
    for app in ARRIVAL_RATE.keys():
        width[app] = dict()
        for epoch in APPLICATIONS[app].max_epochs:
            width[app][epoch] = 4
            if epoch > 15:
                width[app][epoch] = 8
            if epoch > 30:
                width[app][epoch] = 12
    return width

