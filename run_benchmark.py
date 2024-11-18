#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################
"""
Created on Wed Jan 31 13:28:06 2024

@author: R. Pivovar
"""

import argparse
import boto3
import logging
import pandas as pd
from tqdm import tqdm
from CDKBenchEnv.benchmark_utils import get_instance_types_in_family, separate_ec2_types
from datetime import datetime, timedelta
import awswrangler as wr
from src.utils import (
    get_json,
    get_cloudformation_metadata,
    process_monitor_results,
    submit_jobs,
    aws_batch_wait,
    get_instance_specs
)
from skopt import Optimizer
from skopt.space import Real, Integer
import pickle
import copy
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

#TODO: figure out a way to not do this
def load_valid_instance_list():
    with open("valid_list.txt", "r") as f:
        valid_list = f.read().replace("\n", "").replace(" ", "").split(",")
    return valid_list

def get_job_resources(region, metadata):
    job_definition_name = metadata["JDec2benchmark"].split("/")[-1].split(":")[0]
    if "JDec2benchmarkgraviton" in metadata.keys():
        job_definition_name_graviton = (
            metadata["JDec2benchmarkgraviton"].split("/")[-1].split(":")[0]
        )
    else:
        job_definition_name_graviton = None
    return job_definition_name, job_definition_name_graviton


def get_ec2_types(json_config, valid_list):
    ec2_types = []
    for family_types in json_config["ec2_types"]:
        ec2_types.extend(get_instance_types_in_family(family_types))
    ec2_types = [typ for typ in ec2_types if typ in valid_list or typ.split(".")[0] in valid_list]
    ec2_types = [typ for typ in ec2_types if "metal" not in typ]
    graviton_types, non_graviton_types = separate_ec2_types(ec2_types)
    return graviton_types, non_graviton_types


#TODO: add support for graviton in optimization tuning
def process_optimization_iteration(
    batch_client,
    s3_bucket_name,
    json_config,
    non_graviton_types,
    queue_name,
    job_definition_name,
    replicates,
    job_command,
    optimizer,
    optimization_parallel_samples,
    optimization_metric,
):
    x = optimizer.ask(n_points=optimization_parallel_samples)

    ids = []
    xids = {}
    for v in x:
        #v = x[0]

        hashStr = str(hash(np.random.randint(0,int(1e9))))
        ids.append(hashStr)
        xids[hashStr] = v

        args = list(map(str,v))
        job_command_with_args = copy.deepcopy(job_command)
        index = job_command.index('ARGS')
        job_command_with_args[index:index+1] = args

        index = job_command_with_args.index('--cmd')
        job_command_with_args[index:index] = ["-id", hashStr]

        submit_jobs(
            batch_client,
            non_graviton_types,
            queue_name,
            job_definition_name,
            replicates,
            job_command_with_args,
        )

    aws_batch_wait(queue_name, json_config["region_name"])

    file_names = wr.s3.list_objects(f"s3://{s3_bucket_name}")
    current_time = datetime.now()
    within_last_hrs = json_config["review_logs_past_hrs"]

    filtered_file_names = filter_file_names(file_names, current_time, within_last_hrs)

    #TODO: this will be optional if user wants to optimize both parameter and ec2 instance
    # Only want the files that were just run for the BO algo
    runids = [x.split('_')[-3] for x in filtered_file_names]
    filtered_file_names = [filtered_file_names[i] for i, m in enumerate(runids) if m in list(xids.keys())]

    missing_filtered_file_names = [m for m in list(xids.keys()) if m not in runids]


    logger.info(f"Files less than {within_last_hrs} hour(s) old:")
    for file_name in filtered_file_names:
        logger.info(file_name)
    print("xid log",xids)
    # if len(filtered_file_names) == 0:
    #     raise ValueError("ERROR: all job executions failed to return a result.")

    resultids = [x.split('_')[-3] for x in filtered_file_names]
    data = process_results(filtered_file_names, json_config['region_name'])
    data['resultids'] = resultids

    # In case user runscript recorded a non-numeric value, remove these and treat as failed
    data, non_numeric_rows = correct_problem_rows(data)
    missing_filtered_file_names.extend(non_numeric_rows)

    if len(optimizer.yi) > 0:
        current_max_metric = max(optimizer.yi)
        if len(data) > 0:
            current_max_metric = max(data[optimization_metric].max(), current_max_metric)
    else:
        current_max_metric = data[optimization_metric].max()

    # Add missing/failed jobs
    new_rows = []
    columns = data.columns  # Get the column names from the DataFrame
    for missing in missing_filtered_file_names:
        new_row = [current_max_metric] * len(columns)  # Create a list with the desired value for all columns
        new_row[-1] = missing  # Replace the value at the last column with the missing value
        new_row = dict(zip(columns, new_row))  # Create a dictionary mapping column names to values
        new_rows.append(pd.DataFrame([new_row], columns=columns))  # Create a new DataFrame row
    if len(data) > 0:
        data = pd.concat([data] + new_rows, ignore_index=True)  # Concatenate the new rows to the original DataFrame
    else:
        data = pd.concat( new_rows, ignore_index=True)
    metric = data[[optimization_metric,'resultids']]

    sorted_x = [xids[m] for m in metric['resultids']]
    sorted_y = metric[optimization_metric].to_list()
    sorted_y = list(map(float,sorted_y))

    optimizer.tell(sorted_x, sorted_y)

    with open("optimizer_state.pkl", "wb") as f:
        pickle.dump(optimizer, f)


def filter_file_names(file_names, current_time, within_last_hrs):
    filtered_file_names = []
    for file_name in file_names:
        try:
            timestamp = datetime.strptime(file_name.split("_")[-1].split(".")[0], "%H%M%S")
            file_date = datetime.strptime(file_name.split("_")[-2], "%Y%m%d")
            file_datetime = datetime.combine(file_date, timestamp.time())

            if current_time - file_datetime < timedelta(hours=within_last_hrs):
                filtered_file_names.append(file_name)
        except ValueError:
            logger.warning(f"Skipping file {file_name} due to unexpected filename format.")
    return filtered_file_names


def sort_by_timestamp(lst):
    # Define a lambda function to extract the timestamp from each string
    get_timestamp = lambda x: datetime.strptime("_".join(x.split('_')[-2:]).split(".")[0], '%Y%m%d_%H%M%S')

    # Sort the list using the get_timestamp function and reverse=True to get the latest first
    sorted_lst = sorted(lst, key=get_timestamp, reverse=True)

    return sorted_lst


def is_number(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def correct_problem_rows(data):
    non_numeric_rows = []
    for i, row in data.iterrows():
        metric_value = row['custom_metric']
        if isinstance(metric_value, str) and not is_number(metric_value):
            non_numeric_rows.append((row['resultids'], i))

    if non_numeric_rows:
        print("Rows with non-numeric custom_metric values:")
        for resultid, index in non_numeric_rows:
            print(f"resultid: {resultid}")

        # Remove rows with non-numeric custom_metric values
        data = data.drop(data.index[list(row[1] for row in non_numeric_rows)])
        print("\nDataFrame after removing non-numeric rows:")
        print(data)
        non_numeric_ids , _ = zip(*non_numeric_rows)
    else:
        print("All custom_metric values are numeric.")
        non_numeric_ids = []

    return data, non_numeric_ids


def process_results(filtered_file_names, region):
    data = []
    for file_name in tqdm(filtered_file_names, desc="Processing results"):
        try:
            df = wr.s3.read_csv(path=file_name)
            instance_type = file_name.split("/")[-1].split("_")[0]
            df = process_monitor_results(df, instance_type, region)
            Avg_cpu = df["CPU_Usage"].mean()
            if Avg_cpu == 0.0:
                continue
            peak_ram = df["RAM_Usage"].max()
            cost = df["expected_cost"].max()
            runtime = df["runtime"].max()
            if 'custom_metric' in df.columns:
                custom_metric = df["custom_metric"].max()
            else:
                custom_metric = 0.0
            Disk_IO_Read = df["Disk_IO_Read"].mean()
            Disk_IO_Write = df["Disk_IO_Write"].mean()
            Network_Sent = df["Network_Sent"].mean()
            Network_Received = df["Network_Received"].mean()
            data.append(
                [
                    instance_type,
                    Avg_cpu,
                    peak_ram,
                    cost,
                    runtime,
                    Disk_IO_Read,
                    Disk_IO_Write,
                    Network_Sent,
                    Network_Received,
                    custom_metric
                ]
            )
        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}")

    data = pd.DataFrame(
        data,
        columns=[
            "ec2_type",
            "Avg_cpu",
            "peak_RAM",
            "cost",
            "runtime",
            "Disk_IO_Read",
            "Disk_IO_Write",
            "Network_Sent",
            "Network_Received",
            "custom_metric"
        ],
    )
    return data


def run_optimization(json_config):
    valid_list = load_valid_instance_list()
    region = json_config["region_name"]
    replicates = json_config["replications"]
    s3_bucket_name = json_config["s3_bucket_name"]
    job_command = json_config["run_cmd"].split()

    grid_search = False
    if "optimization_method" in json_config.keys():
        if json_config["optimization_method"].lower() == 'grid':
            grid_search = True

    index = job_command.index('--cmd')
    job_command[index:index] = ["-s3", s3_bucket_name]

    metadata = get_cloudformation_metadata("CdkBenchEnvStack", region=region)
    job_definition_name, job_definition_name_graviton = get_job_resources(region, metadata)

    batch_client = boto3.client("batch", region_name=region)

    if len(json_config["ec2_types"]) == 1 and "." in json_config["ec2_types"][0]:
        non_graviton_types = json_config["ec2_types"]
    else:
        graviton_types, non_graviton_types = get_ec2_types(json_config, valid_list)

        #Find the largest
        specs = [get_instance_specs(instance_type) for instance_type in non_graviton_types]
        def extract_cpu_count(row):
            cpu_count_str = row['CPU'].split()[0]
            return int(cpu_count_str)
        max_cpu_index = max(range(len(specs)), key=lambda i: extract_cpu_count(specs[i]))
        non_graviton_types = [non_graviton_types[max_cpu_index]]

    total_iterations = json_config["optimization_iterations"]
    optimization_parallel_samples = json_config["optimization_parallel_samples"]
    optimization_metric = json_config["optimization_metric"]

    instance_queue_map = get_queue_instance_type_map(batch_client, metadata, non_graviton_types)
    queue_name = list(instance_queue_map.keys())[0]

    optimization_dimensions = []
    dim_types = [
        value
        for key, value in json_config.items()
        if "optimization_arg" in key and "type" in key
    ]
    dim_range = [
        value
        for key, value in json_config.items()
        if "optimization_arg" in key and "type" not in key
    ]

    for i, dim_type in enumerate(dim_types):
        value = dim_range[i]
        if dim_type.lower() == "categorical":
            raise ValueError("ERROR: categorical is not setup at the moment.")
        elif dim_type.lower() == "integer":
            optimization_dimensions.append(Integer(value[0], value[1]))
        elif dim_type.lower() == "log-uniform":
            optimization_dimensions.append(Real(value[0], value[1], "log-uniform"))
        else:
            optimization_dimensions.append(Real(value[0], value[1]))

    if grid_search:
        optimizer = Optimizer(
            dimensions=optimization_dimensions, random_state=42, base_estimator="gp",
            n_initial_points=optimization_parallel_samples,
            initial_point_generator='grid'
        )
        total_iterations=1
    else:
        optimizer = Optimizer(
            dimensions=optimization_dimensions, random_state=42, base_estimator="gp"
        )


    for i in range(total_iterations):
        process_optimization_iteration(
            batch_client,
            s3_bucket_name,
            json_config,
            non_graviton_types,
            queue_name,
            job_definition_name,
            replicates,
            job_command,
            optimizer,
            optimization_parallel_samples,
            optimization_metric,
        )

    print(min(optimizer.yi))  # print the best objective found


def get_queue_instance_type_map(batch_client, metadata, instance_types):
    """
    Create a mapping of job queues to the supported instance types from the provided list.

    This function filters compute environments based on the provided metadata,
    checks for instance type compatibility, and maps job queues to their supported instance types.

    Args:
    batch_client (boto3.client): Boto3 AWS Batch client
    metadata (dict): A dictionary containing metadata about the deployment
    instance_types (list): List of EC2 instance types to check for compatibility

    Returns:
    dict: A dictionary with job queue names as keys and lists of supported EC2 instance types as values.
           Only queues with supported instance types are included in the result.
    """
    queue_instance_map = {}

    # Get all compute environments
    compute_environments = batch_client.describe_compute_environments()['computeEnvironments']
    keys = [x.split('/')[-1] for x in metadata.values()]

    #Ensure only looking at ce deployed for this code
    lst = []
    for ce in compute_environments:
        ce_name = ce['computeEnvironmentName']
        if ce_name in keys:
            lst.append(ce)
    compute_environments = lst

    # Create a mapping of compute environment names to their instance types
    ce_instance_map = {}
    for ce in compute_environments:
        ce_name = ce['computeEnvironmentName']
        ce_instance_types = set(ce['computeResources'].get('instanceTypes', []))

        shared_instances = []
        for instance in instance_types:
            instance_prefix = instance.split('.')[0]  # Get the prefix before the dot
            for ce_instance in ce_instance_types:
                #if instance_prefix in ce_instance or ce_instance in instance_prefix:
                if instance_prefix == ce_instance:
                    shared_instances.append(instance)
                    break  # Move to the next instance once a match is found

        ce_instance_map[ce_name] = shared_instances #ce_instance_types.intersection(instance_types)

    # Get all job queues
    job_queues = batch_client.describe_job_queues()['jobQueues']

    # For each job queue, find its compute environments and associated instance types
    for queue in job_queues:
        queue_name = queue['jobQueueName']
        queue_instance_types = set()
        for env in queue['computeEnvironmentOrder']:
            ce_name = env['computeEnvironment']
            ce_name =ce_name.split('/')[-1]
            if ce_name in ce_instance_map.keys():
                queue_instance_types.update(ce_instance_map[ce_name])

        if queue_instance_types:
            queue_instance_map[queue_name] = list(queue_instance_types)

    return queue_instance_map


def run_benchmark(json_config):
    valid_list = load_valid_instance_list()
    region = json_config["region_name"]
    replicates = json_config["replications"]
    s3_bucket_name = json_config["s3_bucket_name"]
    job_command = json_config["run_cmd"].split()
    index = job_command.index('--cmd')
    job_command[index:index] = ["-s3", s3_bucket_name]
    exclude_lst = json_config.get("exclude_ec2_types", [])

    metadata = get_cloudformation_metadata("CdkBenchEnvStack", region=region)
    job_definition_name, job_definition_name_graviton = get_job_resources(region, metadata)

    batch_client = boto3.client("batch", region_name=region)

    graviton_types, non_graviton_types = get_ec2_types(json_config, valid_list)

    #Remove any per user request
    graviton_types = [x for x in graviton_types if not any(exclude in x for exclude in exclude_lst)]
    non_graviton_types = [x for x in non_graviton_types if not any(exclude in x for exclude in exclude_lst)]

    grav_instance_queue_map = get_queue_instance_type_map(batch_client, metadata, graviton_types)
    instance_queue_map = get_queue_instance_type_map(batch_client, metadata, non_graviton_types)

    for queue_name in instance_queue_map.keys():
        job_ids = submit_jobs(
            batch_client,
            instance_queue_map[queue_name],
            queue_name,
            job_definition_name,
            replicates,
            job_command,
        )
    if job_definition_name_graviton is not None:
        for queue_name_graviton in grav_instance_queue_map.keys():
            job_ids += submit_jobs(
                batch_client,
                grav_instance_queue_map[queue_name_graviton],
                queue_name_graviton,
                job_definition_name_graviton,
                replicates,
                job_command,
            )

    for queue_name in instance_queue_map.keys():
        aws_batch_wait(queue_name, region)
    if job_definition_name_graviton is not None:
        for queue_name_graviton in grav_instance_queue_map.keys():
            aws_batch_wait(queue_name_graviton, region)

    file_names = wr.s3.list_objects(f"s3://{s3_bucket_name}")
    current_time = datetime.now()
    within_last_hrs = json_config["review_logs_past_hrs"]

    filtered_file_names = filter_file_names(file_names, current_time, within_last_hrs)

    logger.info(f"Files less than {within_last_hrs} hour(s) old:")
    for file_name in filtered_file_names:
        logger.info(file_name)

    data = process_results(filtered_file_names, region)
    data.to_csv("ec2data.csv", index=False)

def kill_all_jobs(json_config):
    region = json_config["region_name"]
    metadata = get_cloudformation_metadata("CdkBenchEnvStack", region=region)

    # Filter queue names from metadata
    queues = [value for value in metadata.values() if 'job-queue' in value]

    # Initialize Batch client
    batch_client = boto3.client("batch", region_name=region)

    # Job states to terminate
    job_states = ['RUNNING', 'RUNNABLE', 'STARTING']

    for queue in queues:
        for state in job_states:
            # List all jobs in the queue for the current state
            response = batch_client.list_jobs(jobQueue=queue, jobStatus=state)
            job_ids = [job['jobId'] for job in response['jobSummaryList']]

            # Terminate each job
            for job_id in job_ids:
                try:
                    batch_client.terminate_job(
                        jobId=job_id,
                        reason=f'Terminated {state} job by killjobs function'
                    )
                    print(f"Terminated {state} job {job_id} in queue {queue}")
                except Exception as e:
                    print(f"Error terminating {state} job {job_id}: {str(e)}")

        print(f"Finished processing queue: {queue}")

def main(config_path, killjobs):

    #config_path = "./examples/prime_numbers/benchmark_config.json"
    #config_path = "./examples/fem_calculation/benchmark_config.json"
    #config_path = "./examples/fem_calculation_optimization/benchmark_config.json"
    #config_path = "./examples/inferentia_tune/benchmark_config.json"
    #config_path = "./examples/gpu_training_optimization/benchmark_config.json"

    json_config = get_json(config_path)

    if killjobs:
        kill_all_jobs(json_config)
    else:
        if "optimization_iterations" in json_config:
            run_optimization(json_config)
        else:
            run_benchmark(json_config)

#%% main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", dest="config_path", type=str, help="json input file")
    parser.add_argument("-k", "--killjobs", action="store_true", help="Terminate all running jobs.")
    args = parser.parse_args()
    main(args.config_path, args.killjobs)