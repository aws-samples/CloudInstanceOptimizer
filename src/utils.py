#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################


import requests
import pandas as pd
import boto3
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from botocore.exceptions import ClientError
from tqdm import tqdm
import copy
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import os
import tempfile
import subprocess
import uuid



@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def submit_single_job(
    batch_client,
    job_queue_name: str,
    job_definition_name: str,
    job_command: List[str],
    ec2_type: Optional[str] = None,
    rep: int = 0,
    nnodes: int = 1,
    cpu: int = 1,
    ram: int = 1024,
    gpus: int = 0
) -> str:
    legalname = ec2_type.replace('.', '-') if ec2_type else 'default'

    # Include GPU number in the job name if GPUs are requested
    gpu_suffix = f'-{gpus}gpu' if gpus > 0 else ''
    job_name = f'ec2-benchmark-{legalname}{gpu_suffix}-{rep}'

    node_property_overrides = {
        'targetNodes': '0:',
        'containerOverrides': {
            'command': job_command,
            'vcpus': cpu,
            'memory': int(ram * 1e3),
            'environment': [
                {
                    'name': 'EXPECTED_NODES',
                    'value': str(nnodes)
                },
            ],
        },
    }

    # Add GPU requirements if specified
    if gpus > 0:
        node_property_overrides['containerOverrides']['resourceRequirements'] = [
            {
                'type': 'GPU',
                'value': str(gpus)
            }
        ]

    # Only add instanceType if ec2_type is provided
    if ec2_type:
        node_property_overrides['containerOverrides']['instanceType'] = ec2_type

    response = batch_client.submit_job(
        #jobName=f'ec2-benchmark-{legalname}-{rep}',
        jobName=job_name,
        jobDefinition=job_definition_name,
        jobQueue=job_queue_name,
        nodeOverrides={
            'numNodes': nnodes,
            'nodePropertyOverrides': [node_property_overrides]
        }
    )
    return response['jobId']


def submit_jobs(
    batch_client,
    ec2_types: List[str],
    job_queue_name: str,
    job_definition_name: str,
    replicates: int,
    job_command: List[str],
    nnodes: int = 1,
    alltypes: bool = True,
    gpus: int = 0
) -> List[str]:
    """
    Submit jobs to AWS Batch for the specified EC2 types with exponential backoff.

    Args:
        batch_client (botocore.client.Batch): AWS Batch client instance.
        ec2_types (List[str]): List of EC2 instance types to use.
        job_queue_name (str): Name of the AWS Batch job queue.
        job_definition_name (str): Name of the AWS Batch job definition.
        replicates (int): Number of replicate jobs to submit for each EC2 type.
        job_command (List[str]): Command to run for each job.
        nnodes (int): Number of nodes for each job. Defaults to 1.
        alltypes (bool): If True, use all EC2 types; if False, use only the first type without specifying EC2 type. Defaults to True.
        gpus (int): Number of GPUs to request for each job. Defaults to 0.

    Returns:
        List[str]: List of job IDs for the submitted jobs.
    """
    job_ids = []

    if alltypes:
        for ec2_type in tqdm(ec2_types, desc="Submitting jobs"):
            specs = get_instance_specs(ec2_type)
            cpu = int(specs['CPU'].split()[0])
            ram = int(specs['RAM'].split()[0]) * 0.95  # Reserve some RAM for OS
            available_gpus = int(specs['GPU'])

            if gpus > available_gpus:
               print(f"Cannot submit jobs: Requested {gpus} GPUs, but only {available_gpus} available on {ec2_type}")
               return None

            for rep in range(replicates):
                try:
                    job_id = submit_single_job(
                        batch_client,
                        job_queue_name,
                        job_definition_name,
                        job_command,
                        ec2_type,
                        rep,
                        nnodes,
                        cpu,
                        ram,
                        gpus
                    )
                    job_ids.append(job_id)
                except Exception as e:
                    print(f"Failed to submit job for {ec2_type}, replicate {rep}: {str(e)}")
    else:
        ec2_type = ec2_types[0]
        specs = get_instance_specs(ec2_type)
        cpu = int(specs['CPU'].split()[0])
        ram = int(specs['RAM'].split()[0]) * 0.95  # Reserve some RAM for OS
        available_gpus = int(specs['GPU'])

        if gpus > available_gpus:
           print(f"Cannot submit jobs: Requested {gpus} GPUs, but only {available_gpus} available on {ec2_type}")
           return None

        for rep in tqdm(range(replicates), desc="Submitting jobs"):
            try:
                job_id = submit_single_job(
                    batch_client,
                    job_queue_name,
                    job_definition_name,
                    job_command,
                    None,  # ec2_type is None
                    rep,
                    nnodes,
                    cpu,
                    ram,
                    gpus
                )
                job_ids.append(job_id)
            except Exception as e:
                print(f"Failed to submit job for replicate {rep}: {str(e)}")

    return job_ids


def get_running_jobs(queue: str, region: str) -> List[Tuple[str, str]]:
    """
    Retrieves a list of running jobs for a given queue and region.

    Args:
        queue (str): The name of the AWS Batch job queue.
        region (str): The AWS region where the job queue is located.

    Returns:
        List[Tuple[str, str]]: A list of tuples, where each tuple contains the job name and job ID.
    """
    batch_client = boto3.client('batch', region_name=region)
    paginator = batch_client.get_paginator('list_jobs')
    job_statuses = ['SUBMITTED', 'PENDING', 'RUNNABLE', 'STARTING', 'RUNNING']
    jobs_list = []

    for status in job_statuses:
        try:
            for response in paginator.paginate(jobQueue=queue, jobStatus=status):
                if 'jobSummaryList' in response.keys():
                    for job in response['jobSummaryList']:
                        jobs_list.append((job['jobName'], job['jobId']))
        except ClientError as e:
            print(f"Error occurred while retrieving jobs: {e}")

    return jobs_list



@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60)
)
def describe_jobs_with_retry(batch_client, jobs):
    return batch_client.describe_jobs(jobs=jobs)['jobs']


def aws_batch_wait(queue: str, region: str) -> None:
    """
    Waits for all jobs in a given queue and region to complete (either succeed or fail).
    Uses exponential backoff when encountering errors.

    Args:
        queue (str): The name of the AWS Batch job queue.
        region (str): The AWS region where the job queue is located.
    """
    batch_client = boto3.client('batch', region_name=region)
    all_jobs = get_running_jobs(queue, region)
    _, all_job_ids = zip(*all_jobs) if all_jobs else ([], [])

    # Remove any duplicates in the list
    all_job_ids = list(set(all_job_ids))
    running_jobs = copy.deepcopy(all_job_ids)
    count = 0

    while running_jobs:
        count += 1

        try:
            job_statuses = describe_jobs_with_retry(batch_client, running_jobs[:100])
            done_job_ids = [job['jobId'] for job in job_statuses if job['status'] in ['FAILED', 'SUCCEEDED']]

            for done_job_id in done_job_ids:
                if done_job_id in running_jobs:
                    running_jobs.remove(done_job_id)

            fraction_done = (1 - len(running_jobs) / len(all_job_ids)) * 100
            if count % 5 == 0:
                print(f"{fraction_done:.2f}% of jobs have completed.")
        except Exception as e:
            print(f"Error occurred while waiting for jobs: {e}")

        time.sleep(1)


def createslurmdir():
    directory = "slurmoutputs"
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"Directory '{directory}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory}': {e}")


def minutes_to_sbatch_time(minutes):
    days, remainder = divmod(minutes, 24 * 60)
    hours, remainder = divmod(remainder, 60)

    if days > 0:
        return f"{days}-{hours:02d}:{remainder:02d}:00"
    else:
        return f"{hours:02d}:{remainder:02d}:00"


def submit_slurm_job(num_nodes, ecr_repo, image_name, image_version, script_arg, s3_bucket, region, job_timeout):

    createslurmdir()

    # Generate a unique identifier for this job
    job_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of a UUID
    job_name = f"optimization_job_{job_uuid}"

    # Create a temporary file with a consistent naming scheme
    temp_file_path = os.path.join(tempfile.gettempdir(), f"{job_name}.sh")

    sbatch_time = minutes_to_sbatch_time(job_timeout)

    # Create a temporary file
    with open(temp_file_path, 'w') as temp_file:
        script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --output=./slurmoutputs/{job_name}_%j.out
#SBATCH --error=./slurmoutputs/{job_name}_%j.err
#SBATCH --exclusive
#SBATCH --time={sbatch_time}


AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS=$(hostname -I | awk '{{print $1}}')
export AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS

srun --ntasks="${{SLURM_NTASKS}}" --ntasks-per-node=1 aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {ecr_repo}

srun --ntasks="${{SLURM_NTASKS}}" --ntasks-per-node=1 docker pull {ecr_repo}/{image_name}:{image_version}

srun --ntasks="${{SLURM_NTASKS}}" --ntasks-per-node=1 \\
    docker run \\
    --shm-size=20000000m \\
    --network=host \\
    -e EXPECTED_NODES="${{SLURM_NTASKS}}" \\
    -e SLURM_LOCALID="${{SLURM_LOCALID}}" \\
    -e AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS="${{AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS}}" \\
    {ecr_repo}/{image_name}:{image_version} \\
    {script_arg}
"""
        temp_file.write(script_content)
        temp_file_path = temp_file.name

    try:
        # Submit the job to Slurm
        submit_command = f"sbatch {temp_file_path}"
        result = subprocess.run(submit_command, shell=True, check=True, capture_output=True, text=True)

        # Extract job ID from the output
        job_id = result.stdout.strip().split()[-1]
        print(f"Job submitted with ID: {job_id}")

        # Wait for the job to complete
        while True:
            status_command = f"squeue -j {job_id}"
            status_result = subprocess.run(status_command, shell=True, capture_output=True, text=True)

            if job_id not in status_result.stdout:
                print("Job completed")
                break

            time.sleep(10)

    finally:
        # Clean up the temporary file
        os.unlink(temp_file_path)



def get_json(filename: str) -> Dict:
    """
    Read a JSON file and return its contents as a dictionary.

    Parameters
    ----------
    filename : str
        The path to the JSON file.

    Returns
    -------
    Dict
        The contents of the JSON file as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file contents are not valid JSON.
    """
    try:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"File '{filename}' does not exist.")

        with path.open('r', encoding='utf-8') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON file '{filename}': {e}") from e

    return config




def get_cloudformation_metadata(stack_name: str, region: Optional[str] = None) -> Dict[str, str]:
    """
    Parse and reduce raw CloudFormation metadata down to a small Python dictionary.

    Parameters
    ----------
    stack_name : str
        Name of the CloudFormation stack.
    region : str, optional
        AWS region of the account. If not provided, it will be inferred from the environment.

    Returns
    -------
    Dict[str, str]
        A dictionary containing metadata for the following resource types:
            - SiteWise
            - Batch
            - S3
            - DynamoDB
            - RDS
            - IAM
            - SQS
            - Timestream

    Raises
    ------
    ValueError
        If the region is not provided and cannot be inferred from the environment.
    """
    try:
        if region:
            cf_client = boto3.client('cloudformation', region_name=region)
        else:
            cf_client = boto3.client('cloudformation')
    except Exception as e:
        raise ValueError(f"Failed to create CloudFormation client: {e}")

    try:
        response = cf_client.list_stack_resources(StackName=stack_name)
    except Exception as e:
        raise ValueError(f"Failed to list stack resources for stack {stack_name}: {e}")

    metadata: Dict[str, str] = {}
    for resource in response['StackResourceSummaries']:
        resource_type = resource['ResourceType'].lower()
        if any(keyword in resource_type for keyword in ['sitewide', 'batch', 's3', 'dynamodb', 'rds', 'iam', 'sqs', 'timestream']):
            metadata[resource['LogicalResourceId']] = resource['PhysicalResourceId']

    return {key: value for key, value in metadata.items() if 'Default' not in key}



def datetime_to_iso(dt):
    # Convert to UTC
    dt_utc = dt.astimezone(tz=None)

    # Format as ISO 8601 UTC
    iso_8601_utc = dt_utc.isoformat() + 'Z'
    return iso_8601_utc


def get_region_name(region):
    '''
    Copy and pasted from
    https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/Concepts.RegionsAndAvailabilityZones.html

    Can write some webscraping if we care to make this more robust
    '''
    region_dict = {
        "us-east-2": "US East (Ohio)",
        "us-east-1": "US East (N. Virginia)",
        "us-west-1": "US West (N. California)",
        "us-west-2": "US West (Oregon)",
        "af-south-1": "Africa (Cape Town)",
        "ap-east-1": "Asia Pacific (Hong Kong)",
        "ap-south-2": "Asia Pacific (Hyderabad)",
        "ap-southeast-3": "Asia Pacific (Jakarta)",
        "ap-southeast-4": "Asia Pacific (Melbourne)",
        "ap-south-1": "Asia Pacific (Mumbai)",
        "ap-northeast-3": "Asia Pacific (Osaka)",
        "ap-northeast-2": "Asia Pacific (Seoul)",
        "ap-southeast-1": "Asia Pacific (Singapore)",
        "ap-southeast-2": "Asia Pacific (Sydney)",
        "ap-northeast-1": "Asia Pacific (Tokyo)",
        "ca-central-1": "Canada (Central)",
        "ca-west-1": "Canada West (Calgary)",
        "eu-central-1": "Europe (Frankfurt)",
        "eu-west-1": "Europe (Ireland)",
        "eu-west-2": "Europe (London)",
        "eu-south-1": "Europe (Milan)",
        "eu-west-3": "Europe (Paris)",
        "eu-south-2": "Europe (Spain)",
        "eu-north-1": "Europe (Stockholm)",
        "eu-central-2": "Europe (Zurich)",
        "il-central-1": "Israel (Tel Aviv)",
        "me-south-1": "Middle East (Bahrain)",
        "me-central-1": "Middle East (UAE)",
        "sa-east-1": "South America (São Paulo)",
        "us-gov-east-1": "AWS GovCloud (US-East)",
        "us-gov-west-1": "AWS GovCloud (US-West)"
    }

    return region_dict.get(region, "Invalid Region")



def get_ec2_on_demand_price(instance_type: str, region: str) -> List[float]:
    """
    Retrieves the on-demand prices for the given instance type and region.

    Args:
        instance_type (str): The EC2 instance type (e.g., 't2.micro', 'm5.large').
        region (str): The AWS region (e.g., 'us-east-1', 'eu-west-1').

    Returns:
        List[float]: A list of on-demand prices (in USD) for the given instance type and region.
    """
    client = boto3.client('pricing', region_name=region)

    region_name = get_region_name(region)

    filters = [
        {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
        {'Type': 'TERM_MATCH', 'Field': 'location', 'Value': region_name},
        {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
        {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
        {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'shared'},
        {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
    ]

    response = client.get_products(
        ServiceCode='AmazonEC2',
        Filters=filters,
        MaxResults=100
    )

    prices = []
    for price in response['PriceList']:
        price_item = json.loads(price)
        for key, value in price_item['terms']['OnDemand'].items():
            for price_dimensions_key, price_dimensions_value in value['priceDimensions'].items():
                prices.append(price_dimensions_value['pricePerUnit']['USD'])


    return prices



def get_ec2_type(region):
    """
  Retrieve the instance type of the current EC2 instance running in the specified AWS region.

  Args:
      region (str): The AWS region where the EC2 instance is running.

  Returns:
      str: The instance type of the current EC2 instance, or None if an error occurred.

  Raises:
      None

  """

    try:
        # Create a session and a client for the EC2 service
        session = boto3.Session(region_name=region)
        ec2_client = session.client('ec2')

        # Get the instance ID of the current instance
        instance_id = get_ec2_instance_id(region)

        # Get the instance details
        instance_details = ec2_client.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]

        # Get and return the instance type
        instance_type = instance_details['InstanceType']
        return instance_type

    except ClientError as e:
        print(f"Error occurred: {e.response['Error']['Message']}")
        return None


def get_ec2_instance_id(region):
    """
    Retrieves the instance ID of the current EC2 instance.

    This function makes a request to the AWS Instance Metadata Service (IMDS) to retrieve the instance ID
    of the current EC2 instance. It first obtains a token from the IMDS token service, and then uses the
    token to make a request to the IMDS instance identity document service to retrieve the instance ID.

    Args:
        region (str): The AWS region in which the EC2 instance is running. This parameter is currently
            not used by the function.

    Returns:
        str: The instance ID of the current EC2 instance, or None if an error occurs while retrieving
            the instance ID.

    Raises:
        RequestException: If an error occurs while making a request to the AWS Instance Metadata Service.

    Note:
        This function assumes that it is running on an EC2 instance with access to the AWS Instance Metadata
        Service. If run outside of an EC2 instance, it will fail to retrieve the instance ID.
    """

    # Get the instance ID of the current instance
    instance_metadata_url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}

    try:
        # Get the Instance Metadata Service Request Header
        token_response = requests.put(token_url, headers=headers, timeout=5)
        token_response.raise_for_status()
        token = token_response.text

        # Use the token in the request to the instance metadata service
        headers = {"X-aws-ec2-metadata-token": token}
        response = requests.get(instance_metadata_url, headers=headers, timeout=5)
        response.raise_for_status()
        instance_data = json.loads(response.text)
        instance_id = instance_data.get("instanceId")
        return instance_id
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving instance ID: {e}")



def get_instance_region():
    """
    Retrieves the AWS region of the current EC2 instance.
    """
    instance_metadata_url = "http://169.254.169.254/latest/dynamic/instance-identity/document"
    token_url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}

    try:
        # Get the Instance Metadata Service Request Header
        token_response = requests.put(token_url, headers=headers, timeout=5)
        token_response.raise_for_status()
        token = token_response.text

        # Use the token in the request to the instance metadata service
        headers = {"X-aws-ec2-metadata-token": token}
        response = requests.get(instance_metadata_url, headers=headers, timeout=5)
        response.raise_for_status()
        instance_data = response.json()
        instance_region = instance_data.get("region")
        return instance_region
    except requests.exceptions.RequestException as e:
        print(f"Error retrieving instance region: {e}")
        return None


def get_ec2_metadata(path='/latest/meta-data/'):
    """
    Fetches EC2 instance metadata using IMDSv2.

    :param path: Metadata path to fetch.
    :return: The requested metadata as a string.
    """
    # The IMDSv2 endpoint
    token_url = 'http://169.254.169.254/latest/api/token'
    metadata_url = f'http://169.254.169.254{path}'

    # Headers for the PUT request to get the token
    token_headers = {'X-aws-ec2-metadata-token-ttl-seconds': '21600'}

    # Getting the session token
    token_response = requests.put(token_url, headers=token_headers, timeout=10)
    if token_response.status_code != 200:
        raise Exception('Failed to retrieve token for IMDSv2')

    # Headers for the GET request to get the metadata
    metadata_headers = {'X-aws-ec2-metadata-token': token_response.text}

    # Getting the metadata
    metadata_response = requests.get(metadata_url, headers=metadata_headers, timeout=10)
    if metadata_response.status_code != 200:
        raise Exception('Failed to retrieve EC2 metadata')

    return metadata_response.text


def get_instance_specs(instance_type, region=None):
    """
    Retrieves the RAM, CPU, and GPU specifications for a given Amazon EC2 instance type.

    Args:
        instance_type (str): The Amazon EC2 instance type (e.g., 't2.micro', 'm5.large', 'p3.2xlarge').
        region (str, optional): The AWS region to use. If None, uses the default region.

    Returns:
        dict: A dictionary containing the RAM, CPU, and GPU specifications for the given instance type.
              The RAM is displayed in GiB, the CPU is displayed as the number of vCPUs, and GPU is the number of GPUs.
        None: If the instance type is not found or an error occurs.

    Example:
        >>> specs = get_instance_specs('p3.2xlarge')
        >>> print(specs)
        {'RAM': '61 GiB', 'CPU': '8 vCPUs', 'GPU': '1'}
    """
    # Create an EC2 client
    ec2_client = boto3.client('ec2', region_name=region)

    try:
        # Describe the instance type
        response = ec2_client.describe_instance_types(InstanceTypes=[instance_type])

        # If the instance type is found, extract the RAM, CPU, and GPU details
        if response['InstanceTypes']:
            instance_details = response['InstanceTypes'][0]
            ram = instance_details['MemoryInfo']['SizeInMiB']
            cpu = instance_details['VCpuInfo']['DefaultVCpus']
            ram_display = f"{ram // 1024} GiB"  # Convert MiB to GiB
            cpu_display = f"{cpu} vCPUs"

            # Check for GPU information
            gpu_count = 0
            if 'GpuInfo' in instance_details:
                gpu_count = instance_details['GpuInfo']['Gpus'][0]['Count']


            specs = {
                'RAM': ram_display,
                'CPU': cpu_display,
                'GPU': str(gpu_count)
            }

            return specs
        else:
            print(f"Instance type '{instance_type}' not found.")
            return None

    except Exception as e:
        print(f"Error retrieving specs for instance type '{instance_type}': {str(e)}")
        return None


def process_monitor_results(df: pd.DataFrame, instance_type: str, region: str) -> pd.DataFrame:
    """
    Process monitoring results for an EC2 instance and add expected cost and runtime columns.

    Args:
        df (pd.DataFrame): DataFrame containing monitoring data with a 'Timestamp' column.
        instance_type (str): The EC2 instance type.
        region (str): The AWS region where the instance is running.

    Returns:
        pd.DataFrame: The original DataFrame with two new columns: 'expected_cost' and 'runtime'.
    """
    prices = get_ec2_on_demand_price(instance_type, region)
    price = float(prices[0])

    first_row = df.iloc[0]
    last_row = df.iloc[-1]
    first_timestamp = pd.to_datetime(first_row['Timestamp'])
    last_timestamp = pd.to_datetime(last_row['Timestamp'])

    # Calculate the time difference in seconds
    runtime = (last_timestamp - first_timestamp).total_seconds()

    # Convert to expected costs
    expected_cost = runtime / 3600 * price

    # Add new columns to the DataFrame
    df['expected_cost'] = expected_cost
    df['runtime'] = runtime

    return df




