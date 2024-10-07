#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################

import psutil
from datetime import datetime, timedelta
import csv
import time
from threading import Thread
import os
from utils import get_instance_region, get_ec2_instance_id, get_ec2_type
import awswrangler as wr
import pandas
import sys


class ResourceMonitor(Thread):
    """
    A threaded resource monitor that logs system resource usage to a CSV file.

    Args:
        duration (int): The duration in seconds for which the resource monitor should run.
        filename (str): The path to the CSV file where the resource usage data will be written.
        interval (int, optional): The interval in seconds between each resource usage measurement. Defaults to 2 seconds.

    Attributes:
        duration (int): The duration in seconds for which the resource monitor should run.
        interval (int): The interval in seconds between each resource usage measurement.
        filename (str): The path to the CSV file where the resource usage data will be written.
        stop_event (bool): A flag to indicate whether the resource monitor should stop running.

    Methods:
        run(): The main method that runs the resource monitoring loop and writes the data to the CSV file.
        stop(): A method to stop the resource monitor by setting the `stop_event` flag to True.

    Usage:
        monitor = ResourceMonitor(duration=300, filename='resource_usage.csv')
        monitor.start()
        # Wait for the monitor to finish or stop it manually
        monitor.stop()
        monitor.join()
    """

    def __init__(self, duration, filename, interval=2):
        Thread.__init__(self)
        self.duration = duration
        self.interval = interval
        self.filename = filename
        self.stop_event = False

    def run(self):
        # Open a CSV file to write the data
        with open(self.filename, 'w', newline='') as csvfile:
            fieldnames = ['Timestamp', 'CPU_Usage', 'RAM_Usage', 'Disk_IO_Read', 'Disk_IO_Write', 'Network_Sent', 'Network_Received']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            start_time = datetime.now()
            end_time = start_time + timedelta(seconds=self.duration)

            # Get initial disk I/O and network counters
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()

            while datetime.now() < end_time and not self.stop_event:
                cpu_usage = psutil.cpu_percent()
                ram_usage = psutil.virtual_memory().used / (1024 ** 3)  # GB

                # Get disk I/O counters
                new_disk_io = psutil.disk_io_counters()
                disk_io_read = (new_disk_io.read_bytes - disk_io.read_bytes) / (1024 ** 2)  # MB
                disk_io_write = (new_disk_io.write_bytes - disk_io.write_bytes) / (1024 ** 2)  # MB
                disk_io = new_disk_io

                # Get network I/O counters
                new_net_io = psutil.net_io_counters()
                network_sent = (new_net_io.bytes_sent - net_io.bytes_sent) / (1024 ** 2)  # MB
                network_received = (new_net_io.bytes_recv - net_io.bytes_recv) / (1024 ** 2)  # MB
                net_io = new_net_io

                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                writer.writerow({
                    'Timestamp': timestamp,
                    'CPU_Usage': cpu_usage,
                    'RAM_Usage': ram_usage,
                    'Disk_IO_Read': disk_io_read,
                    'Disk_IO_Write': disk_io_write,
                    'Network_Sent': network_sent,
                    'Network_Received': network_received
                })
                time.sleep(self.interval)  # Adjust the sleep time as needed

    def stop(self):
        self.stop_event = True



#%% main
if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmd', dest='cmd', type=str, nargs='+', required=True, help="Cmd for the code that is stressing the system.")
    parser.add_argument('-s3', dest='s3', type=str, required=True, help="s3 bucket to write too")
    parser.add_argument('-id', dest='id', type=str, help="An id number to add to output file. ")
    parser.add_argument('-cf', dest='cf', type=str, help="Custom metric file.  At termination of test, the single number in this file "
                                                        +"will be added to the resource monitor log. This enables optimizing "
                                                        + "(minimize) on this value.")
    args = parser.parse_args()

    # Start monitoring in the background
    monitor_duration = 14400  # Duration in seconds

    s3_bucket = args.s3
    region = get_instance_region()

    ec2type = get_ec2_type(region)
    instance_id = get_ec2_instance_id(region)

    # Get the current timestamp as a unique identifier
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.id is None:
        local_file = f"{ec2type}_{instance_id}_{timestamp}.csv"
    else:
        local_file = f"{ec2type}_{args.id}_{timestamp}.csv"

    resource_monitor = ResourceMonitor(monitor_duration, local_file)
    resource_monitor.start()

    # Perform your operations or stress test here
    print("Starting stress application.")
    sys.stdout.flush()
    cmd = " ".join(args.cmd)
    print(f"Running command {cmd}")
    sys.stdout.flush()

    failed = False
    try:
        rc = os.system(cmd)

        if rc == 0:
            print('Stress application completed successfully.')
        else:
            print(f'Stress application failed with exit status {rc}.')
            failed = True
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        failed = True

    sys.stdout.flush()

    # Stop monitoring
    resource_monitor.stop()
    resource_monitor.join()
    print('Resource monitoring stopped.')

    if not failed:
        if args.cf is not None:
            print("Custom metric option selected.")
            if os.path.isfile(args.cf):
                with open(args.cf, 'r') as f:
                    metric = f.read()
                csv_logs = pandas.read_csv(local_file)
                csv_logs['custom_metric'] = [metric] * csv_logs.shape[0]
                wr.s3.to_csv( df = csv_logs, path=f's3://{s3_bucket}/{local_file}')
            else:
                print("ERROR: Could not find the custom metric file.")
        else:
            wr.s3.upload(local_file=local_file, path=f's3://{s3_bucket}/{local_file}')

    print("End")

    sys.stdout.flush()


