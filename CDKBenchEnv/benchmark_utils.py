#!/usr/bin/env python3
# -*- coding: utf-8 -*-
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: MIT-0                                     #
######################################################################

import boto3



def get_instance_types_in_family(instance_type):
    """
  Returns a list of all EC2 instance types in the same family as the provided instance type.

  Args:
      instance_type (str): The EC2 instance type for which the family members should be retrieved.

  Returns:
      list: A list of EC2 instance type strings belonging to the same family as the provided instance type.

  Example:
      >>> get_instance_types_in_family('m5.large')
      ['m5.large', 'm5.xlarge', 'm5.2xlarge', 'm5.4xlarge', ...]
  """

    ec2 = boto3.client('ec2')

    # Extract the instance family from the provided instance type
    family = instance_type.split('.')[0]

    # Get all available instance types
    response = ec2.describe_instance_types()
    instance_types = response['InstanceTypes']

    # Filter for instance types in the same family
    family_instance_types = [
        instance_type['InstanceType']
        for instance_type in instance_types
        if instance_type['InstanceType'].lower().startswith(family.lower())
    ]

    return family_instance_types


def separate_ec2_types(ec2_types):
    """
   Separates a list of EC2 instance types into two lists: one containing Graviton instance types
   and another containing non-Graviton instance types.

   Args:
       ec2_types (list): A list of EC2 instance type strings (e.g., ['t3.micro', 't3a.small', 'm5.large']).

   Returns:
       tuple: A tuple containing two lists:
           - graviton_types (list): A list of Graviton instance types from the input list.
           - non_graviton_types (list): A list of non-Graviton instance types from the input list.

   Example:
       >>> ec2_types = ['t3.micro', 't3a.small', 'm5.large', 'm6g.medium']
       >>> graviton_types, non_graviton_types = separate_ec2_types(ec2_types)
       >>> print(graviton_types)
       ['m6g.medium']
       >>> print(non_graviton_types)
       ['t3.micro', 't3a.small', 'm5.large']
   """
    graviton_types = []
    non_graviton_types = []

    for instance_type in ec2_types:
        if 'g' in instance_type.split('.')[0]:
            graviton_types.append(instance_type)
        else:
            non_graviton_types.append(instance_type)

    return graviton_types, non_graviton_types
