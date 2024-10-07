#!/usr/bin/env python3
######################################################################
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. #
# SPDX-License-Identifier: Apache-2.0                                #
######################################################################

import json

import aws_cdk as cdk

from cdk_bench_env.cdk_bench_env_stack import CdkBenchEnvStack
from cdk_nag import AwsSolutionsChecks, NagSuppressions
from aws_cdk import Aspects




def get_json(filename):
    with open(filename,'r') as f:
        config = json.load(f)
    return config



#%%
app = cdk.App()

'''
Use the following command to pass in different json inputs.
cdk synth -c json_file_path=/path/to/your/config.json

e.g.

cdk synth -c json_file_path=../examples/inferentia_tune/benchmark_config.json
cdk deploy -c json_file_path=../examples/inferentia_tune/benchmark_config.json

cdk synth -c json_file_path=../examples/prime_numbers/benchmark_config.json
cdk deploy -c json_file_path=../examples/prime_numbers/benchmark_config.json

cdk synth -c json_file_path=../examples/fem_calculation_optimization/benchmark_config.json
cdk deploy -c json_file_path=../examples/fem_calculation_optimization/benchmark_config.json

cdk synth -c json_file_path=../examples/fem_calculation/benchmark_config.json
cdk deploy -c json_file_path=../examples/fem_calculation/benchmark_config.json

cdk synth -c json_file_path=../examples/gpu_training_optimization/benchmark_config.json
cdk deploy -c json_file_path=../examples/gpu_training_optimization/benchmark_config.json
'''

json_file_path = app.node.try_get_context("json_file_path")
if json_file_path:
    json_config = get_json(json_file_path)
else:
    json_config = get_json('../benchmark_config.json')  # Default JSON file


stack = CdkBenchEnvStack(app, "CdkBenchEnvStack", json_config
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.

    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),

    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */

    #env=cdk.Environment(account='123456789012', region='us-east-1'),

    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
    )

Aspects.of(stack).add(AwsSolutionsChecks())

NagSuppressions.add_stack_suppressions(stack, [
    {"id": "AwsSolutions-IAM4", "reason": "Open for ease of use, user needs to select their own."},
    {"id": "AwsSolutions-IAM5", "reason": "Open for ease of use, user needs to select their own."},
    {"id": "AwsSolutions-VPC7", "reason": "Flow Logs not required for this VPC, arch should be temporary."}
])

app.synth()


