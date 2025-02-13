from aws_cdk import (
    Duration,
    Stack,
    CfnOutput
)
from constructs import Construct
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_batch as batch
from aws_cdk import aws_iam as iam
from benchmark_utils import get_instance_types_in_family, separate_ec2_types
import boto3


def get_available_azs(region_name):
    ec2_client = boto3.client('ec2', region_name=region_name)
    response = ec2_client.describe_availability_zones(
        Filters=[
            {'Name': 'opt-in-status', 'Values': ['opt-in-not-required', 'opted-in']},
            {'Name': 'state', 'Values': ['available']}
        ]
    )
    return [az['ZoneName'] for az in response['AvailabilityZones']]


class CdkBenchEnvStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, json_config: dict, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.json_config = json_config

        # Forced to create a new VPC, mnp in batch seems to not work correctly in default vpc
        #self.vpc = ec2.Vpc(self, "VPC", max_azs=6)
        # Get all available AZs using boto3
        available_azs = get_available_azs(json_config["region_name"])
        self.vpc = ec2.Vpc(
            self,
            "benchmarkVPC",
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PUBLIC,
                    name="Public",
                    cidr_mask=22
                ),
                ec2.SubnetConfiguration(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    name="Private",
                    cidr_mask=22
                )
            ],
            availability_zones=available_azs  # Explicitly specify the AZs to get all available
        )


        if "batch_backend" in json_config.keys():
            batch_backend = json_config["batch_backend"].lower()
        else:
            batch_backend = 'ecs'


        # Create a security group for multinode communication
        self.multinode_sg = ec2.SecurityGroup(
            self,
            "MultinodeSecurityGroup",
            vpc=self.vpc,
            description="Security group for multinode Batch jobs",
            allow_all_outbound=True
        )

        # Allow all traffic between instances in this security group
        self.multinode_sg.add_ingress_rule(
            peer=self.multinode_sg,
            connection=ec2.Port.all_traffic(),
            description="Allow all traffic between nodes"
        )

        batch_iam = self.create_batch_instance_role(json_config['s3_bucket_name'], backend=batch_backend)

        #TODO: volume_size should be user option
        template = ec2.CfnLaunchTemplate(self, "CreatedTemplate",
            launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                block_device_mappings=[ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                    device_name="/dev/xvda",
                    ebs=ec2.CfnLaunchTemplate.EbsProperty(
                        delete_on_termination=True,
                        encrypted=True,
                        iops=3000,
                        throughput=123,
                        volume_size=1000,
                        volume_type="gp3"
                    ),
                )],
            ),
            launch_template_name="CDKBenchmarkTemplate"
        )

        available_ec2_types = dir(ec2.InstanceClass)
        requested_instance = json_config["ec2_types"]
        requested_type = list(map(lambda x: x.split('.')[0], requested_instance))

        with open("../valid_list.txt", 'r') as f:
            valid_list = f.read()
        valid_list = valid_list.replace('\n','').replace(' ','').split(',')

        ec2types = []
        for family_types in json_config['ec2_types']:
            ec2types.extend(get_instance_types_in_family(family_types))

        use_valid_list = json_config.get("valid_list", True)
        if use_valid_list is True or (isinstance(use_valid_list, str) and use_valid_list.lower() == 'true'):
            ec2types = [typ for typ in ec2types if typ in valid_list or typ.split('.')[0] in valid_list]
        ec2types = [typ for typ in ec2types if 'metal' not in typ]

        if len(ec2types) == 0:
            raise ValueError(f"ERROR: no valid ec2 requested. Must be one of {valid_list}")

        graviton_types_in_region, non_graviton_types_in_region = separate_ec2_types(ec2types)

        exclude_lst = json_config.get("exclude_ec2_types", [])
        graviton_types_in_region = [x for x in graviton_types_in_region
                                    if not any(exclude in x for exclude in exclude_lst)]
        non_graviton_types_in_region = [x for x in non_graviton_types_in_region
                                        if not any(exclude in x for exclude in exclude_lst)]

        instance_classes = []
        graviton_instance_classes = []
        selected_instance_types = []
        for ec2type in available_ec2_types:
            if any(x in ec2type.lower() for x in requested_type):
                ec2type_lower = ec2type.lower()

                if ec2type_lower in (x.split('.')[0] for x in non_graviton_types_in_region):
                    print(ec2type_lower)
                    selected_instance_types.append(ec2type_lower)
                    instance_classes.append(getattr(ec2.InstanceClass, ec2type))

                if "G" in ec2type and ec2type_lower in (x.split('.')[0] for x in graviton_types_in_region):
                    print(ec2type_lower)
                    selected_instance_types.append(ec2type_lower)
                    graviton_instance_classes.append(getattr(ec2.InstanceClass, ec2type))

        # Output the selected instance types
        CfnOutput(self, "SelectedInstanceTypes",
            value=", ".join(selected_instance_types),
            description="EC2 instance types selected for the compute environment"
        )


        if batch_backend == 'eks':
            raise ValueError("ERROR: EKS with Batch does not support multinode yet")

        if any([True for x in non_graviton_types_in_region if "inf" in x or "trn" in x]):
            ami_image = [batch.EcsMachineImage(
                image=ec2.MachineImage.from_ssm_parameter("/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id")
            )]
        else:
            ami_image = None

        #TODO: consider making this use input
        #MAX_TYPES_PER_ENV = 8
        MAX_TYPES_PER_ENV = 3

        def create_compute_env_and_queue(self, instance_classes, env_name, queue_name, batch_iam, template, ami_image=None):
            compute_env = batch.ManagedEc2EcsComputeEnvironment(
                self,
                env_name,
                vpc=self.vpc,
                instance_role=batch_iam,
                minv_cpus=self.json_config.get("batch_min_cpu",0),
                maxv_cpus=self.json_config.get("batch_max_cpu",100000),
                use_optimal_instance_classes=False,
                instance_classes=instance_classes,
                launch_template=ec2.LaunchTemplate.from_launch_template_attributes(
                    self,
                    f"LaunchTemplate-{env_name}",
                    launch_template_id=template.attr_launch_template_id,
                    version_number="$Latest"
                ),
                images=ami_image,
                security_groups=[self.multinode_sg]
            )

            job_queue = batch.JobQueue(
                self,
                queue_name,
                priority=10,
                job_state_time_limit_actions=[batch.JobStateTimeLimitAction(
                    max_time=Duration.minutes(json_config['job_timeout']),
                    reason=batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
                )],
            )
            job_queue.add_compute_environment(compute_env, 1)

            return compute_env, job_queue

        compute_envs = []
        job_queues = []

        for i in range(0, len(instance_classes), MAX_TYPES_PER_ENV):
            env_name = f"benchmark-batch-env-{i//MAX_TYPES_PER_ENV}"
            queue_name = f"JobQueue-ec2benchmark-{i//MAX_TYPES_PER_ENV}"
            env, queue = create_compute_env_and_queue(
                self,
                instance_classes[i:i+MAX_TYPES_PER_ENV],
                env_name,
                queue_name,
                batch_iam,
                template,
                ami_image
            )
            compute_envs.append(env)
            job_queues.append(queue)

        if 'container_arn_arm' in json_config.keys():
            graviton_compute_envs = []
            graviton_job_queues = []

            for i in range(0, len(graviton_instance_classes), MAX_TYPES_PER_ENV):
                env_name = f"benchmark-batch-graviton-env-{i//MAX_TYPES_PER_ENV}"
                queue_name = f"JobQueue-ec2benchmark-graviton-{i//MAX_TYPES_PER_ENV}"
                env, queue = create_compute_env_and_queue(
                    self,
                    graviton_instance_classes[i:i+MAX_TYPES_PER_ENV],
                    env_name,
                    queue_name,
                    batch_iam,
                    template
                )
                graviton_compute_envs.append(env)
                graviton_job_queues.append(queue)

        #TODO: 12 should be user input or a query
        if ami_image is not None:
            devices = [batch.CfnJobDefinition.DeviceProperty(
                container_path=f"/dev/neuron{i}",
                host_path=f"/dev/neuron{i}",
                permissions=["READ", "WRITE", "MKNOD"]
            ) for i in range(12)]
        else:
            devices = None

        container = json_config['container_arn']
        gpu_count = json_config.get('gpu_count', 0)
        #This gets overridden during main python run
        if isinstance(gpu_count, list):
            gpu_count=1

        container_properties_dict = {
            'image': container,
            'job_role_arn': batch_iam.role_arn,
            'command': ['python', 'run.py'],
            'privileged': bool(json_config.get("docker_privileged", False)),
            'linux_parameters': batch.CfnJobDefinition.LinuxParametersProperty(
                shared_memory_size=int(2000e3),
                devices=devices,
            ),
            'log_configuration': batch.CfnJobDefinition.LogConfigurationProperty(
                log_driver="awslogs"),
            'vcpus': 1,
            'memory': 1024
        }

        if gpu_count > 0:
            container_properties_dict['resource_requirements'] = [
                batch.CfnJobDefinition.ResourceRequirementProperty(
                    type="GPU",
                    value=str(gpu_count)
                )
            ]

        container_properties = batch.CfnJobDefinition.ContainerPropertiesProperty(**container_properties_dict)

        if 'container_arn_arm' in json_config.keys():
            containerARM = json_config['container_arn_arm']
            container_properties_dict['image'] = containerARM
            container_propertiesARM = batch.CfnJobDefinition.ContainerPropertiesProperty(**container_properties_dict)

        nnodes = json_config.get("ec2_multinode_count",1)
        #This gets overridden during main python run
        if isinstance(nnodes, list):
            nnodes=1
        job_definition = batch.CfnJobDefinition(self, "JDec2benchmark",
            type="multinode",
            container_properties=container_properties,
            timeout=batch.CfnJobDefinition.TimeoutProperty(
                attempt_duration_seconds=json_config["job_timeout"]*60
            ),
            node_properties=batch.CfnJobDefinition.NodePropertiesProperty(
                main_node=0,
                num_nodes=nnodes,
                node_range_properties=[batch.CfnJobDefinition.NodeRangePropertyProperty(
                    container=container_properties,
                    target_nodes="0:"
                )]
            )
        )

        if 'container_arn_arm' in json_config.keys():
            job_definition = batch.CfnJobDefinition(self, "JDec2benchmarkgraviton",
                type="multinode",
                container_properties=container_propertiesARM,
                timeout=batch.CfnJobDefinition.TimeoutProperty(
                    attempt_duration_seconds=json_config["job_timeout"]*60
                ),
                node_properties=batch.CfnJobDefinition.NodePropertiesProperty(
                    main_node=0,
                    num_nodes=nnodes,
                    node_range_properties=[batch.CfnJobDefinition.NodeRangePropertyProperty(
                        container=container_propertiesARM,
                        target_nodes="0:"
                    )]
                )
            )

    def create_batch_instance_role(self, s3_bucket_name, backend='ecs'):
        batch_instance_role = iam.Role(
            self,
            "ec2-benchmark-BatchInstanceRole",
            assumed_by=iam.CompositePrincipal(iam.ServicePrincipal("ec2.amazonaws.com"),
                                              iam.ServicePrincipal("eks.amazonaws.com"),
                                              iam.ServicePrincipal("ecs-tasks.amazonaws.com"))
        )

        if backend == 'ecs':
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonEC2ContainerServiceforEC2Role"))
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy"))
        elif backend == 'eks':
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly"))
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKS_CNI_Policy"))
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodePolicy"))
            batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore"))
        else:
            raise ValueError(f"ERROR: unknown backend for IAM role setup {backend}")

        s3_access_policy = iam.ManagedPolicy(self, "CustomS3AccessPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
                    resources=[
                        f"arn:aws:s3:::{s3_bucket_name}",
                        f"arn:aws:s3:::{s3_bucket_name}/*"
                    ]
                )
            ]
        )

        batch_instance_role.add_managed_policy(s3_access_policy)
        batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSBatchServiceRole"))

        batch_instance_role.add_to_policy(iam.PolicyStatement(
            actions=["cloudformation:ListStackResources"],
            resources=["*"],
        ))

        return batch_instance_role