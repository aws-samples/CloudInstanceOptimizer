
from aws_cdk import (
    Duration,
    Stack
)
from constructs import Construct
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_batch as batch
from aws_cdk import aws_iam as iam
from benchmark_utils import get_instance_types_in_family, separate_ec2_types


class CdkBenchEnvStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, json_config: dict, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)


        # Forced to create a new VPC, mnp in batch seems to not work correctly in default vpc
        self.vpc = ec2.Vpc(self, "VPC", max_azs=6)

        if "batch_backend" in json_config.keys():
            batch_backend = json_config["batch_backend"].lower()
        else:
            batch_backend = 'ecs'

        batch_iam = self.create_batch_instance_role( json_config['s3_bucket_name'],  backend=batch_backend)

        #TODO: should the volume size be a user input?
        template =  ec2.CfnLaunchTemplate(self, "CreatedTemplate",
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
                            ),
                                ],
                        ),
                        launch_template_name="CDKBenchmarkTemplate"
                    )


        available_ec2_types = dir(ec2.InstanceClass)
        requested_instance = json_config["ec2_types"]
        requested_type = list(map(lambda x: x.split('.')[0], requested_instance))

        #TODO: ugh
        with open("../valid_list.txt", 'r') as f:
            valid_list = f.read()
        valid_list = valid_list.replace('\n','').replace(' ','').split(',')


        #Get all available in region
        ec2types = []
        for family_types in json_config['ec2_types']:
            ec2types.extend(  get_instance_types_in_family(family_types) )

        ec2types = [typ for typ in ec2types if typ in valid_list or typ.split('.')[0] in valid_list]
        ec2types = [typ for typ in ec2types if 'metal' not in typ]

        if len(ec2types) == 0:
            raise ValueError(f"ERROR: no valid ec2 requested. Must be one of {valid_list}")

        #seperate graviton (ARM) types
        graviton_types_in_region, non_graviton_types_in_region = separate_ec2_types(ec2types)

        instance_classes = []
        graviton_instance_classes = []
        for ec2type in available_ec2_types:
            if any([x in ec2type.lower() for x in requested_type]):
                if "G" in ec2type:
                    if any([ec2type.lower() == x.split('.')[0] for x in graviton_types_in_region]):
                        print(ec2type.lower())
                        graviton_instance_classes.append( getattr( ec2.InstanceClass, ec2type)    )
                else:
                    if any([ec2type.lower() == x.split('.')[0] for x in non_graviton_types_in_region]):
                        print(ec2type.lower())
                        instance_classes.append( getattr( ec2.InstanceClass, ec2type)    )


        #TODO: multinode is not available with Batch -> EKS, which we need for this app
        if batch_backend == 'eks':
            raise ValueError("ERROR: EKS with Batch does not support multinode yet")
        else:

            if any([True for x in non_graviton_types_in_region if "inf" in x or "trn" in x ]):
                ami_image = [batch.EcsMachineImage(
                                                image = ec2.MachineImage.from_ssm_parameter( "/aws/service/ecs/optimized-ami/amazon-linux-2023/neuron/recommended/image_id"  )
                                                )]
            else:
                ami_image = None


            batch_compute_environment = batch.ManagedEc2EcsComputeEnvironment(
                self,
                "benchmark-batch-env",
                vpc=self.vpc,
                instance_role = batch_iam,
                minv_cpus=0,
                maxv_cpus=100000,
                use_optimal_instance_classes = False,
                instance_classes=instance_classes,
                launch_template  = ec2.LaunchTemplate.from_launch_template_attributes(self,
                                                                                    "LaunchTemplate-bench",
                                                                                    launch_template_id = template.attr_launch_template_id,
                                                                                    version_number ="$Latest") ,
                images = ami_image
                )


            if 'container_arn_arm' in json_config.keys():
                batch_compute_environment_G = batch.ManagedEc2EcsComputeEnvironment(
                    self,
                    "benchmark-batch-graviton-env",
                    vpc=self.vpc,
                    instance_role = batch_iam,
                    minv_cpus=0,
                    maxv_cpus=100000,
                    use_optimal_instance_classes = False,
                    instance_classes=graviton_instance_classes,
                    launch_template  = ec2.LaunchTemplate.from_launch_template_attributes(self,
                                                                                        "LaunchTemplate-bench-graviton",
                                                                                        launch_template_id = template.attr_launch_template_id,
                                                                                        version_number = "$Latest")
                    )

            #TODO: do we need 12 to be a user input or maybe query how many there are available for neuron devices?
            if ami_image is not None:
                devices=[batch.CfnJobDefinition.DeviceProperty(
                                                container_path=f"/dev/neuron{i}",
                                                host_path=f"/dev/neuron{i}",
                                                permissions=["READ", "WRITE", "MKNOD"]
                                                )
                                                for i in range(12)
                                                ]
            else:
                devices = None

            container = json_config['container_arn']
            container_properties = batch.CfnJobDefinition.ContainerPropertiesProperty(
                            image=container,
                            job_role_arn=batch_iam.role_arn,
                            command=['python', 'run.py'],
                            privileged=bool(json_config.get("docker_privileged",False)),
                            linux_parameters = batch.CfnJobDefinition.LinuxParametersProperty(
                                shared_memory_size=int(2000e3),
                                devices=devices,
                                ),
                            log_configuration=batch.CfnJobDefinition.LogConfigurationProperty(
                                log_driver="awslogs"),
                            vcpus=1,
                            memory=1024
                            # resource_requirements= [batch.CfnJobDefinition.ResourceRequirementProperty(
                            #                             type="GPU",
                            #                             value="1"
                            #                         )],

                            )

            if 'container_arn_arm' in json_config.keys():
                containerARM = json_config['container_arn_arm']
                container_propertiesARM = batch.CfnJobDefinition.ContainerPropertiesProperty(
                                image=containerARM,
                                job_role_arn=batch_iam.role_arn,
                                command=['python', 'run.py'],
                                privileged=bool(json_config.get("docker_privileged",False)),
                                linux_parameters = batch.CfnJobDefinition.LinuxParametersProperty(
                                    shared_memory_size=int(2000e3),
                                    devices=devices,
                                    ),
                                log_configuration=batch.CfnJobDefinition.LogConfigurationProperty(
                                    log_driver="awslogs"),
                                vcpus=1,
                                memory=1024
                                # resource_requirements= [batch.CfnJobDefinition.ResourceRequirementProperty(
                                #                             type="GPU",
                                #                             value="1"
                                #                         )],
                                )


            #define Batch job
            #TODO: probably eventually want to benchmark actual multinodes as well
            nnodes = 1
            #https://docs.aws.amazon.com/cdk/api/v2/python/aws_cdk.aws_batch/CfnJobDefinition.html
            job_definition = batch.CfnJobDefinition(self, "JDec2benchmark",
                                                    type="multinode",
                                                    container_properties=container_properties,
                                                    timeout = batch.CfnJobDefinition.TimeoutProperty(
                                                            attempt_duration_seconds=json_config["job_timeout"]*60
                                                        ),
                                                    node_properties=batch.CfnJobDefinition.NodePropertiesProperty(
                                                        main_node=0,
                                                        num_nodes=nnodes,
                                                        node_range_properties=[batch.CfnJobDefinition.NodeRangePropertyProperty(
                                                            container = container_properties,
                                                            target_nodes="0:"# + str(nnodes-1)
                                                            )]
                                                  )
                                                    )

            if 'container_arn_arm' in json_config.keys():
                job_definition = batch.CfnJobDefinition(self, "JDec2benchmarkgraviton",
                                                        type="multinode",
                                                        container_properties=container_propertiesARM,
                                                        timeout = batch.CfnJobDefinition.TimeoutProperty(
                                                                attempt_duration_seconds=json_config["job_timeout"]*60
                                                            ),
                                                        node_properties=batch.CfnJobDefinition.NodePropertiesProperty(
                                                            main_node=0,
                                                            num_nodes=nnodes,
                                                            node_range_properties=[batch.CfnJobDefinition.NodeRangePropertyProperty(
                                                                container = container_propertiesARM,
                                                                target_nodes="0:"# + str(nnodes-1)
                                                                )]
                                                      )
                                                        )

        #add queue to the compute environment
        job_queue = batch.JobQueue(self,
                                    "JobQueue-ec2benchmark",
                                    priority=10,
                                    job_state_time_limit_actions = [batch.JobStateTimeLimitAction(
                                            max_time=Duration.minutes(json_config['job_timeout']),
                                            reason=batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
                                        )],
                                    )
        job_queue.add_compute_environment(batch_compute_environment, 1)


        if 'container_arn_arm' in json_config.keys():
            job_queue_G = batch.JobQueue(self,
                                        "JobQueue-ec2benchmark-graviton",
                                        priority=10,
                                        job_state_time_limit_actions = [batch.JobStateTimeLimitAction(
                                                max_time=Duration.minutes(json_config['job_timeout']),
                                                reason=batch.JobStateTimeLimitActionsReason.JOB_RESOURCE_REQUIREMENT,
                                            )],
                                        )
            job_queue_G.add_compute_environment(batch_compute_environment_G, 1)




    def create_batch_instance_role(self, s3_bucket_name, backend='ecs' ):
          '''any of the AWS Batch jobs will require the ability to read,
          write, overwrite, or delete from s3 buckets
          '''

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


          # Create custom policies
          s3_access_policy = iam.ManagedPolicy(self, "CustomS3AccessPolicy",
            statements=[
                iam.PolicyStatement(
                    actions=["s3:GetObject", "s3:PutObject"],
                    resources=[f"arn:aws:s3:::{s3_bucket_name}/*"]
                )
            ]
          )

          batch_instance_role.add_managed_policy(s3_access_policy)
          batch_instance_role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSBatchServiceRole"))

          batch_instance_role.add_to_policy(iam.PolicyStatement(
              actions=[
                       "cloudformation:ListStackResources"
                       ],
              resources=["*"],
          ))
          return batch_instance_role
