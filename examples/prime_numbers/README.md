## Simple example

In this example we show the basics of running the ec2benchmarking.  A simple prime number calculator will be run on the ec2 families specified in the ```benchmark_config.json``` file with the ```"ec2_types"``` json key. 

Build the container from the parent directory:

```
docker build . -f ./examples/prime_numbers/Dockerfile -t ec2benchmark
```

If you wish to test ARM architectures such as Graviton EC2 instances, you must provision a Graviton EC2 instance and build the ARM container from that instance.

```
docker build . -f ./examples/prime_numbers/Dockerfile-ARM -t ec2benchmarkarm
```

Once the containers are built, push them to ECR (Elastic Container Registry) and update the json config file with the correct ARN of the containers.

Note the ```"run_cmd"``` which specifies the command that will be run in the container on each EC2 instance.  Users only need to modify what comes after the ```--cmd``` portion.  In this example we are calling a python script called ```stress_test.py```.  Thus the system monitor starts logging all activity on the EC2 instance, then executes the user command. Once completed, the system monitor closes all logs and pushes them to the S3 Bucket specified in the config file. In the inferentia example we can see why we the ```python monitor_system.py``` is in the json.  The inferentia example includes multiple different python environments and thus the json allows for using a specific environment to run the benchmarking tool. 

To run this example:

```
python run_benchmark.py -j ./examples/prime_numbers/benchmark_config.json
```

Note that some instances are either quite popular causing low availability or do not have many instances available in the wild.  Or, a user knows a priori that they do not want to review specific instance types or families. The input file will accept an EC2 exclusion filter in the json input:

```
"exclude_ec2_types":[ "m6id", "m5n", ".16xlarge", ".24xlarge", ".32xlarge" ,".48xlarge"  ]
```
