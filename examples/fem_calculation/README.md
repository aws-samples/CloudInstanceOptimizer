## Building MOOSE Containers for AWS Batch

The [MOOSE](https://mooseframework.inl.gov/index.html) FEM has moved their pre-built containers to singularity.  As of 8/12/24, AWS Batch does not support singularity and thus, the container needs to be converted to a Docker.

To do this conversion, we first must install [Singularity](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html).

Then, inside the ```./examples/fem_calculation/``` directory, generate a singularity image:

```singularity build moose.sif docker://idaholab/moose:latest```

Now that we have the latest singularity MOOSE image, we can build our Docker container which installs singularity inside the docker and enables running the latest MOOSE container on AWS Batch. **OPTIONAL:** You can update the example Dockerfile to use the latest singularity versions found in the instructions [here](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html).

```docker build . -f ./examples/fem_calculation/Dockerfile-FEM -t ec2benchmark```

Notice the ```runmoose.sh``` is a convenience script that includes:

```singularity exec -B /:/mnt moose.sif /mnt/runscript.sh $@```

which starts the MOOSE container inside the Docker container. All command line arguments are passed to the user defined ```runscript.sh```, which executes inside the MOOSE container.

For this example, no additional arguments are passed to MOOSE.  However, in the optimization demo, we provide the values for a hyperparameter to tune. 

As a final note, since we are using a singularity container inside a docker container, we must send the "privileged" option to ensure the singularity container can read the underlying filesystem within the parent container.  

Hence, as an example ensure the following ```docker_privileged``` is in your json config file:

```
"run_cmd": "/env_benchmark/bin/python /monitor_system.py --cmd ./runmoose.sh ARGS",
"docker_privileged": "True",
"ec2_types":["c6"],
```
