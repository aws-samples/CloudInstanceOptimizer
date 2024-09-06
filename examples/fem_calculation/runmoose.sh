#!/bin/bash
singularity exec -B /:/mnt moose.sif /mnt/runscript.sh $@