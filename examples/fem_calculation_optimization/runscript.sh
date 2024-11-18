#!/bin/bash
mpirun -n $1 moose-opt -i stressed.i
