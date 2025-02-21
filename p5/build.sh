#!/bin/bash

mpicc -g -Wall mpi-integrate.c -o integral -lm

mpiexec -n 4 -hostfile hosts ./integral 1 7 0.001
