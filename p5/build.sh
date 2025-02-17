#!/bin/bash

mpicc -g -Wall mpi-integrate.c -o mpi-integrate

mpiexec -n 4 -hostfile hosts ./mpi-integrate
