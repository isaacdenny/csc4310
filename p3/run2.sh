#!/bin/bash

g++ -Wall min_parallel.cpp -o mp -fopenmp -lpthread

output=$(./mp 1000000 10 n | grep -o '[0-9].[0-9]\+')
echo $output
read -d " " time_generate time_serial serial_min time_manual manual_min time_reduction reduction_min <<< "$output"

echo $time_generate 
echo $time_serial
