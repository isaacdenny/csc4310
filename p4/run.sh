#!/bin/bash

gcc -Wall img_pipeline.c -o ip -fopenmp -lpthread


total_serial=0
total_parallel=0

# 1 million elements
for n in {1..20};
do
	output=$(./ip sample_5184x3456.ppm out | grep -o '[0-9].[0-9]\+')
	echo $output
	read -d " " time_serial time_parallel <<< "$output"
	
	# add new values to sums
	total_serial=$(echo "$total_serial + $time_serial" | bc)
	total_parallel=$(echo "$total_parallel + $time_parallel" | bc)

done

avg_serial=$(echo "scale=10; $total_serial / 20" | bc)
avg_parallel=$(echo "scale=10; $total_parallel / 20" | bc)

speedup_parallel=$(echo "scale=10; $avg_serial / $avg_parallel" | bc)

echo "Avg serial: $avg_serial"
echo "Avg parallel: $avg_parallel"
echo "Speedup serial/parallel: $speedup_parallel"
