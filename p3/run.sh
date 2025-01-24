#!/bin/bash

g++ -Wall min_parallel.cpp -o mp -fopenmp -lpthread


total_serial=0
total_manual=0
total_reduction=0

# 1 million elements
for n in {1..20};
do
	output=$(./mp 10000000 10 n | grep -o '[0-9].[0-9]\+')
	read -d " " time_generate time_serial serial_min time_manual manual_min time_reduction reduction_min <<< "$output"
	
	# add new values to sums
	total_serial=$(echo "$total_serial + $time_serial" | bc)
	total_manual=$(echo "$total_manual + $time_manual" | bc)
	total_reduction=$(echo "$total_reduction + $time_reduction" | bc)

done

avg_serial=$(echo "scale=10; $total_serial / 20" | bc)
avg_manual=$(echo "scale=10; $total_manual / 20" | bc)
avg_reduction=$(echo "scale=10; $total_reduction / 20" | bc)

speedup_manual=$(echo "scale=10; $avg_serial / $avg_manual" | bc)
speedup_reduction=$(echo "scale=10; $avg_serial / $avg_reduction" | bc)

echo "Avg serial: $avg_serial"
echo "Avg manual: $avg_manual"
echo "Avg reduction: $avg_reduction"
echo "Speedup serial/manual: $speedup_manual"
echo "Speedup serial/reduction: $speedup_reduction"
