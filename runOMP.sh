#!/usr/bin/bash
gcc isingOMP.c -lm -fopenmp -o OMPising

timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > OMPdata_${timestamp}.csv

for i in {1..4}
do
    echo Run $i
    ./OMPising $timestamp
done

