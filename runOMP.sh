#!/usr/bin/bash
gcc isingOMP.c -lm -fopenmp -o OMPIsing

timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > OMPdata_${timestamp}.csv

N=$((100))
num_N=$((4))

for i in {1..4}
do
    echo Run $i
    ./OMPIsing $timestamp $N $num_N
done

