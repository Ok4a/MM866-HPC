#!/usr/bin/bash
mpicc isingMPI.c -o MPIising  -lm

timestamp=$(date +%s)
#echo $timestamp

echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > MPIdata_${timestamp}.csv

for i in {1..3}
do
    echo $((2**$i))
    mpirun --use-hwthread-cpus -n $((2**$i)) ./MPIising $timestamp
    wait
done