#!/usr/bin/bash
mpicc isingMPI.c -o MPIising  -lm

timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > MPIdata_${timestamp}.csv

for i in {1..4}
do
  echo Run $i
  for j in {0..3}
  do
      echo Number of threads: $((2**$j))
      mpirun --use-hwthread-cpus -n $((2**$j)) ./MPIising $timestamp
  done
done