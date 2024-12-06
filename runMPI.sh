#!/usr/bin/bash
mpicc isingMPI.c -o MPIIsing  -lm

timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > MPIdata_${timestamp}.csv

N=$((1000))
num_N=$((8))

for i in {1..5}
do
  echo Run $i
  for j in {1..6}
  do
      echo Number of threads: $((2**$j))
      mpirun --use-hwthread-cpus -n $((2**$j)) ./MPIIsing $timestamp $N $num_N
  done
done