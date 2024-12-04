#!/usr/bin/bash
mpicc isingMPI.c -o MPIIsing  -lm

timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > MPIdata_${timestamp}.csv

N=$((100))
num_N=$((4))

for i in {1..4}
do
  echo Run $i
  for j in {1..3}
  do
      echo Number of threads: $((2**$j))
      mpirun --use-hwthread-cpus -n $((2**$j)) ./MPIIsing $timestamp $N $num_N
  done
done