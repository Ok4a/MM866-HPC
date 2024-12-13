# compile the OpenMPI Ising model c file
mpicc isingMPI.c -o MPIIsing  -lm

# UNIX time stamp  to create a unique data file name
timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > MPIdata_${timestamp}.csv

# the starting side length
N=$((1000))
# the number of different side lengths
num_N=$((8))

for i in {1..5} # for loop for running the model a set number of times
do
  echo Run $i
  for j in {1..6} # for loop for using different number of threads
  do
      echo Number of threads: $((2**$j))
      # running the OpenMPI Ising model
      mpirun --use-hwthread-cpus -n $((2**$j)) ./MPIIsing $timestamp $N $num_N
  done
done