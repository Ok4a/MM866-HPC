# compile the OpenMP Ising model c file
gcc isingOMP.c -lm -fopenmp -o OMPIsing

# UNIX time stamp  to create a unique data file name
timestamp=$(date +%s)
echo "N, beta, avg_energy, std_energy, avg_mag, std_mag, tot_time, num_threads" > OMPdata_${timestamp}.csv

# the starting side length
N=$((1000))
# the number of different side lengths
num_N=$((8))

for i in {1..5} # for loop for running the model a set number of times
do
    echo Run $i
    # running the OpenMP ising model
    ./OMPIsing $timestamp $N $num_N
done

