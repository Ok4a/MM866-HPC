# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <omp.h>
# include <time.h>
# include <string.h>

/* Variables NoSweeps for the number of iterations of the ISING model
we perform and BILLION to be used to convert nano-seconds to seconds.*/
# define NoSweeps 1000
# define BILLION  1000000000L;

/* Allocate array of pointers for the grid*/
int **grid;

// A function for the magnetization. Takes grid side length N as input
double magn(int N)
{
    // Indices over the grid
    int i = 0;
    int j = 0;

    // Variable to store the sum over spins
    double sum = 0.0;

    // Parallelize the double for loop over the grid
    # pragma parallel for collapse(2) reduction(+: sum)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Increment sum by the spin at grid position [i][j]
            sum += grid[i][j];
        }
    }
    // Divide sum by N twice to get the magnetization
    sum /= N;
    sum /= N;

    // Return the magnetization
    return (sum);
}

// A function for the energy of the grid. Takes grid side length N as input
int energy(int N)
{
    // Indices i and j for the grid and variables sum and spin_sum
    int i = 0;
    int j = 0;
    int sum = 0;
    int spin_sum = 0;

    // Indices of the right, left, up and down neighbours of the current element in the grid
    int right;
    int left;
    int up;
    int down;
    
    // Parallelize the double for loop over the grid
    # pragma omp parallel for private(right, left, up, down, spin_sum, j) collapse(2) reduction(+:sum)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Find the indices of the neighbours with periodic boundary conditions
            right = (i + 1) % N;
            left = (i - 1 + N) % N;
            up = (j - 1 + N) % N;
            down = (j + 1) % N;

            // Evaluate the spin sum
            spin_sum = grid[right][j] + grid[left][j] + grid[i][up] + grid[i][down];

            // Increment energy by the spin_sum
            sum += -1 * grid[i][j] * spin_sum;
        }
    }
    // Return the energy
    return (sum);
}

/* A function for the difference in energy between two grids. Takes grid side length N and 
grid position indices i and j as input*/
int energy_diff(int N, int i, int j)
{
    // Variables diff and spin_sum
    int diff = 0;
    int spin_sum = 0;

    // Finds neighbours of grid[i][j]
    int right = (i + 1) % N;
    int left = (i - 1 + N) % N;
    int up = (j - 1 + N) % N;
    int down = (j + 1) % N;

    // Evaluates the spin_sum at grid[i][j]
    spin_sum = grid[right][j] + grid[left][j] + grid[i][up] + grid[i][down];

    /* Computes the difference in energy between the previous grid and the grid
    with changed spin at grid[i][j]*/
    diff = -1 * ((-1 * grid[i][j]) * spin_sum - grid[i][j] * spin_sum);

    // Returns the energy difference
    return (diff);
}

/* A function that sweeps through the grid and changes spins based on the ISING model.
Takes grid side length N, parameter beta, grid position indices start_i and start_j 
and a seed as input. */
void sweep(int N, double beta, int start_i, int start_j, unsigned int seed){
    // Indices for grid position
    int i;
    int j;

    // Variable diff
    int diff;

    // Initialize variables p and rnd for the simulation
    double p = 0.0;
    double rnd = 0.0;
    
    // Parallelize double for loop over the grid
    # pragma omp parallel for private(diff, p, rnd, seed, j) collapse(2)
    for (i = start_i; i < N; i += 2)
    {
        for (j = start_j; j < N; j+= 2)
        {
            // Computes energy difference with current position flipped
            diff = energy_diff(N, i, j);
            if (diff <= 0)
            {
                // If energy_diff is negative we flip the spin
                grid[i][j] = -grid[i][j];
            }
            else
            {
                /* Otherwise, we flip the spin with probability exp(-beta * diff)
                according to the ISING model*/
                p = exp(-beta * diff);
                rnd = rand_r(&seed) / ((double) RAND_MAX);
                if (rnd < p)
                {
                    grid[i][j] = -grid[i][j];
                }
            }
        }
    }
}

// Function that simulates the ISING model
double* ising(int N, double beta)
{ 
    /* Indices for the grid i and j and k for number of sweeps
    we perform as well as number of simulations*/
    int i = 0;
    int j = 0;

    /* A variable r we will use to generate random numbers*/
    int r = 0;

    // Vector for output values of ising model
    double* output = (double*) malloc(4 * sizeof(double));
    unsigned int seed;

    // Creates arrays to store energy and magnetization for each iteration
    double energy_vec[NoSweeps];
    double mag_vec[NoSweeps];

    /* Variables avg_mag and avg_energy for storing the computed average magnetization
    and energy*/
    double avg_mag = 0.0;
    double avg_energy = 0.0;

    // Variables for storing the standard deviation of the energy and the magnetization
    double std_energy = 0.0;
    double std_mag = 0.0;
    
    // Creates the square grid of side length N
    grid = (int **) malloc(N * sizeof(int *));
    for (i = 0; i < N; i++)
    {
        grid[i] = (int *) malloc(N * sizeof(int));
    }

    // Gives a random seed to each thread
    # pragma omp parallel private(i, j, r, seed)
    seed = 42 + omp_get_thread_num();

    // Parallelizes the initialization of the grid
    # pragma omp for collapse(2)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Generates randomly 0 or 1
            r = rand_r(&seed) % 2;
            if (r)
            {
                // If r == 1, set the current position of the grid to 1
                grid[i][j] = 1;
            }
            else
            {
                // If r == 0, set the current position of the grid to -1
                grid[i][j] = -1;
            }
        }
    }

    /* A for loop that sweeps through the grid and changes spins according to the
    ISING model. This and the following for loop do the same thing, this for-loop
    takes care of the burn-in period.*/
    for (i = 0; i < 100; i++)
    {
        /* Sweeps each combination of even and odd indices to avoid
        interdependence between sweeps across different threads.*/
        sweep(N, beta, 0, 0, seed);
        sweep(N, beta, 0, 1, seed);
        sweep(N, beta, 1, 0, seed);
        sweep(N, beta, 1, 1, seed);
    }

    /* Simulates the ISING model for NoSweeps number of iterations after the burn-in
    period*/
    for (i = 0; i < NoSweeps; i++)
    {
        sweep(N, beta, 0, 0, seed);
        sweep(N, beta, 0, 1, seed);
        sweep(N, beta, 1, 0, seed);
        sweep(N, beta, 1, 1, seed);

        // For each iteration we compute the magnetization and the energy of the grid
        mag_vec[i] = magn(N);
        energy_vec[i] = energy(N);
    }


    /* This for loop sums over the vectors mag_vec and energy_vec in order
    to compute the average energy and magnetization across all the
    simulations.*/
    # pragma omp parallel for reduction(+ : avg_mag, avg_energy)
    for (i = 0; i < NoSweeps; i++)
    {
        /* Increments sums avg_mag and avg_energy by the value
        of mag_vec and energy_vec at the i'th iteration
        of the simulation*/
        avg_mag += mag_vec[i];
        avg_energy += energy_vec[i];
    }
    /* Divide avg_mag and avg_energy by NoSweeps to get the average
    magnetization and energy*/
    avg_mag /= NoSweeps;
    avg_energy /= NoSweeps;

    // A for-loop to compute the standard deviation of the magnetization and the energy
    # pragma omp parallel for reduction(+ : std_energy, std_mag)
    for (i = 0; i < NoSweeps; i++)
    {
        /* Increments the sums by the square of the difference between
        current value of energy_vec / avg_vec and the average of the corresponding
        variables.*/
        std_energy += (avg_energy - energy_vec[i]) * (avg_energy - energy_vec[i]);
        std_mag += (avg_mag - mag_vec[i]) * (avg_mag - mag_vec[i]);
    }

    /* Computes standard deviation of the energy and magnetization by taking the square
    root of the sums computed before dividing by NoSweeps - 1. This is to get
    the unbiased sample-variance and the square-root get the estimated
    standard deviation*/
    std_energy = sqrt(std_energy/(NoSweeps - 1));
    std_mag = sqrt(std_mag/(NoSweeps - 1));

    /* Assigns the values avg_energy, std_energy, avg_mag and std_mag
    to be outputs of the function*/
    output[0] = avg_energy;
    output[1] = std_energy;
    output[2] = avg_mag;
    output[3] = std_mag;

    // Returns the output vector
    return(output);
}

// Main function
void main(int argc, char **argv)
{
    /* Create variables start and stop of type timespec*/
    struct timespec start, stop;

    /* A float for the total time*/
    double tot_time;
    
    /* Points to the file we will send the data to*/
    FILE *data_file;

    /* Allocates string for the unique filepath */
    char file_path[26] = "OMPdata_";

    /* ".csv" suffix to be added last*/
    char csv[] = ".csv";
   
    /* Specifies the file the data is to be added to*/
    strcat(file_path, argv[1]);
    strcat(file_path, csv);


    /* Index i and j*/
    int i;
    int j;
    
    /* Int for the number of threads we use*/
    int num_threads;
    
    // Parameter beta for the simulation
    double beta = 0.5;
  
    
    /* Number of different grid side lengths num_N, array of integers N_vec
    to store different grid side lengths and assigns first value of N_vec */
    int num_N = strtod(argv[3], NULL);
    int N_vec[num_N];
    N_vec[0] = strtod(argv[2], NULL);

    // For loop to create vector of N values
    for (i = 1; i < num_N; i++)
    {
        /* We want to approximately double the number of entries in the grid
        so we multiply the previous value of N by sqrt(2)*/
        N_vec[i] = N_vec[i - 1] * sqrt(2);      
    } 

    for (i = 0; i < num_N; i++)
    {
        /* We would like the side length of the grid to be divisible by the max
        thread count. We will at most use 64 threads. 64 is a power of 2, so every time we
        double the number of threads, the new number of threads will be divisible by 64 and
        consequently, the grid side length will also be divisible by that number of threads */
        N_vec[i] += 64 - (N_vec[i] % 64);
    }



    for (i = 0; i < 7; i++)
    {   
        /* We pick the number of threads to be 2^i*/
        num_threads = pow(2, i);

        /* Prints the number of threads we currently use*/
        printf("Number of threads: %d\n", num_threads);
        
        /* Sets the current number of used threads*/
        omp_set_num_threads(num_threads);

        // A for loop to collect data for each simulation of the ISING model
        for (int j = 0; j < num_N; j++)
        {
            /* The first three lines time the process of simulating the ISING model
            using the current value of N*/
            clock_gettime(CLOCK_REALTIME, &start);
            double* data = ising(N_vec[j], beta);
            clock_gettime(CLOCK_REALTIME, &stop);

            // This line computes the total time of the simulation in seconds plus nano-seconds
            tot_time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec ) / (double)BILLION;

            // We open a test file to append the data to
            data_file = fopen(file_path, "a");

            /* In each row of the data-file we write out the current value of N, the chosen parameter beta
            the avg and std of the energy and magnetization for the simulation and the total time
            the simulation took*/
            fprintf(data_file,"%d, %f, %f, %f, %f, %f, %f, %d\n", N_vec[j], beta, data[0], data[1], data[2], data[3], tot_time, num_threads);
            
            // We then close the file
            fclose(data_file);
        }    
    }
    
}