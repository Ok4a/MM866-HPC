# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <omp.h>
# include <time.h>
# include <string.h>

# define NoSweeps 1000
# define BILLION  1000000000L;
int **grid;

// A function for the magnetization. Takes grid length N as input
double magn(int N)
{
    // Indices over the grid
    int i = 0;
    int j = 0;

    //  Sum to be incremented
    double sum = 0.0;

    // Parallelize the double for loop over the grid
    # pragma parallel for collapse(2) reduction(+: sum)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Increment sum by the spin at grid position i, j
            sum += grid[i][j];
        }
    }
    // Divide sum by N twice to get the magnetization
    sum /= N;
    sum /= N;

    // Return the magnetization
    return (sum);
}

// A function for the energy of the grid. Takes grid length N as input
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

/* A function for the difference in energy between two grids. Takes grid length N and 
grid indices i and j as input*/
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

    // Returns energy difference
    return (diff);
}

/* A function that sweeps through the grid and changes spins if it decreases the energy.
Takes grid length N, parameter beta, grid indices start_i and start_j and a seed as input. */
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
    # pragma omp parallel for private(diff, p, rnd, seed) collapse(2)
    for (i = start_i; i < N; i += 2)
    {
        for (j = start_j; j < N; j+= 2)
        {
            // Computes energy difference with current indices
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
    /* Indices for the grid i and j, r for random ints 0 or 1
    and k for number of sweeps we perform as well as number of simulations*/
    int i = 0;
    int j = 0;
    int r = 0;
    int k = 0;

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
    
    // Creates the square grid of length N
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
    for (k = 0; k < 100; k++)
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
    for (k = 0; k < NoSweeps; k++)
    {
        sweep(N, beta, 0, 0, seed);
        sweep(N, beta, 0, 1, seed);
        sweep(N, beta, 1, 0, seed);
        sweep(N, beta, 1, 1, seed);

        // For each iteration we compute the magnetization and the energy of the grid
        mag_vec[k] = magn(N);
        energy_vec[k] = energy(N);
    }


    /* This for loop sums over the vectors mag_vec and energy_vec in order
    to compute the average energy and magnetization across all the
    simulations.*/
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
    for (i = 0; i < NoSweeps; i++)
    {
        /* Increments the sums by the square of the difference between
        current value of energy_vec / avg_vec.*/
        std_energy += (avg_energy - energy_vec[i]) * (avg_energy - energy_vec[i]);
        std_mag += (avg_mag - mag_vec[i]) * (avg_mag - mag_vec[i]);
    }

    /* Computes standard deviation of the energy and magnetization by taking the square
    root of the sums computed before divided by NoSweeps - 1. This is to get
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
void main(int argc, char const *argv[])
{
    struct timespec start, stop;
    double tot_time;
    FILE *data_file;


    char file_path[] = "OMPdata_";
    char csv[] = ".csv";
    char time_str[14];
   
    sprintf(time_str, "%ld", (unsigned long)time(NULL));
    strcat(file_path, time_str);
    strcat(file_path, csv);
    data_file = fopen(file_path, "w");
    fprintf(data_file,"%s, %s, %s, %s, %s, %s, %s, %s\n", "N", "beta", "avg_energy", "std_energy", "avg_mag", "std_mag", "tot_time", "num_threads");
    fclose(data_file);


    /* Index i, number of different grid lengths num_N and
    array of integers N_vec to store different grid lengths*/
    int i;
    int j;
    int num_N = 4;
    int *N_vec;
    int num_threads;
    
    // Parameter beta for the simulation
    double beta = 0.5;

    // Allocate space for vector of values of N
    N_vec = (int *) malloc(num_N * sizeof(int));
    
    // Assign first value of N_vec
    N_vec[0] = 100;

    // For loop to create vector of N values
    for (i = 1; i < num_N; i++)
    {
        /* New value of N is previous value multiplied by sqrt(2). This is so
        the next grid constructs a grid of doubled size compared to the grid with length
        N_vec[i - 1]*/
        N_vec[i] = N_vec[i - 1] * sqrt(2);
        if ((N_vec[i] % 2) == 1)
        {
            // If current value is odd, we add 1 to make the new N even
            N_vec[i] += 1;
        } 

    }



    for (j = 1; j < 3; j++)
    {   
        num_threads = pow(2,j);
        printf("%d", num_threads);
        
        omp_set_num_threads(num_threads);

        // A for loop to collect data for each simulation of the ISING model
        for (int i = 0; i < num_N; i++)
        {
            /* The first three lines time the process of simulating the ISING model
            using the current value of N*/
            clock_gettime(CLOCK_REALTIME, &start);
            double* data = ising(N_vec[i], beta);
            clock_gettime(CLOCK_REALTIME, &stop);

            // This line computes the total time of the simulation in seconds plus nano-seconds
            tot_time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec ) / (double)BILLION;

            // We open a test file to write out the data to
            data_file = fopen(file_path, "a");

            /* In each row of the data-file we write out the current value of N, the chosen parameter beta
            the avg and std of the energy and magnetization for the simulation and the total time
            the simulation took*/
            fprintf(data_file,"%d, %f, %f, %f, %f, %f, %f, %d\n", N_vec[i], beta, data[0], data[1], data[2], data[3], tot_time, num_threads);
            
            // We then close the file
            fclose(data_file);
        }    
    }
    
}