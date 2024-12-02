#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
# include <time.h>

#define NoSweeps 1000
#define BILLION  1000000000L;
int **grid;

// A function for the magnetization. Takes grid length N as input
double magn(int N, int num_rows)
{
    // Indices over the grid
    int i = 0;
    int j = 0;

    //  Sum to be incremented
    double sum = 0.0;

    // Parallelize the double for loop over the grid
    for (i = 1; i < num_rows-2; i++)
    {
        for (j = 0; j < N; j++)
        {
            // Increment sum by the spin at grid position i,j
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
int energy(int N, int num_rows)
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
    for (i = 1; i < num_rows-2; i++)
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

// A function for the difference in energy between two grids. Takes grid length N and 
// grid indices i and j as input
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

    // Computes the difference in energy between the previous grid and the grid
    // with changed spin at grid[i][j]
    diff = -1 * ((-1 * grid[i][j]) * spin_sum - grid[i][j] * spin_sum);

    // Returns energy difference
    return (diff);
}

// A function that sweeps through the grid and changes spins if it decreases the energy.
// Takes grid length N, parameter beta, grid indices start_i and start_j and a seed as input.
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
    for (i = start_i; i < N; i += 2)
    {
        for (j = start_j; j < N; j+= 2)
        {
            // Computes energy difference with curren
            diff = energy_diff(N, i, j);
            if (diff <= 0)
            {
                grid[i][j] = -grid[i][j];
            }
            else
            {
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

double* ising(int N, double beta)
{ 
    MPI_Init(NULL, NULL);
    int num_threads = 0;
    int id = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    MPI_Status status;
    MPI_Request req;
    
    int num_rows = N/num_threads + 2;


    int i = 0;
    int j = 0;
    int r = 0;
    int k = 0;



    unsigned int seed;

    double *mag_vec;
    int *energy_vec;

    if (id == 0)
    {
        mag_vec = (double *) malloc(NoSweeps * sizeof(double));
        energy_vec = (int *) malloc(NoSweeps * sizeof(double));
    }
    
    double mag_temp = 0.0;
    int energy_temp = 0.0;

    // Vector for output values of ising
    /*double* output = (double*) malloc(4 * sizeof(double));
    

    int diff = 0;

    double p = 0.0;
    double rnd = 0.0;
    double mag;
    int ener;

    double avg_mag = 0.0;
    double avg_energy = 0.0;

    double std_energy = 0.0;
    double std_mag = 0.0;*/
    
    grid = (int **) malloc(num_rows * sizeof(int *));
    for (i = 0; i < num_rows; i++)
    {
        grid[i] = (int *) malloc(N * sizeof(int));
    }

    seed = 42 + id;
    for (i = 1; i < num_rows - 1; i++)
    {
        for (j = 0; j < N; j++)
        {
            r = rand_r(&seed) % 2;
            if (r)
            {
                grid[i][j] = 1;
            }
            else
            {
                grid[i][j] = -1;
            }
        }
    }
    //printf("id: %d: %d, %d, %d, %d\n", id, grid[0][1],grid[1][1],grid[num_rows-2][1],grid[num_rows-1][1]);
    

    if ((id % 2) == 0)
    {
        // Sends grid information from the even numbered threads to the odd
        MPI_Send(grid[1], N, MPI_INT, (id-1+num_threads%num_threads), id, MPI_COMM_WORLD);
        MPI_Send(grid[num_rows-2], N, MPI_INT, (id + 1 + num_threads % num_threads), id, MPI_COMM_WORLD);
        
        // Resv grid information from threads with odd id to even id
        MPI_Recv(grid[num_rows-1], N, MPI_INT, (id + 1 + num_threads % num_threads), (id + 1 + num_threads % num_threads), MPI_COMM_WORLD, &status);
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads % num_threads), (id - 1 + num_threads % num_threads), MPI_COMM_WORLD, &status);
    }

    if ((id % 2) == 1)
    {
        // Resv grid information from threads with even id to odd id
        MPI_Recv(grid[num_rows-1], N, MPI_INT, (id + 1 + num_threads % num_threads), (id + 1 + num_threads % num_threads), MPI_COMM_WORLD, &status);
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads % num_threads), (id - 1 + num_threads % num_threads), MPI_COMM_WORLD, &status);

        // Sends grid information from threads with odd id to even id        
        MPI_Send(grid[1], N, MPI_INT, (id - 1 + num_threads % num_threads), id, MPI_COMM_WORLD);
        MPI_Send(grid[num_rows-2], N, MPI_INT, (id + 1 + num_threads % num_threads), id, MPI_COMM_WORLD);
    }
    
    printf("id: %d: %d, %d, %d, %d\n", id, grid[0][1],grid[1][1],grid[num_rows-2][1],grid[num_rows-1][1]);

    // testing of getting the mag and energy of the system
    mag_temp = magn(N, num_rows);
    MPI_Reduce(&mag_temp, &mag_vec[1], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    energy_temp = energy(N, num_rows);
    MPI_Reduce(&energy_temp, &energy_vec[1], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (id == 0)
    {
        printf("%f, %d\n", mag_vec[1], energy_vec[1]);
    }
    
    
    /*
    for (k = 0; k < 100; k++)
    {
        sweep(N, beta, 0, 0, seed);
        sweep(N, beta, 0, 1, seed);
        sweep(N, beta, 1, 0, seed);
        sweep(N, beta, 1, 1, seed);
    }

    for (k = 0; k < NoSweeps; k++)
    {
        sweep(N, beta, 0, 0, seed);
        sweep(N, beta, 0, 1, seed);
        sweep(N, beta, 1, 0, seed);
        sweep(N, beta, 1, 1, seed);

        mpi[k] = magn(N);
        energy_vec[k] = energy(N);
    }


    for (i = 0; i < NoSweeps; i++)
    {
        avg_mag += mag_vec[i];
        avg_energy += energy_vec[i];
    }
    avg_mag /= NoSweeps;
    avg_energy /= NoSweeps;

    for (i = 0; i < NoSweeps; i++)
    {
        std_energy += (avg_energy - energy_vec[i]) * (avg_energy - energy_vec[i]);
        std_mag += (avg_mag - mag_vec[i]) * (avg_mag - mag_vec[i]);
    }

    
    std_energy = sqrt(std_energy/(NoSweeps - 1));
    std_mag = sqrt(std_mag/(NoSweeps - 1));

    //printf("Avg E %f +- %f, avg M %f +- %f\n", sum_ener, std_ener, sum_mag, std_mag);

    output[0] = avg_energy;
    output[1] = std_energy;
    output[2] = avg_mag;
    output[3] = std_mag;

    //printf("Output %f %f %f %f\n", output[0], output[1], output[2], output[3]);

    return(output);*/
    MPI_Finalize();
}

void main(int argc, char **argv)
{
    
    /*struct timespec start, stop;
    double tot_time;
    FILE *data_file;

    int i;
    int num_N = 4;
    int *N_vec;
    

    int N = 1000;
    double beta = 0.5;

    // Allocate space for vector of values of N
    N_vec = (int *) malloc(num_N * sizeof(int));
    N_vec[0] = 1000;
    /*N_vec[1] = 1414;
    N_vec[2] = 2000;
    N_vec[3] = 2828;

    // For loop to create vector of N values
    for (i = 1; i < num_N; i++)
    {
        
        N_vec[i] = N_vec[i - 1] * sqrt(2);
        if ((N_vec[i] % 2) == 1)
        {
            // If current value is odd, we add 1 to make the new N even
            N_vec[i] += 1;
        } 

    }



    omp_set_num_threads(12);
    for (int i = 0; i < num_N; i++)
    {
        clock_gettime(CLOCK_REALTIME, &start);
        double* data = ising(N_vec[i], beta);
        clock_gettime(CLOCK_REALTIME, &stop);
        tot_time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec ) / (double)BILLION;
        data_file = fopen("test.csv", "a");
        fprintf(data_file,"%d, %f, %f, %f, %f, %f, %f\n", N_vec[i], beta, data[0], data[1], data[2], data[3], tot_time);
        fclose(data_file);
        //printf("Data %f %f %f %f %f\n", data[0], data[1], data[2], data[3], tot_time);
        printf("%d\n", N_vec[i]);
    }
    */


   
   ising(1000, 0.5);
   
}