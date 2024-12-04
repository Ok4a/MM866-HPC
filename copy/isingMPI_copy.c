# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <mpi.h>
# include <time.h>
# include <string.h>

# define NoSweeps 1000
# define BILLION  1000000000L;
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
    for (i = 1; i < num_rows-1; i++)
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
    for (i = 1; i < num_rows-1; i++)
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
void sweep(int N, int num_rows, double beta, unsigned int seed, int id, int num_threads)
    {

    MPI_Status status1;
    MPI_Request req1;
    // Indices for grid position
    int i;
    int j;

    // Variable diff
    int diff;

    // Initialize variables p and rnd for the simulation
    double p = 0.0;
    double rnd = 0.0;
    
    // Parallelize double for loop over the grid
    for (i = 1; i < num_rows - 1; i++)
    {
        if (i = 2)
        {
            /* The following line sends the second row of the grid in thread with current id to
            the thread with id - 1. It does after having modified the second row by flipping
            spins*/
            MPI_Isend(grid[1], N, MPI_INT, (id - 1 + num_threads) % num_threads, id+100, MPI_COMM_WORLD, &req1);
        }

        if (i = num_rows - 2)
        {
            /* Thread*with current id receives the final row of the grid in the thread with id + 1*/
            MPI_Recv(grid[num_rows - 1], N, MPI_INT, (id + 1 + num_threads) % num_threads, (id + 1 + num_threads) % num_threads + 100, MPI_COMM_WORLD, &status1);
        }
        
        /* A for loop where we go through the grids and compute the energy
        of the grid with the flipped spins*/
        for (j = 0; j < N; j++)
        {
            /* Computes the energy difference if we were to flip the current
            spin*/
            diff = energy_diff(N, i, j);
            if (diff <= 0)
            {
                /* If energy difference is negative we flip the spin*/
                grid[i][j] = -grid[i][j];
            }
            else
            {
                /* Otherwise we flip the spin with probability p = exp(-beta * diff)*/
                p = exp(-beta * diff);
                rnd = rand_r(&seed) / ((double) RAND_MAX);
                if (rnd < p)
                {
                    grid[i][j] = -grid[i][j];
                }
            }
        }
    }

    /* The following if-statements transfer the information of the updated grids first from 
    threads with even id's to threads with odd id's and then from odd to even*/
    if ((id % 2) == 0)
    {
        // Sends grid information from threads with even id to threads with id+-1  
        MPI_Send(grid[num_rows - 2], N, MPI_INT, (id + 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
        
        // Resv grid information from threads with odd id to threads with id+-1  
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads) % num_threads, (id - 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status1);
    }

    if ((id % 2) == 1)
    {
        // Resv grid information from threads with even id to threads with id+-1  
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads) % num_threads, (id - 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status1);

        // Sends grid information from threads with odd id to threads with id+-1       
        MPI_Send(grid[num_rows - 2], N, MPI_INT, (id + 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
    }


}
// Function that simulates the ISING model
double* ising(int N, double beta, int id, int num_threads)
{    

    

    MPI_Status status;
    MPI_Request req;
    
    /* Each thread has a grid of dimension num_rows by N.
    num_rows is computed as N / num_threads + 2*/
    int num_rows = N / num_threads + 2;

    /* Indices for allocating space for the grids and for
    initializing the grid*/
    int i = 0;
    int j = 0;


    /* Random int r for 0 or 1 used for initializing the grid and seed*/
    int r = 0;
    unsigned int seed;

    /* Vectors for storing the energy and magnetization at each iteration of the simulation
    as well as a vector for output the avg and std of the energy and magnetization of the 
    simulation*/
    double *mag_vec;
    int *energy_vec;
    double *output;

    /* We only let thread 0 store the output data, so we create the vectors for that thread only*/
    if (id == 0)
    {
        /* Allocating space for the previously created*/
        output = (double*) malloc(4 * sizeof(double));
        mag_vec = (double *) malloc(NoSweeps * sizeof(double));
        energy_vec = (int *) malloc(NoSweeps * sizeof(double));
    }
    /* Here we create variables for storing the average magnetization and energy*/
    double avg_mag = 0.0;
    double avg_energy = 0.0;

    /* Here we create variables for storing the standard deviation of the energy and
    magnetization*/
    double std_energy = 0.0;
    double std_mag = 0.0;
    
    /* Variables mag_temp and energy_temp that each thread will use to store
    the result of its energy and magnetization computations before sending them to
    thread 0*/
    double mag_temp = 0.0;
    int energy_temp = 0.0;
    
    /* Initializes the grid for each grid to be a num_rows by N matrix*/
    grid = (int **) malloc(num_rows * sizeof(int *));
    for (i = 0; i < num_rows; i++)
    {
        grid[i] = (int *) malloc(N * sizeof(int));
    }

    /* Each thread gets a unique seed based on their id*/
    seed = 42 + id;

    /* Each thread then initializes the second to second-to-last rows of its grid.
    These rows correspond to the overall grid while the first and final rows will be
    filled by other threads for when the computations are made. This is why the first
    for loop over rows is from row i = 1 instead of 0 and to row num_rows - 2 instead of
    num_rows - 1*/
    for (i = 1; i < num_rows - 1; i++)
    {
        for (j = 0; j < N; j++)
        {
            /* Creates a random int 0 or 1*/
            r = rand_r(&seed) % 2;
            if (r)
            {
                /* Sets spin at grid entry [i][j] to be +1 if r == 1*/
                grid[i][j] = 1;
            }
            else
            {
                /* Sets spin at grid entry [i][j] to be -1 if r == 0*/
                grid[i][j] = -1;
            }
        }
    }    
    
    /* The following two if-statements has the threads send the necessary information 
    of their grids to each other. Thread with current id sends its second row to the thread with id-1
    and its second to last row to the thread with id+1. It then receives the information of the
    second row of the grid in the thread with id+1 as its final row and the information
    of the second to last row of the grid in the thread with id-1 as its first row. The threads with 
    even numbered id's are the first to send and receive and then the threads with odd numbered 
    id's receive and send*/
    if ((id % 2) == 0)
    {
        // Sends grid information from threads with even id to threads with id+-1  
        MPI_Send(grid[1], N, MPI_INT, (id - 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
        MPI_Send(grid[num_rows - 2], N, MPI_INT, (id + 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
        
        // Resv grid information from threads with odd id to threads with id+-1  
        MPI_Recv(grid[num_rows - 1], N, MPI_INT, (id + 1 + num_threads) % num_threads, (id + 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status);
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads) % num_threads, (id - 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status);
    }
    if ((id % 2) == 1)
    {
        // Resv grid information from threads with even id to threads with id+-1  
        MPI_Recv(grid[num_rows - 1], N, MPI_INT, (id + 1 + num_threads) % num_threads, (id + 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status);
        MPI_Recv(grid[0], N, MPI_INT, (id - 1 + num_threads) % num_threads, (id - 1 + num_threads) % num_threads, MPI_COMM_WORLD, &status);

        // Sends grid information from threads with odd id to threads with id+-1       
        MPI_Send(grid[1], N, MPI_INT, (id - 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
        MPI_Send(grid[num_rows - 2], N, MPI_INT, (id + 1 + num_threads) % num_threads, id, MPI_COMM_WORLD);
    } 
    /* The send and receive orders are in opposite orders in the two if-statements since we otherwise
    have a deadlock in the code*/
    
    /* We then sweep the grid and perform the ISING simulation for 100 iterations.
    This corresponds to the burn-in period*/
    for (i = 0; i < 100; i++)
    {
        /* Performs the simulation of the ISING model*/
        sweep(N, num_rows, beta, seed, id, num_threads);
    }

    /* After the burn-in period we perform the simulation for NoSweeps iterations*/
    for (i = 0; i < NoSweeps; i++)
    {
        /* We sweep through the grid like in the previous for loop*/
        sweep(N, num_rows, beta, seed, id, num_threads);

        /* Each thread computes the magnetization for its own sub grid and sends
        the results to thread 0 to be added together in the variable mag_vec[i]*/
        mag_temp = magn(N, num_rows);
        MPI_Reduce(&mag_temp, &mag_vec[i], 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        /* Same for energy, each thread sends the information to be added in energy_vec[i] 
        by thread 0.*/
        energy_temp = energy(N, num_rows);
        MPI_Reduce(&energy_temp, &energy_vec[i], 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    /* After all the simulations thread 0 computes the average energy and magnetization
    for the simulation*/
    if (id == 0)
    {
        for (i = 0; i < NoSweeps; i++)
        {
            /* Thread 0 sums over the vectors mag_vec and energy_vec*/
            avg_mag += mag_vec[i];
            avg_energy += energy_vec[i];
            
        }
        /* Computes average energy and magnetization over the simulation by dividing
        the sums by the number of iterations NoSweeps*/
        avg_mag /= NoSweeps;
        avg_energy /= NoSweeps;

        /* We compute the standard deviations by summing over the square differences of
        energies and magnetizations from the averages*/
        for (i = 0; i < NoSweeps; i++)
        {
            std_energy += (avg_energy - energy_vec[i]) * (avg_energy - energy_vec[i]);
            std_mag += (avg_mag - mag_vec[i]) * (avg_mag - mag_vec[i]);
        }

        /* Afterwards we divide by NoSweeps - 1 to get the sample variance and take the squareroot
        of the sample variance to get the estimated standard deviation*/
        std_energy = sqrt(std_energy/(NoSweeps - 1));
        std_mag = sqrt(std_mag/(NoSweeps - 1));

        /* We then store the values avg_energy, std_energy, avg_mag and std_mag in the vector output*/
        output[0] = avg_energy;
        output[1] = std_energy;
        output[2] = avg_mag;
        output[3] = std_mag;
        
    }
    // The function then returns the output
    return(output);
   
}

void main(int argc, char **argv)
{

    MPI_Init(NULL, NULL);
    int num_threads = 0;
    int id = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    struct timespec start, stop;
    double tot_time;
    FILE *data_file;

    char file_path[26] = "MPIdata_";
    char csv[] = ".csv";
   
    if (id == 0)
    {
        //sprintf(time_str, "%ld", (unsigned long)time(NULL));
        strcat(file_path, argv[1]);
        strcat(file_path, csv);
        /*data_file = fopen(file_path, "w");
        fprintf(data_file,"%s, %s, %s, %s, %s, %s, %s, %s\n", "N", "beta", "avg_energy", "std_energy", "avg_mag", "std_mag", "tot_time", "num_threads");
        fclose(data_file);*/
        
    }

    int i;


    double* data;
    data = (double*) malloc(4 * sizeof(double));

    double beta = 0.5;

    // Allocate space for vector of values of N
    int num_N = strtod(argv[3], NULL);
    int N_vec[num_N];
    N_vec[0] = strtod(argv[2], NULL);

    // For loop to create vector of N values
    for (i = 1; i < num_N; i++)
    {
        N_vec[i] = N_vec[i - 1] * sqrt(2);
        N_vec[i] += (N_vec[i] % 2);       
    } 

    for (i = 0; i < num_N; i++)
    {
        N_vec[i] += 64 - (N_vec[i] % 64);
    }


    for (int i = 0; i < num_N; i++)
    {
        if (id == 0)
        {
            clock_gettime(CLOCK_REALTIME, &start);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        data = ising(N_vec[i] , beta, id, num_threads);
        if (id == 0)
        {
            clock_gettime(CLOCK_REALTIME, &stop);
            tot_time = (stop.tv_sec - start.tv_sec) + (double)(stop.tv_nsec - start.tv_nsec ) / (double)BILLION;
            data_file = fopen(file_path, "a");
            fprintf(data_file,"%d, %f, %f, %f, %f, %f, %f, %d\n", N_vec[i], beta, data[0], data[1], data[2], data[3], tot_time, num_threads);
            fclose(data_file);
        }
    }
    MPI_Finalize();
}