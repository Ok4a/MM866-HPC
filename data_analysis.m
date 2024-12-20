clc; clear; close;
% Load data
OMP_data = readmatrix("Real stuff/OMPdata_1733390325.csv");
MPI_data = readmatrix("Real stuff/MPIdata_1733910289.csv");

% Extract runtime data from OMP matrix. 8, 7, and 5 are for the numbers
% N, numbers of core counts and number of runs respectively
runtime_OMP = reshape(OMP_data(:, 7), [8, 7, 5]);

% Matrices of averages and standard deviations of runtime for OMP
% where row i and column j is the avg/std for side length i and
% core count j
avg_runtimeOMP = mean(runtime_OMP, 3);
std_runtimeOMP = std(runtime_OMP, 0, 3);

% Extract runtime data from MPI matrix. For MPI we used 6 different core
% counts instead of 7
runtime_MPI = reshape(MPI_data(:, 7), [8, 6, 5]);

% Same as for OMP, now for MPI
avg_runtimeMPI = mean(runtime_MPI, 3);
std_runtimeMPI = std(runtime_MPI, 0, 3);

% Vector of different values of N used
N_vec = OMP_data(1: 8, 1);

% Vectors for different numbers of cores used for OMP and MPI
% respectively
Core_vecOMP = 2.^(0: 6);
Core_vecMPI = 2.^(1: 6);

% Compute strong scale speedup for OpenMP and OpenMP
OMP_speedup_strong = avg_runtimeOMP(:, 1) ./ avg_runtimeOMP(:, 1: end);
MPI_speedup_strong = avg_runtimeMPI(:, 1) ./ avg_runtimeMPI(:, 1: end);

% Strong scaling plots

% OMP
hold on
plot(Core_vecOMP, OMP_speedup_strong, "-x");
xticks(Core_vecOMP)
line([1, 64], [1, 64])
axis("tight")
title("Strong Scaling OpenMP")
legend("1024", "1472", "2048","2880", "4032",...
    "5696", "8000", "11328","Ideal Speedup", "Location", "northwest")
xlabel("Thread Count")
ylabel("Speedup")
hold off
set(gcf, "Position", [100, 100, 900, 500])
close

% MPI
hold on
plot(Core_vecMPI, MPI_speedup_strong, "-x");
xticks(Core_vecMPI)
line([1, 64], [1, 64])
axis("tight")
title("Strong Scaling OpenMPI")
legend("1024", "1472", "2048","2880", "4032",...
    "5696", "8000", "11328","Ideal Speedup", "Location", "northwest")
xlabel("Thread Count")
ylabel("Speedup")
hold off
set(gcf, "Position", [100, 100, 900, 500])
close

% Extract diagonal entries of avg_runtime as the workload
workloadOMP = diag(avg_runtimeOMP);
workloadMPI = diag(avg_runtimeMPI);

% Compute speed up in relation to workload
OMP_speedup_weak = workloadOMP(1) ./ workloadOMP;
MPI_speedup_weak = (workloadMPI(1)) ./ workloadMPI;

% Weak scaling plots

% OMP and MPI
hold on
plot(Core_vecOMP, OMP_speedup_weak, "-x");
plot(Core_vecMPI, MPI_speedup_weak, "--o");
xticks(Core_vecOMP)
line([1, 64], [1, 1])
legend("Efficiency OpenMP", "Efficiency OpenMPI", "Ideal Efficiency")
axis("tight")
title("Weak Scaling OpenMP and OpenMPI")
xlabel("Thread Count")
ylabel("Speedup")
hold off
set(gcf, "Position", [100, 100, 900, 500])
close

% Errorbar plots

% OMP
errorbar((N_vec), (avg_runtimeOMP), (std_runtimeOMP))
title("Average runtime with errorbars OpenMP")
%axis("tight")
xticks((N_vec))
legend("1", "2", "4", "8", "16", "32", "64", "Location","northwest")
xlabel("Grid side length")
ylabel("Runtime")
set(gcf, "Position", [100, 100, 900, 500])
close

% MPI
errorbar(N_vec, (avg_runtimeMPI), (std_runtimeMPI))
title("Average runtime with errorbars OpenMPI")
%axis("tight")
xticks(N_vec)
legend("2", "4", "8", "16", "32", "64", "Location","northwest")
xlabel("Grid side length")
ylabel("Runtime")
set(gcf, "Position", [100, 100, 900, 500])
close