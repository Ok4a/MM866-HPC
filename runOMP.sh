#!/usr/bin/bash
gcc isingOMP.c -lm -fopenmp -o OMPising

./OMPising

