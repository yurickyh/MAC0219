#! /bin/bash

set -o xtrace

MEASUREMENTS=15
SIZE=4096
PROCESSES=(1 8 16 32 64)
INITIAL_THREADS=1
THREADS=$INITIAL_THREADS
THREADS_ITERATIONS=6

make
if [ ! -d "results" ]; then
    mkdir results
fi

for PROCESS in ${PROCESSES[@]}; do
    perf stat -r $MEASUREMENTS -n mpirun -H localhost:$PROCESS ./mandelbrot_ompi -0.188 -0.012 0.554 0.754 $SIZE >> mandelbrot_ompi.log 2>&1
done
mv *.log results

for PROCESS in ${PROCESSES[@]}; do
    for ((i=1; i<=THREADS_ITERATIONS; i++)) do
        perf stat -r $MEASUREMENTS -n mpirun -H localhost:$PROCESS ./mandelbrot_mpi_ompi -0.188 -0.012 0.554 0.754 $SIZE $THREADS >> mandelbrot_mpi_ompi.log 2>&1
        THREADS=$(($THREADS * 2))
    done
    THREADS=$INITIAL_THREADS    
done
mv *.log results
