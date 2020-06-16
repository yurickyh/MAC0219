#! /bin/bash

set -o xtrace

MEASUREMENTS=15
THREADS_ITERATIONS=6
INITIAL_THREADS=1

THREADS=$INITIAL_THREADS
SIZE=4096

NAMES1=('mandelbrot_seq' 'mandelbrot_seq_sem')
NAMES2=('mandelbrot_pth' 'mandelbrot_omp')

make
mkdir results

for NAME in ${NAMES1[@]}; do
    perf stat -r $MEASUREMENTS -n ./$NAME -0.188 -0.012 0.554 0.754 $SIZE >> $NAME.log 2>&1
    mv *.log results
    rm output.ppm
done

for NAME in ${NAMES2[@]}; do

    for ((j=1; j<=THREADS_ITERATIONS; j++)) do
        perf stat -r $MEASUREMENTS -n ./$NAME -0.188 -0.012 0.554 0.754 $SIZE $THREADS >> $NAME.log 2>&1
        THREADS=$(($THREADS * 2))
    done
    THREADS=$INITIAL_THREADS
    mv *.log results
done
