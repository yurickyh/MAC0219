#! /bin/bash

set -o xtrace

MEASUREMENTS=10
ITERATIONS=4
INITIAL_SIZE=16
THREADS_ITERATIONS=6
INITIAL_THREADS=1

THREADS=$INITIAL_THREADS
SIZE=$INITIAL_SIZE

NAMES1=('mandelbrot_seq' 'mandelbrot_seq_sem')
NAMES2=('mandelbrot_pth' 'mandelbrot_omp')

make
mkdir results

for NAME in ${NAMES1[@]}; do
    mkdir results/$NAME

    for ((i=1; i<=$ITERATIONS; i++)); do
            perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE >> full.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE >> seahorse.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE >> elephant.log 2>&1
            perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE >> triple_spiral.log 2>&1
            SIZE=$(($SIZE * 2))
    done

    SIZE=$INITIAL_SIZE

    mv *.log results/$NAME
    rm output.ppm
done

for NAME in ${NAMES2[@]}; do
    mkdir results/$NAME

    for ((i=1; i<=$ITERATIONS; i++)); do
            for ((j=1; j<=THREADS_ITERATIONS; j++)) do
                    perf stat -r $MEASUREMENTS ./$NAME -2.5 1.5 -2.0 2.0 $SIZE $THREADS >> full.log 2>&1
                    perf stat -r $MEASUREMENTS ./$NAME -0.8 -0.7 0.05 0.15 $SIZE $THREADS >> seahorse.log 2>&1
                    perf stat -r $MEASUREMENTS ./$NAME 0.175 0.375 -0.1 0.1 $SIZE $THREADS >> elephant.log 2>&1
                    perf stat -r $MEASUREMENTS ./$NAME -0.188 -0.012 0.554 0.754 $SIZE $THREADS >> triple_spiral.log 2>&1
                    THREADS=$(($THREADS * 2))
            done
            THREADS=$INITIAL_THREADS
            SIZE=$(($SIZE * 2))
    done

    SIZE=$INITIAL_SIZE

    mv *.log results/$NAME
    rm output.ppm
done
