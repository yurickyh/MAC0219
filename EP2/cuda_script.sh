#! /bin/bash

set -o xtrace

MEASUREMENTS=15
SIZE=4096
GRID=(1 8 16 32 64)

make
if [ ! -d "results" ]; then
    mkdir results
fi

for G in ${GRID[@]}; do
    perf stat -r $MEASUREMENTS -n ./mandelbrot_cuda -0.188 -0.012 0.554 0.754 $SIZE $G $G >> mandelbrot_cuda.log 2>&1
done
mv *.log results

