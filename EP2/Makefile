OUTPUT=mandelbrot

IMAGE=.ppm

CC=gcc
CC_OPT=-std=c11
CC_MPI=mpicc

CC_OMP=-fopenmp
CC_PTH=-pthread

.PHONY: all
all: $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_seq_sem $(OUTPUT)_ompi

$(OUTPUT)_omp: $(OUTPUT)_omp.c
	$(CC) -o $(OUTPUT)_omp $(CC_OPT) $(CC_OMP) $(OUTPUT)_omp.c

$(OUTPUT)_pth: $(OUTPUT)_pth.c
	$(CC) -o $(OUTPUT)_pth $(CC_OPT) $(CC_PTH) $(OUTPUT)_pth.c

$(OUTPUT)_seq: $(OUTPUT)_seq.c
	$(CC) -o $(OUTPUT)_seq $(CC_OPT) $(OUTPUT)_seq.c

$(OUTPUT)_seq_sem: $(OUTPUT)_seq_sem.c
	$(CC) -o $(OUTPUT)_seq_sem $(CC_OPT) $(OUTPUT)_seq_sem.c

$(OUTPUT)_ompi: $(OUTPUT)_ompi.c
	$(CC_MPI) -o $(OUTPUT)_ompi $(CC_OPT) $(OUTPUT)_ompi.c

.PHONY: clean
clean:
	rm $(OUTPUT)_omp $(OUTPUT)_pth $(OUTPUT)_seq $(OUTPUT)_seq_sem *$(IMAGE)
