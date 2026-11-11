CC = /opt/homebrew/bin/gcc-16
MPICC = OMPI_CC=$(CC) /opt/homebrew/bin/mpicc
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp

all: lcs_seq lcs_parallel

lcs_seq: lcs.c
	$(CC) $(CFLAGS) -DBENCHMARK lcs.c -o lcs_seq

lcs_parallel: lcs_parallel.c
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -DBENCHMARK lcs_parallel.c -o lcs_parallel

debug:
	$(CC) $(CFLAGS) -DDEBUGMATRIX lcs.c -o lcs_seq_debug
	$(MPICC) $(CFLAGS) $(OMPFLAGS) -DDEBUGMATRIX lcs_parallel.c -o lcs_parallel_debug

clean:
	rm -f lcs_seq lcs_parallel lcs_seq_debug lcs_parallel_debug *.o

