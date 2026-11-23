CC=mpicc
CFLAGS=-fopenmp -O3 -Wall -lm

all:
	$(CC) $(CFLAGS) lcs_parallel.c -o lcs_parallel

debug:
	$(CC) $(CFLAGS) -DDEBUG lcs_parallel.c -o lcs_parallel

clean:
	rm -f lcs_parallel