CC = mpicc
CFLAGS = -fopenmp -O3 -Wall -lm

ifeq ($(shell uname -s),Darwin)
    ifneq ($(wildcard /opt/homebrew/bin/gcc-16),)
        export OMPI_CC = /opt/homebrew/bin/gcc-16
    endif
endif

all:
	$(CC) $(CFLAGS) lcs_parallel.c -o lcs_parallel
	gcc -O3 lcs.c -o lcs_seq

debug:
	$(CC) $(CFLAGS) -DDEBUG lcs_parallel.c -o lcs_parallel

clean:
	rm -f lcs_parallel lcs_seq
