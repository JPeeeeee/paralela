#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

typedef unsigned short mtype;

/* Read sequence from a file to a char vector.
   Filename is passed as parameter */
char* read_seq(char *fname) {
    FILE *fseq = NULL;
    long size = 0;
    char *seq = NULL;
    int i = 0;

    fseq = fopen(fname, "rt");
    if (fseq == NULL ) {
        printf("Error reading file %s\n", fname);
        exit(1);
    }

    fseek(fseq, 0L, SEEK_END);
    size = ftell(fseq);
    rewind(fseq);

    seq = (char *) calloc(size + 1, sizeof(char));
    if (seq == NULL ) {
        printf("Error allocating memory for sequence %s.\n", fname);
        exit(1);
    }

    while (!feof(fseq)) {
        seq[i] = fgetc(fseq);
        if ((seq[i] != '\n') && (seq[i] != EOF))
            i++;
    }
    seq[i] = '\0';

    fclose(fseq);
    return seq;
}

/* Allocate contiguous 2D score matrix */
mtype ** allocateContiguousMatrix(int rows, int cols) {
    mtype ** matrix = (mtype **) malloc(rows * sizeof(mtype *));
    mtype * data = (mtype *) calloc((size_t)rows * cols, sizeof(mtype));
    if (!matrix || !data) {
        fprintf(stderr, "Error allocating memory for matrix (%d x %d)\n", rows, cols);
        exit(1);
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = &data[(size_t)i * cols];
    }
    return matrix;
}

void freeContiguousMatrix(mtype ** matrix) {
    if (matrix) {
        if (matrix[0]) free(matrix[0]);
        free(matrix);
    }
}

/* Debug print matrix aligned with sequences */
void printMatrix(char * seqA, char * seqB, mtype ** scoreMatrix, int sizeA, int sizeB) {
    int i, j;
    printf("Score Matrix:\n");
    printf("========================================\n");
    printf("        ");
    for (j = 0; j < sizeA; j++)
        printf("%5c   ", seqA[j]);
    printf("\n");
    for (i = 0; i < sizeB + 1; i++) {
        if (i == 0)
            printf("    ");
        else
            printf("%c   ", seqB[i - 1]);
        for (j = 0; j < sizeA + 1; j++) {
            printf("%5d   ", scoreMatrix[i][j]);
        }
        printf("\n");
    }
    printf("========================================\n");
}

/* Single-rank Tiled Wavefront using OpenMP */
mtype LCS_OpenMP(mtype ** scoreMatrix, int sizeA, int sizeB, char * seqA, char * seqB) {
    int tile_size = 128;
    if (sizeA < 128 || sizeB < 128) tile_size = 16;
    if (sizeA < 16 || sizeB < 16) tile_size = 4;

    int TI = (sizeB + tile_size - 1) / tile_size;
    int TJ = (sizeA + tile_size - 1) / tile_size;

    for (int diag = 0; diag < TI + TJ - 1; diag++) {
        #pragma omp parallel for schedule(dynamic)
        for (int bi = 0; bi < TI; bi++) {
            int bj = diag - bi;
            if (bj >= 0 && bj < TJ) {
                int i_start = bi * tile_size + 1;
                int i_end = min((bi + 1) * tile_size, sizeB);
                int j_start = bj * tile_size + 1;
                int j_end = min((bj + 1) * tile_size, sizeA);

                for (int i = i_start; i <= i_end; i++) {
                    char bi_char = seqB[i - 1];
                    for (int j = j_start; j <= j_end; j++) {
                        if (seqA[j - 1] == bi_char) {
                            scoreMatrix[i][j] = scoreMatrix[i - 1][j - 1] + 1;
                        } else {
                            mtype up = scoreMatrix[i - 1][j];
                            mtype left = scoreMatrix[i][j - 1];
                            scoreMatrix[i][j] = max(up, left);
                        }
                    }
                }
            }
        }
    }
    return scoreMatrix[sizeB][sizeA];
}

/* Hybrid MPI + OpenMP Pipelined Wavefront */
mtype LCS_Hybrid_MPI(int rank, int nprocs, int sizeA, int sizeB, char * seqA, char * seqB, mtype *** outFullMatrix) {
    // 1D Column block distribution across MPI ranks
    int base_cols = sizeA / nprocs;
    int rem = sizeA % nprocs;
    int start_col, end_col;

    if (rank < rem) {
        start_col = rank * (base_cols + 1) + 1;
        end_col = start_col + base_cols;
    } else {
        start_col = rem * (base_cols + 1) + (rank - rem) * base_cols + 1;
        end_col = start_col + base_cols - 1;
    }
    int num_local_cols = end_col - start_col + 1;

    // Allocate local submatrix: (sizeB + 1) rows, (num_local_cols + 1) columns
    // local col 0 holds the boundary values received from rank - 1 (or 0 for rank 0)
    mtype ** localMatrix = allocateContiguousMatrix(sizeB + 1, num_local_cols + 1);

    // Row block size for pipelining
    int block_size = 256;
    if (sizeB < 256) block_size = 32;
    if (sizeB < 32) block_size = 4;
    if (sizeB < 4) block_size = 1;

    int num_rblocks = (sizeB + block_size - 1) / block_size;

    mtype * recv_buf = (mtype *) malloc(block_size * sizeof(mtype));
    mtype * send_buf = (mtype *) malloc(block_size * sizeof(mtype));

    for (int rb = 0; rb < num_rblocks; rb++) {
        int r_start = rb * block_size + 1;
        int r_end = min((rb + 1) * block_size, sizeB);
        int r_len = r_end - r_start + 1;

        // Receive left boundary from rank - 1
        if (rank > 0) {
            MPI_Recv(recv_buf, r_len, MPI_UNSIGNED_SHORT, rank - 1, rb, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int r = 0; r < r_len; r++) {
                localMatrix[r_start + r][0] = recv_buf[r];
            }
        }

        // Compute local block using OpenMP across internal sub-tiles or rows
        int tile_size = 64;
        if (num_local_cols < 64 || r_len < 64) tile_size = 16;
        if (num_local_cols < 16 || r_len < 16) tile_size = 4;

        int TI = (r_len + tile_size - 1) / tile_size;
        int TJ = (num_local_cols + tile_size - 1) / tile_size;

        for (int diag = 0; diag < TI + TJ - 1; diag++) {
            #pragma omp parallel for schedule(dynamic)
            for (int bi = 0; bi < TI; bi++) {
                int bj = diag - bi;
                if (bj >= 0 && bj < TJ) {
                    int i_start = r_start + bi * tile_size;
                    int i_end = min(r_start + (bi + 1) * tile_size - 1, r_end);
                    int j_start = bj * tile_size + 1;
                    int j_end = min((bj + 1) * tile_size, num_local_cols);

                    for (int i = i_start; i <= i_end; i++) {
                        char bi_char = seqB[i - 1];
                        for (int j = j_start; j <= j_end; j++) {
                            int gj = start_col + j - 1;
                            if (seqA[gj - 1] == bi_char) {
                                localMatrix[i][j] = localMatrix[i - 1][j - 1] + 1;
                            } else {
                                mtype up = localMatrix[i - 1][j];
                                mtype left = localMatrix[i][j - 1];
                                localMatrix[i][j] = max(up, left);
                            }
                        }
                    }
                }
            }
        }

        // Send right boundary to rank + 1
        if (rank < nprocs - 1) {
            for (int r = 0; r < r_len; r++) {
                send_buf[r] = localMatrix[r_start + r][num_local_cols];
            }
            MPI_Send(send_buf, r_len, MPI_UNSIGNED_SHORT, rank + 1, rb, MPI_COMM_WORLD);
        }
    }

    free(recv_buf);
    free(send_buf);

    mtype final_score = 0;
    if (rank == nprocs - 1) {
        final_score = localMatrix[sizeB][num_local_cols];
    }

    // Broadcast final score to all ranks (including rank 0)
    MPI_Bcast(&final_score, 1, MPI_UNSIGNED_SHORT, nprocs - 1, MPI_COMM_WORLD);

#ifdef DEBUGMATRIX
    // Gather full matrix at rank 0 for verification / debug printing
    mtype ** fullMatrix = NULL;
    if (rank == 0) {
        fullMatrix = allocateContiguousMatrix(sizeB + 1, sizeA + 1);
        // Copy rank 0 columns
        for (int i = 0; i <= sizeB; i++) {
            for (int j = 1; j <= num_local_cols; j++) {
                fullMatrix[i][j] = localMatrix[i][j];
            }
        }
        // Receive from other ranks
        for (int p = 1; p < nprocs; p++) {
            int p_start_col, p_end_col;
            if (p < rem) {
                p_start_col = p * (base_cols + 1) + 1;
                p_end_col = p_start_col + base_cols;
            } else {
                p_start_col = rem * (base_cols + 1) + (p - rem) * base_cols + 1;
                p_end_col = p_start_col + base_cols - 1;
            }
            int p_ncols = p_end_col - p_start_col + 1;
            mtype * p_col_buf = (mtype *) malloc((sizeB + 1) * sizeof(mtype));
            for (int j = 1; j <= p_ncols; j++) {
                MPI_Recv(p_col_buf, sizeB + 1, MPI_UNSIGNED_SHORT, p, j, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                for (int i = 0; i <= sizeB; i++) {
                    fullMatrix[i][p_start_col + j - 1] = p_col_buf[i];
                }
            }
            free(p_col_buf);
        }
        *outFullMatrix = fullMatrix;
    } else {
        // Send local columns to rank 0
        mtype * p_col_buf = (mtype *) malloc((sizeB + 1) * sizeof(mtype));
        for (int j = 1; j <= num_local_cols; j++) {
            for (int i = 0; i <= sizeB; i++) {
                p_col_buf[i] = localMatrix[i][j];
            }
            MPI_Send(p_col_buf, sizeB + 1, MPI_UNSIGNED_SHORT, 0, j, MPI_COMM_WORLD);
        }
        free(p_col_buf);
    }
#endif

    freeContiguousMatrix(localMatrix);
    return final_score;
}

int main(int argc, char ** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char *seqA = NULL, *seqB = NULL;
    int sizeA = 0, sizeB = 0;

    double t_start_total = MPI_Wtime();
    double t_start_seq = MPI_Wtime();

    // Rank 0 reads the sequences and broadcasts sizes and data
    if (rank == 0) {
        seqA = read_seq("fileA.in");
        seqB = read_seq("fileB.in");
        sizeA = strlen(seqA);
        sizeB = strlen(seqB);
    }

    MPI_Bcast(&sizeA, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeB, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        seqA = (char *) malloc((sizeA + 1) * sizeof(char));
        seqB = (char *) malloc((sizeB + 1) * sizeof(char));
    }

    MPI_Bcast(seqA, sizeA + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(seqB, sizeB + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    double t_end_seq = MPI_Wtime();
    double t_seq_io = t_end_seq - t_start_seq;

    mtype score = 0;
    mtype ** fullMatrix = NULL;

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start_comp = MPI_Wtime();

    if (nprocs == 1) {
        // Pure OpenMP path on rank 0
        fullMatrix = allocateContiguousMatrix(sizeB + 1, sizeA + 1);
        score = LCS_OpenMP(fullMatrix, sizeA, sizeB, seqA, seqB);
#ifndef DEBUGMATRIX
        freeContiguousMatrix(fullMatrix);
        fullMatrix = NULL;
#endif
    } else {
        // Hybrid MPI + OpenMP path
        score = LCS_Hybrid_MPI(rank, nprocs, sizeA, sizeB, seqA, seqB, &fullMatrix);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end_comp = MPI_Wtime();
    double t_comp = t_end_comp - t_start_comp;
    double t_total = t_end_comp - t_start_total;

#ifdef DEBUGMATRIX
    if (rank == 0 && fullMatrix != NULL) {
        printMatrix(seqA, seqB, fullMatrix, sizeA, sizeB);
        freeContiguousMatrix(fullMatrix);
    }
#endif

    if (rank == 0) {
        printf("\nScore: %d\n", score);
#ifdef BENCHMARK
        printf("TIME_TOTAL: %lf\n", t_total);
        printf("TIME_SEQ: %lf\n", t_seq_io);
        printf("TIME_COMP: %lf\n", t_comp);
#endif
    }

    free(seqA);
    free(seqB);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

