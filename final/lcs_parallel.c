#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a, b) (((a) < (b)) ? (a) : (b))
#endif

#define MPI_MTYPE_BYTES sizeof(mtype)

#define K 4

typedef unsigned short mtype;

/* Read sequence from a file to a char vector.
 Filename is passed as parameter */

char *read_seq(char *fname) {
  // file pointer
  FILE *fseq = NULL;
  // sequence size
  long size = 0;
  // sequence pointer
  char *seq = NULL;
  // sequence index
  int i = 0;

  // open file
  fseq = fopen(fname, "rt");
  if (fseq == NULL) {
    printf("Error reading file %s\n", fname);
    exit(1);
  }

  // find out sequence size to allocate memory afterwards
  fseek(fseq, 0L, SEEK_END);
  size = ftell(fseq);
  rewind(fseq);

  // allocate memory (sequence)
  seq = (char *)calloc(size + 1, sizeof(char));
  if (seq == NULL) {
    printf("Erro allocating memory for sequence %s.\n", fname);
    exit(1);
  }

  // read sequence from file
  while (!feof(fseq)) {
    seq[i] = fgetc(fseq);
    if ((seq[i] != '\n') && (seq[i] != EOF))
      i++;
  }
  // insert string terminator
  seq[i] = '\0';

  // close file
  fclose(fseq);

  // return sequence pointer
  return seq;
}

mtype **allocateScoreMatrix(int sizeA, int sizeB) {
  int i;
  // Allocate memory for LCS score matrix
  mtype **scoreMatrix = (mtype **)malloc((sizeB + 1) * sizeof(mtype *));
  for (i = 0; i < (sizeB + 1); i++)
    scoreMatrix[i] = (mtype *)calloc(sizeA + 1, sizeof(mtype));
  return scoreMatrix;
}

void initScoreMatrix(mtype **scoreMatrix, int sizeA, int sizeB) {
  int i, j;
  // Fill first line of LCS score matrix with zeroes
  for (j = 0; j < (sizeA + 1); j++)
    scoreMatrix[0][j] = 0;

  // Do the same for the first collumn
  for (i = 1; i < (sizeB + 1); i++)
    scoreMatrix[i][0] = 0;
}

/* Arredondamento de num/den com .5 para cima, sem ponto flutuante */
static inline int round_div(int num, int den) {
  return (2 * num + den) / (2 * den);
}

/* Algoritmo 1 do artigo (linhas 11 a 13):
   BLen = BCount / procNum
   BStart = round(BLen * procRank)
   BEnd   = round(BLen * (procRank + 1)) - 1   */
static inline void block_range(int bcount, int procNum, int rank, int *BStart,
                               int *BEnd) {
  *BStart = round_div(bcount * rank, procNum);
  *BEnd = round_div(bcount * (rank + 1), procNum) - 1;
}

/* Calcula um bloco individual (K x K), percorrendo suas diagonais internas;
   os elementos de cada diagonal interna do bloco são calculados em paralelo com
   OpenMP */
static void compute_block(mtype **scoreMatrix, char *seq_row, char *seq_col,
                          int row_start, int row_end, int col_start,
                          int col_end) {
  int num_rows = row_end - row_start + 1; /* quantidade de linhas do bloco  */
  int num_cols = col_end - col_start + 1; /* quantidade de colunas do bloco */
  int total_internal_diags =
      num_rows + num_cols - 1; /* total de diagonais internas   */

  for (int diag_idx = 0; diag_idx < total_internal_diags; diag_idx++) {
    int min_row_offset = max(0, diag_idx - num_cols + 1);
    int max_row_offset = min(diag_idx, num_rows - 1);

#pragma omp parallel for schedule(static)
    for (int row_offset = min_row_offset; row_offset <= max_row_offset;
         row_offset++) {
      int row = row_start + row_offset;
      int col = col_start + (diag_idx - row_offset);
      if (seq_row[row - 1] == seq_col[col - 1])
        scoreMatrix[row][col] = scoreMatrix[row - 1][col - 1] + 1;
      else
        scoreMatrix[row][col] =
            max(scoreMatrix[row - 1][col], scoreMatrix[row][col - 1]);
    }
  }
}

int LCS_MPI(mtype **scoreMatrix, int m, int n, int k, int procNum, int procRank,
            char *A, char *B) {
  /* Algoritmo 1: linhas 6 a 8 (com teto, para tratar tamanhos não múltiplos de
   * k) */
  // int BRow = (m + k - 1) / k;
  // int BCol = (n + k - 1) / k;
  int BRows = m / k;
  int BCols = n / k;
  int BDiags = BRows + BCols - 1;

  int maxBlocks = min(BRows, BCols);
  mtype *borderBuf = malloc((size_t)maxBlocks * 2 * k * sizeof(mtype));
  int *recv_counts = malloc(procNum * sizeof(int));
  int *displacements = malloc(procNum * sizeof(int));

  for (int i = 0; i < BDiags; i++) {
    /* Blocos (block_row, block_col) com block_row + block_col = diag_idx */
    int min_block_row = max(0, i - BCols + 1);
    int max_block_row = min(i, BRows - 1);
    int blocks_in_diag = max_block_row - min_block_row + 1;

    /* Calcula a fatia de blocos de cada rank e os parâmetros para o
     * MPI_Allgatherv */
    for (int r = 0; r < procNum; r++) {
      int r_start, r_end;
      block_range(blocks_in_diag, procNum, r, &r_start, &r_end);
      int num_blocks_rank = (r_end >= r_start) ? (r_end - r_start + 1) : 0;
      recv_counts[r] = num_blocks_rank * 2 * k * (int)MPI_MTYPE_BYTES;
      displacements[r] = r_start * 2 * k * (int)MPI_MTYPE_BYTES;
    }

    int BStart, BEnd;
    block_range(blocks_in_diag, procNum, procRank, &BStart, &BEnd);

    /* Cada processo calcula sequencialmente os blocos que lhe foram atribuídos
     */
    for (int t = BStart; t <= BEnd; t++) {
      int block_row = min_block_row + t;
      int block_col = i - block_row;
      int row_start = block_row * k + 1;
      int row_end = min((block_row + 1) * k, m);
      int col_start = block_col * k + 1;
      int col_end = min((block_col + 1) * k, n);

      compute_block(scoreMatrix, A, B, row_start, row_end, col_start, col_end);

      /* Empacota a última linha e a última coluna do bloco no buffer de bordas
       */
      mtype *border_ptr = borderBuf + (size_t)t * 2 * k;
      for (int col = col_start; col <= col_end; col++)
        border_ptr[col - col_start] = scoreMatrix[row_end][col];
      for (int row = row_start; row <= row_end; row++)
        border_ptr[k + row - row_start] = scoreMatrix[row][col_end];
    }

    /* Comunicação coletiva: todos os processos trocam as bordas calculadas
     * nesta diagonal */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, borderBuf, recv_counts,
                   displacements, MPI_BYTE, MPI_COMM_WORLD);

    /* Desempacota as bordas dos blocos que foram calculados pelos outros
     * processos */
    for (int t = 0; t < blocks_in_diag; t++) {
      if (t >= BStart && t <= BEnd)
        continue;

      int block_row = min_block_row + t;
      int block_col = i - block_row;
      int row_start = block_row * k + 1;
      int row_end = min((block_row + 1) * k, m);
      int col_start = block_col * k + 1;
      int col_end = min((block_col + 1) * k, n);

      mtype *border_ptr = borderBuf + (size_t)t * 2 * k;
      for (int col = col_start; col <= col_end; col++)
        scoreMatrix[row_end][col] = border_ptr[col - col_start];
      for (int row = row_start; row <= row_end; row++)
        scoreMatrix[row][col_end] = border_ptr[k + row - row_start];
    }
  }

  int final_lcs_score = scoreMatrix[m][n];

  free(borderBuf);
  free(recv_counts);
  free(displacements);
  return final_lcs_score;
}

void printMatrix(char *seqA, char *seqB, mtype **scoreMatrix, int sizeA,
                 int sizeB) {
  int i, j;

  // print header
  printf("[DEBUG] start matrix\n");

  // print LCS score matrix allong with sequences
  printf("[DEBUG]    ");
  printf("%2c    ", ' ');

  for (j = 0; j < sizeA; j++)
    printf("%2c   ", seqA[j]);
  printf("\n");
  for (i = 0; i < sizeB + 1; i++) {
    if (i == 0)
      printf("[DEBUG]     ");
    else
      printf("[DEBUG] %c   ", seqB[i - 1]);
    for (j = 0; j < sizeA + 1; j++) {
      printf("%2d   ", scoreMatrix[i][j]);
    }
    printf("\n");
  }

  printf("[DEBUG] end matrix\n");
}

void freeScoreMatrix(mtype **scoreMatrix, int sizeB) {
  int i;
  for (i = 0; i < (sizeB + 1); i++)
    free(scoreMatrix[i]);
  free(scoreMatrix);
}

int main(int argc, char **argv) {
  // sequence pointers for both sequences
  char *seqA, *seqB;

  // sizes of both sequences and MPI process info
  int sizeA, sizeB, proc_rank, num_procs;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  // read both sequences
  seqA = read_seq("fileA.in");
  seqB = read_seq("fileB.in");

  // find out sizes
  sizeA = strlen(seqA);
  sizeB = strlen(seqB);

  // allocate LCS score matrix
  mtype **scoreMatrix = allocateScoreMatrix(sizeA, sizeB);

  // initialize LCS score matrix
  initScoreMatrix(scoreMatrix, sizeA, sizeB);

  // Synchronize all processes before measuring execution time
  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  // fill up the rest of the matrix and return final score (element located at
  // the last line and column)
  mtype score =
      LCS_MPI(scoreMatrix, sizeB, sizeA, K, num_procs, proc_rank, seqB, seqA);

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  double elapsed_time = end_time - start_time;

/* if you wish to see the entire score matrix,
 for debug purposes, define DEBUG. */
#ifdef DEBUG
  if (proc_rank == 0)
    printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

  // print score and measured execution time (only rank 0)
  if (proc_rank == 0) {
    printf("Score: %d\n", score);
    printf("Time: %.6f\n", elapsed_time);
  }

  // free score matrix
  freeScoreMatrix(scoreMatrix, sizeB);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
