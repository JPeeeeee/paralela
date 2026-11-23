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

/* Calcula um bloco, percorrendo suas diagonais; os elementos de cada
   diagonal do bloco são calculados em paralelo com OpenMP */
static void compute_block(mtype **scoreMatrix, char *A, char *B, int r0, int r1,
                          int c0, int c1) {
  int line = r1 - r0 + 1;    /* linhas do bloco  */
  int col = c1 - c0 + 1;     /* colunas do bloco */
  int diag = line + col - 1; /* diagonais do bloco */

  for (int d = 0; d < diag; d++) {
    int ilo = max(0, d - col + 1);
    int ihi = min(d, line - 1);

#pragma omp parallel for schedule(static)
    for (int ii = ilo; ii <= ihi; ii++) {
      int row = r0 + ii;
      int col = c0 + (d - ii);
      if (A[row - 1] == B[col - 1])
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
  int BRow = m / k;
  int BCol = n / k;
  int BDia = BRow + BCol - 1;

  int maxBlocks = min(BRow, BCol);
  mtype *buf = malloc((size_t)maxBlocks * 2 * k * sizeof(mtype));
  int *counts = malloc(procNum * sizeof(int));
  int *displs = malloc(procNum * sizeof(int));

  for (int i = 0; i < BDia; i++) {
    /* blocos (bi, bj) com bi + bj = i; BCount = Block_Count(i) */
    int lo = max(0, i - BCol + 1);
    int hi = min(i, BRow - 1);
    int BCount = hi - lo + 1;

    /* intervalo de blocos de cada rank (necessário ao Allgatherv) */
    for (int r = 0; r < procNum; r++) {
      int s, e;
      block_range(BCount, procNum, r, &s, &e);
      int cnt = (e >= s) ? (e - s + 1) : 0;
      counts[r] = cnt * 2 * k * (int)MPI_MTYPE_BYTES;
      displs[r] = s * 2 * k * (int)MPI_MTYPE_BYTES;
    }
    int BStart, BEnd;
    block_range(BCount, procNum, procRank, &BStart, &BEnd);

    /* cada processo executa seus blocos sequencialmente */
    for (int t = BStart; t <= BEnd; t++) {
      int bi = lo + t;
      int bj = i - bi;
      int r0 = bi * k + 1, r1 = min((bi + 1) * k, m);
      int c0 = bj * k + 1, c1 = min((bj + 1) * k, n);

      compute_block(scoreMatrix, A, B, r0, r1, c0, c1);

      /* empacota última linha e última coluna do bloco */
      mtype *p = buf + (size_t)t * 2 * k;
      for (int col = c0; col <= c1; col++)
        p[col - c0] = scoreMatrix[r1][col];
      for (int row = r0; row <= r1; row++)
        p[k + row - r0] = scoreMatrix[row][c1];
    }

    /* todos os processos recebem as bordas dos blocos da diagonal */
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, buf, counts, displs,
                   MPI_BYTE, MPI_COMM_WORLD);

    /* desempacota as bordas dos blocos calculados por outros processos */
    for (int t = 0; t < BCount; t++) {
      if (t >= BStart && t <= BEnd)
        continue;
      int bi = lo + t;
      int bj = i - bi;
      int r0 = bi * k + 1, r1 = min((bi + 1) * k, m);
      int c0 = bj * k + 1, c1 = min((bj + 1) * k, n);
      mtype *p = buf + (size_t)t * 2 * k;
      for (int col = c0; col <= c1; col++)
        scoreMatrix[r1][col] = p[col - c0];
      for (int row = r0; row <= r1; row++)
        scoreMatrix[row][c1] = p[k + row - r0];
    }
  }

  int result = scoreMatrix[m][n];

  free(buf);
  free(counts);
  free(displs);
  return result;
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

  // sizes of both sequences
  int sizeA, sizeB, rank, size;

  //
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

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

  // fill up the rest of the matrix and return final score (element locate at
  // the last line and collumn)
  mtype score = LCS_MPI(scoreMatrix, sizeB, sizeA, K, size, rank, seqB, seqA);

/* if you wish to see the entire score matrix,
 for debug purposes, define DEBUG. */
#ifdef DEBUG
  if (rank == 0)
    printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

  // print score
  if (rank == 0)
    printf("\nScore: %d\n", score);

  // free score matrix
  freeScoreMatrix(scoreMatrix, sizeB);

  MPI_Finalize();

  return EXIT_SUCCESS;
}
