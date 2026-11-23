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

#ifndef DEFAULT_K
#define DEFAULT_K 100
#endif

typedef unsigned short mtype;

/* Read sequence from a file to a char vector.
   Filename is passed as parameter */
char *read_seq(char *fname) {
  FILE *fseq = NULL;
  long size = 0;
  char *seq = NULL;
  int i = 0;

  fseq = fopen(fname, "rt");
  if (fseq == NULL) {
    printf("Error reading file %s\n", fname);
    exit(1);
  }

  fseek(fseq, 0L, SEEK_END);
  size = ftell(fseq);
  rewind(fseq);

  seq = (char *)calloc(size + 1, sizeof(char));
  if (seq == NULL) {
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

mtype **allocateScoreMatrix(int sizeA, int sizeB) {
  int i;
  mtype **scoreMatrix = (mtype **)malloc((sizeB + 1) * sizeof(mtype *));
  for (i = 0; i < (sizeB + 1); i++)
    scoreMatrix[i] = (mtype *)calloc(sizeA + 1, sizeof(mtype));
  return scoreMatrix;
}

void initScoreMatrix(mtype **scoreMatrix, int sizeA, int sizeB) {
  int i, j;
  for (j = 0; j < (sizeA + 1); j++)
    scoreMatrix[0][j] = 0;

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

/* Computa um bloco individual (K x K) de forma sequencial e vetorizada.
   A paralelização OpenMP atua sobre a distribuição dos múltiplos blocos
   independentes de cada diagonal, maximizando localidade temporal/espacial. */
static inline void compute_block(mtype **scoreMatrix, char *seq_row, char *seq_col,
                                 int row_start, int row_end, int col_start,
                                 int col_end) {
  for (int r = row_start; r <= row_end; r++) {
    char cr = seq_row[r - 1];
    for (int c = col_start; c <= col_end; c++) {
      if (cr == seq_col[c - 1])
        scoreMatrix[r][c] = scoreMatrix[r - 1][c - 1] + 1;
      else
        scoreMatrix[r][c] = max(scoreMatrix[r - 1][c], scoreMatrix[r][c - 1]);
    }
  }
}

int LCS_MPI(mtype **scoreMatrix, int m, int n, int k, int procNum, int procRank,
            char *A, char *B) {
  /* Divisão com teto para garantir que todas as linhas e colunas sejam cobertas */
  int BRows = (m + k - 1) / k;
  int BCols = (n + k - 1) / k;
  int BDiags = BRows + BCols - 1;

  int maxBlocks = min(BRows, BCols);
  mtype *borderBuf = malloc((size_t)maxBlocks * 2 * k * sizeof(mtype));
  int *recv_counts = malloc(procNum * sizeof(int));
  int *displacements = malloc(procNum * sizeof(int));

  for (int i = 0; i < BDiags; i++) {
    /* Blocos (block_row, block_col) com block_row + block_col = i */
    int min_block_row = max(0, i - BCols + 1);
    int max_block_row = min(i, BRows - 1);
    int blocks_in_diag = max_block_row - min_block_row + 1;

    /* Calcula a fatia de blocos de cada rank e os parâmetros para o MPI_Allgatherv */
    for (int r = 0; r < procNum; r++) {
      int r_start, r_end;
      block_range(blocks_in_diag, procNum, r, &r_start, &r_end);
      int num_blocks_rank = (r_end >= r_start) ? (r_end - r_start + 1) : 0;
      recv_counts[r] = num_blocks_rank * 2 * k * (int)MPI_MTYPE_BYTES;
      displacements[r] = r_start * 2 * k * (int)MPI_MTYPE_BYTES;
    }

    int BStart, BEnd;
    block_range(blocks_in_diag, procNum, procRank, &BStart, &BEnd);

    /* Paralelização OpenMP sobre os blocos independentes atribuídos a este rank */
    #pragma omp parallel for schedule(dynamic)
    for (int t = BStart; t <= BEnd; t++) {
      int block_row = min_block_row + t;
      int block_col = i - block_row;
      int row_start = block_row * k + 1;
      int row_end = min((block_row + 1) * k, m);
      int col_start = block_col * k + 1;
      int col_end = min((block_col + 1) * k, n);

      compute_block(scoreMatrix, A, B, row_start, row_end, col_start, col_end);

      /* Empacota a última linha e a última coluna do bloco no buffer de bordas */
      mtype *border_ptr = borderBuf + (size_t)t * 2 * k;
      for (int col = col_start; col <= col_end; col++)
        border_ptr[col - col_start] = scoreMatrix[row_end][col];
      for (int row = row_start; row <= row_end; row++)
        border_ptr[k + row - row_start] = scoreMatrix[row][col_end];
    }

    /* Comunicação coletiva: sincroniza e troca bordas se houver mais de 1 rank */
    if (procNum > 1) {
      MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, borderBuf, recv_counts,
                     displacements, MPI_BYTE, MPI_COMM_WORLD);

      /* Desempacota as bordas dos blocos calculados pelos outros processos */
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
  printf("[DEBUG] start matrix\n");
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
  char *seqA, *seqB;
  int sizeA, sizeB, proc_rank, num_procs;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  seqA = read_seq("fileA.in");
  seqB = read_seq("fileB.in");

  sizeA = strlen(seqA);
  sizeB = strlen(seqB);

  /* Configuração do tamanho de bloco K:
     Prioridade: 1) argv[1], 2) env LCS_BLOCK_SIZE, 3) heurística por tamanho de N */
  int block_k = DEFAULT_K;
  char *env_k = getenv("LCS_BLOCK_SIZE");
  if (argc > 1 && atoi(argv[1]) > 0) {
    block_k = atoi(argv[1]);
  } else if (env_k != NULL && atoi(env_k) > 0) {
    block_k = atoi(env_k);
  } else {
    if (sizeA >= 20000) block_k = 200;
    else if (sizeA >= 10000) block_k = 100;
    else if (sizeA >= 2000) block_k = 50;
    else if (sizeA >= 500) block_k = 25;
    else block_k = 4;
  }

  mtype **scoreMatrix = allocateScoreMatrix(sizeA, sizeB);
  initScoreMatrix(scoreMatrix, sizeA, sizeB);

  MPI_Barrier(MPI_COMM_WORLD);
  double start_time = MPI_Wtime();

  mtype score =
      LCS_MPI(scoreMatrix, sizeB, sizeA, block_k, num_procs, proc_rank, seqB, seqA);

  MPI_Barrier(MPI_COMM_WORLD);
  double end_time = MPI_Wtime();
  double elapsed_time = end_time - start_time;

#ifdef DEBUG
  if (proc_rank == 0)
    printMatrix(seqA, seqB, scoreMatrix, sizeA, sizeB);
#endif

  if (proc_rank == 0) {
    printf("Score: %d\n", score);
    printf("Time: %.6f\n", elapsed_time);
  }

  freeScoreMatrix(scoreMatrix, sizeB);
  free(seqA);
  free(seqB);

  MPI_Finalize();
  return EXIT_SUCCESS;
}
