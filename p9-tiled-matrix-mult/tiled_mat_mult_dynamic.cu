/**
 * Isaac Denny
 * CSC 4310
 * 4/16/2025
 * Multiplies 2 matrices using cuda and small tile sizes
 * to fit in shared memory. Handle with care.
 *
 * Compile: nvcc tiled_mat_mult.cu -o tmm
 * Run: ./tmm <tile width> <input matrix A - file> <input matrix B – file>
 * <output matrix - file>
 */
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>

#define MAX_TILE_WIDTH 32

/**
 *	Reads a file to a square matrix using the format:
 *	N
 *	1.11 ... N.NN
 *	...
 *  N.NN
 *
 * pre: null pointer address
 * post: M points to sizeof(float) * N * N bytes, file closed
 */
int readFileToMat(float **M, int *dim, char *filename);

/**
 *	Writes a square matrix to a file using the format:
 *	N
 *	1.11 ... N.NN
 *	...
 *  N.NN
 *
 * pre: M points to sizeof(float) * N * N bytes
 * post: file filled with matrix data and closed
 */
int writeMatToFile(float *M, int dim, char *filename);

/**
 *	Prints a matrix to the console
 *	pre: none
 *	post: none
 */
void printMat(float *M, int dim);

/**
 *	Helper to set up device enviroment for the
 *	tiled matrix multiplication algorithm
 *	pre: no device memory allocated, all float* are allocated already
 *	post: device memory is cleaned up, P is filled with results
 */
int tiledMatrixMult(float *M, float *N, float **P, int dim, int tile_width);

/**
 *	Tiled matrix multiplication algorithm kernel to be run on the device
 *	using shared memory to speed it up.
 *	pre: all memory is allocated and grid is defined
 *	post: P is filled with results of tiling
 */
__global__ void matrixMulKernel(float *M, float *N, float *P, int dim,
                                int tile_width);

int main(int argc, char **argv)
{
  if (argc < 4) {
    fprintf(stderr, "Usage: <cuda_mult_v2 <tile width> <input matrix A - file> "
                    "<input matrix B – file> <output matrix - file>\n");
    return 1;
  }

  int tile_width = atoi(argv[1]);
  char *M_mat_file = argv[2];
  char *N_mat_file = argv[3];
  char *P_mat_file = argv[4];

  if (tile_width >= MAX_TILE_WIDTH) {
    fprintf(stderr, "Error: Max tile width = 32\n");
    return 2;
  }

  int M_dim, N_dim;
  float *M = NULL, *N = NULL;
  readFileToMat(&M, &M_dim, M_mat_file);
  readFileToMat(&N, &N_dim, N_mat_file);

  // printMat(M, M_dim);
  // printMat(N, N_dim);

  if (M_dim != N_dim) {
    fprintf(stderr, "Invalid matrix dimensions\n");
    return 3;
  }

  float *P = (float *)malloc(sizeof(float) * M_dim * M_dim);

  tiledMatrixMult(M, N, &P, M_dim, tile_width);

  // printMat(P, M_dim);

  writeMatToFile(P, M_dim, P_mat_file);

  free(M);
  free(N);
  free(P);

  return 0;
}

int readFileToMat(float **M, int *dim, char *filename)
{
  FILE *mat_file = fopen(filename, "r");
  if (mat_file == NULL) {
    perror("file error");
    return 1;
  }

  int d = 0;
  fscanf(mat_file, "%d", &d);
  if (d < 2) {
    fprintf(stderr, "Bad file dimensions\n");
    return 2;
  }

  float *mat = (float *)malloc(sizeof(float) * d * d);
  bzero(mat, sizeof(float) * d * d);

  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      fscanf(mat_file, "%f", mat + i * d + j);
    }
  }

  *dim = d;
  *M = mat;
  return 0;
}

void printMat(float *M, int dim)
{
  printf("--- MATRIX START %d ---\n", dim);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      printf("%6.2f", M[i * dim + j]);
    }
    printf("\n");
  }
  printf("--- MATRIX END ---\n");
}

int writeMatToFile(float *M, int dim, char *filename)
{
  FILE *file = fopen(filename, "w");
  if (file == NULL) {
    perror("file error");
    return 1;
  }

  fprintf(file, "%d\n", dim);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      fprintf(file, "%6.2f", M[i * dim + j]);
    }
    fprintf(file, "\n");
  }

  return 0;
}

int tiledMatrixMult(float *M, float *N, float **P, int dim, int tile_width)
{
  float *M_dev, *N_dev, *P_dev;

  cudaError_t err = cudaMalloc((void **)&M_dev, sizeof(float) * dim * dim);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating device memory for M: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  err = cudaMalloc((void **)&N_dev, sizeof(float) * dim * dim);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating device memory for N: %s\n",
            cudaGetErrorString(err));
    return 2;
  }

  err = cudaMalloc((void **)&P_dev, sizeof(float) * dim * dim);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating device memory for P: %s\n",
            cudaGetErrorString(err));
    return 3;
  }

  err = cudaMemcpy(M_dev, M, sizeof(float) * dim * dim, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying M to device: %s\n", cudaGetErrorString(err));
    return 4;
  }

  err = cudaMemcpy(N_dev, N, sizeof(float) * dim * dim, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying N to device: %s\n", cudaGetErrorString(err));
    return 5;
  }

  // each block calculates the partial results of a tile
  dim3 threadsPerBlock(tile_width, tile_width);
  dim3 blocks((dim + tile_width - 1) / tile_width,
              (dim + tile_width - 1) / tile_width);
  int smemsize = 2 * tile_width * tile_width * sizeof(float);

  struct timeval t1, t2;
  double runtime;
  gettimeofday(&t1, NULL);
  matrixMulKernel<<<blocks, threadsPerBlock, smemsize>>>(M_dev, N_dev, P_dev,
                                                         dim, tile_width);
  gettimeofday(&t2, NULL);
  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Kernel time: %lf\n", runtime);

  err =
      cudaMemcpy(*P, P_dev, sizeof(float) * dim * dim, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying P_dev to host: %s\n",
            cudaGetErrorString(err));
    return 6;
  }

  err = cudaFree(M_dev);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing M_dev memory: %s\n",
            cudaGetErrorString(err));
    return 7;
  }

  err = cudaFree(N_dev);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing N_dev memory: %s\n",
            cudaGetErrorString(err));
    return 8;
  }

  err = cudaFree(P_dev);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing P_dev memory: %s\n",
            cudaGetErrorString(err));
    return 9;
  }

  return 0;
};

__global__ void matrixMulKernel(float *M, float *N, float *P, int Width,
                                int tile_width)
{
  extern __shared__ float shared_mem[];
  float *Mds = shared_mem;
  float *Nds = shared_mem + tile_width * tile_width;

  int Row = blockIdx.y * tile_width + threadIdx.y;
  int Col = blockIdx.x * tile_width + threadIdx.x;

  // Build the Pvalue for this tile
  float Pvalue = 0;
  for (int ph = 0; ph < Width / tile_width; ++ph) {
    Mds[threadIdx.y * tile_width + threadIdx.x] =
        M[Row * Width + ph * tile_width + threadIdx.x];
    Nds[threadIdx.y * tile_width + threadIdx.x] =
        N[(ph * tile_width + threadIdx.y) * Width + Col];
    __syncthreads();

    for (int k = 0; k < tile_width; k++) {
      Pvalue +=
          Mds[threadIdx.y * tile_width + k] * Nds[k * tile_width + threadIdx.x];
    }

    __syncthreads();
  }

  P[Row * Width + Col] = Pvalue;
}
