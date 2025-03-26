/**
 * Isaac Denny
 * CSC 4310
 * March 21 2025
 * A matrix multiplication program using cuda
 * Compile: nvcc -g matrix-mult-cuda.cu -o matrix-mult-cuda
 * Run ./matrix-mult-cuda <output_file> <input_file_a> <input_file_b>
 */
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

struct {
  int n;
  int m;
  int *data;
} typedef matrix;

void printMatrix(int *a, int n, int m);
int read_file_to_matrix(char *filename, matrix **a);
int write_matrix_to_file(char *filename, matrix *a);
int multiply_matrices(matrix *a, matrix *b, matrix *c);
__global__ void multiply_matrices_kernel(int *a, int *b, int *c, int n, int m,
                                         int p);

int main(int argc, char **argv) {

  if (argc != 4) {
    fprintf(stderr, "Usage: ./mult <output_file> <input_matrix_file_A> "
                    "<input_matrix_file_B>\n");
    return 1;
  }

  matrix *a = (matrix *)malloc(sizeof(matrix));
  int err = read_file_to_matrix(argv[2], &a);
  if (err > 0) {
    fprintf(stderr, "Error reading file to matrix A: %d\n", err);
    return 2;
  }

  matrix *b = (matrix *)malloc(sizeof(matrix));
  err = read_file_to_matrix(argv[3], &b);
  if (err > 0) {
    fprintf(stderr, "Error reading file to matrix B: %d\n", err);
    return 3;
  }

  matrix *c = (matrix *)malloc(sizeof(matrix) + sizeof(int) * a->n * b->m);
  if (c == NULL) {
    perror("Error allocating memory for c");
    return 4;
  }
  c->data = (int *)(c + 1);
  c->n = a->n;
  c->m = b->m;

  err = multiply_matrices(a, b, c);
  if (err > 0) {
    fprintf(stderr, "Error multiplying matrices: %d\n", err);
    return 5;
  }

  err = write_matrix_to_file(argv[1], c);
  if (err > 0) {
    fprintf(stderr, "Error writing matrix to file: %d\n", err);
    return 6;
  }

  free(a);
  free(b);
  free(c);

  return 0;
}

int read_file_to_matrix(char *filename, matrix **a) {
  FILE *input_file = fopen(filename, "r");
  if (input_file == NULL) {
    perror("Error reading file");
    return 1;
  }

  int n, m;
  fscanf(input_file, "%d %d", &n, &m);
  if (n < 0 || m < 0) {
    fprintf(stderr, "Invalid file format\n");
    return 2;
  }

  matrix *b = (matrix *)realloc(*a, sizeof(matrix) + sizeof(int) * n * m);

  b->n = n;
  b->m = m;
  b->data = (int *)(b + 1);

  for (int i = 0; i < n * m; i++) {
    fscanf(input_file, "%d", b->data + i);
  }

  *a = b;

  return 0;
}

void printMatrix(int *a, int n, int m) {
  printf("--- PRINT MATRIX START ---\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      printf("%d ", a[i * m + j]);
    }
    printf("\n");
  }
  printf("--- PRINT MATRIX END ---\n");
}

int write_matrix_to_file(char *filename, matrix *a) {
  FILE *output_file = fopen(filename, "w");
  if (output_file == NULL) {
    perror("Error reading file");
    return 1;
  }

  fprintf(output_file, "%d %d\n", a->n, a->m);
  if (a->n < 0 || a->m < 0) {
    fprintf(stderr, "Invalid file format\n");
    return 2;
  }

  for (int i = 0; i < a->n; i++) {
    for (int j = 0; j < a->m; j++) {
      fprintf(output_file, "%6d ", a->data[i * a->m + j]);
    }
    fprintf(output_file, "\n");
  }

  return 0;
}

int multiply_matrices(matrix *a, matrix *b, matrix *c) {
  if (a->m != b->n) {
    fprintf(
        stderr,
        "Error multiplying matrices: Row count of A != Column count of B\n");
    return 1;
  }

  // allocate device memory for matrices
  int *a_d, *b_d, *c_d;
  cudaError_t err = cudaMalloc((void **)&a_d, sizeof(int) * a->n * a->m);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating memory on device for matrix A: %s\n",
            cudaGetErrorString(err));
    return 1;
  }

  err = cudaMalloc((void **)&b_d, sizeof(int) * b->n * b->m);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating memory on device for matrix B: %s\n",
            cudaGetErrorString(err));
    return 2;
  }

  err = cudaMalloc((void **)&c_d, sizeof(int) * c->n * c->m);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error allocating memory on device for matrix C: %s\n",
            cudaGetErrorString(err));
    return 3;
  }

  // copy to device memory
  err = cudaMemcpy((void *)a_d, (void *)(a->data), sizeof(int) * a->n * a->m,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying matrix A to device: %s\n",
            cudaGetErrorString(err));
    return 4;
  }

  err = cudaMemcpy((void *)b_d, (void *)(b->data), sizeof(int) * b->n * b->m,
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying matrix B to device: %s\n",
            cudaGetErrorString(err));
    return 5;
  }

  dim3 threadsPerBlock(32, 32);
  dim3 blocksInGrid((b->m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (a->n + threadsPerBlock.y - 1) / threadsPerBlock.y);
  printf(
      "Cuda kernel initialized with dimensions: (%d, %d) and %d, %d blocks\n",
      threadsPerBlock.x, threadsPerBlock.y, blocksInGrid.x, blocksInGrid.y);
  multiply_matrices_kernel<<<blocksInGrid, threadsPerBlock>>>(a_d, b_d, c_d,
                                                              a->n, a->m, b->m);

  err = cudaMemcpy((void *)(c->data), (void *)c_d, sizeof(int) * c->n * c->m,
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error copying matrix C to host: %s\n",
            cudaGetErrorString(err));
    return 6;
  }

  err = cudaFree(a_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing memory A: %s\n", cudaGetErrorString(err));
    return 7;
  }
  err = cudaFree(b_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing memory B: %s\n", cudaGetErrorString(err));
    return 8;
  }
  err = cudaFree(c_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Error freeing memory C: %s\n", cudaGetErrorString(err));
    return 9;
  }

  return 0;
}

__global__ void multiply_matrices_kernel(int *a, int *b, int *c, int n, int m,
                                         int p) {

  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n || j >= p) {
    return;
  }

  int sum = 0;
  for (int k = 0; k < m; k++) {
    sum += a[i * m + k] * b[k * p + j];
  }

  c[i * p + j] = sum;
}
