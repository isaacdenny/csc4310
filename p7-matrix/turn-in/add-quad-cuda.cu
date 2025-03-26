/**
 *	Isaac Denny
 *	CSC 4310
 *	15 March 2025
 *
 *	A cuda matrix-quadrant adder: adds the 4 quadrants of a
 *	matrix from a file and writes the result to a file in the format
 *
 *	N
 *	x1 y1 ... N
 *	...
 *	xN yN ... N
 *
 *	where N is the dimensions of the square matrix
 *
 *	Compile: nvcc serial.c -o add-quad-cuda
 *	Run: ./add-quad-cuda <input_file> <output_file>
 */

#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 *	Reads a matrix file into an array of ints and returns the array.
 *	The matrix should be freed by the user
 *	pre: filename unopened file
 *	post: file is closed and data is loaded into int* result
 */
int *readFileToMatrix(char *filename, int *n);

/**
 *	Writes a matrix to a file; the matrix should be freed by the user,
 *	the function returns >0 if failed.
 *	pre: filename unopened file
 *	post: matrix is written to file
 */
void writeMatrixToFile(char *filename, int *y, int n);

/**
 *	Adds 4 quadrants to a resulting matrix of size N/2xN/2,
 *	the function returns resulting matrix.
 *	pre: x is matrix
 *	post: quadrant sum returned
 */
int *addQuadrants(int *x, int n);

/**
 * kernel for adding quadrants of a matrix. to be run across threads on
 * device only
 * pre: no device computation, host-allocated x and y matrices
 * post: y matrix filled with results
 */
__global__ void quadAddKernel(int *x, int *y, int n);

int main(int argc, char **argv)
{

  if (argc != 3) {
    fprintf(stderr, "Usage: serial <input_file> <output_file>\n");
    return 1;
  }

  struct timeval t1, t2, t3, t4;
  double runtime;
  gettimeofday(&t1, NULL);

  int n = 0;
  int *x = readFileToMatrix(argv[1], &n);

  gettimeofday(&t3, NULL);
  int *y = addQuadrants(x, n);
  gettimeofday(&t4, NULL);

  writeMatrixToFile(argv[2], y, n);

  free(x);
  free(y);

  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Overall time: %lf\n", runtime);

  runtime = t4.tv_sec - t3.tv_sec + (t4.tv_usec - t3.tv_usec) / 1.e6;
  printf("Cuda setup, adding quadrants, and cleanup time: %lf\n", runtime);
  return 0;
}

int *readFileToMatrix(char *filename, int *n)
{

  FILE *input_file = fopen(filename, "r");
  if (input_file == NULL) {
    perror("Error opening input file");
    exit(2);
  }

  int n_temp = 0;
  int num_read = fscanf(input_file, "%d\n", &n_temp);
  if (num_read < 1) {
    perror("Error reading from input file");
    exit(3);
  }

  if (n_temp <= 0) {
    fprintf(stderr, "Error: n cannot be 0");
    exit(4);
  }

  int *x = (int *)malloc(sizeof(int) * n_temp * n_temp);
  for (int i = 0; i < n_temp; i++) {
    for (int j = 0; j < n_temp; j++) {
      num_read = fscanf(input_file, "%d", &x[i * n_temp + j]);
    }

    // get rid of newline
    fscanf(input_file, "\n");
  }

  *n = n_temp;

  fclose(input_file);
  return x;
}

void writeMatrixToFile(char *filename, int *y, int n)
{
  FILE *output_file = fopen(filename, "w");
  if (output_file == NULL) {
    perror("Error opening output file");
    exit(5);
  }

  fprintf(output_file, "%d\n", n / 2);

  for (int i = 0; i < n / 2; i++) {
    for (int j = 0; j < n / 2; j++) {
      fprintf(output_file, "%d ", y[i * (n / 2) + j]);
    }

    fprintf(output_file, "\n");
  }

  fclose(output_file);
}

int *addQuadrants(int *x, int n)
{
  int *y_h = (int *)malloc(sizeof(int) * n / 2 * n / 2);

  // initialize memory on device
  int *x_d, *y_d;
  cudaError_t err = cudaMalloc(&x_d, n * n * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device matrix x (error code %s)\n",
            cudaGetErrorString(err));
    exit(4);
  }

  err = cudaMalloc(&y_d, n / 2 * n / 2 * sizeof(int));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device matrix y (error code %s)\n",
            cudaGetErrorString(err));
    exit(5);
  }

  // copy matrix x to device
  err = cudaMemcpy(x_d, x, n * n * sizeof(int), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy matrix x to device (error code %s)\n",
            cudaGetErrorString(err));
    exit(6);
  }

  struct timeval t1, t2;
  double runtime;

  // start the kernel
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((n / 2 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (n / 2 + threadsPerBlock.y - 1) / threadsPerBlock.y);
  printf("CUDA kernel launched with dimensions: %d %d\n", blocksPerGrid.x,
         blocksPerGrid.y);

  gettimeofday(&t1, NULL);
  quadAddKernel<<<blocksPerGrid, threadsPerBlock>>>(x_d, y_d, n);
  gettimeofday(&t2, NULL);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(7);
  }

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Cuda adding quadrants time: %lf\n", runtime);

  // copy results back to host
  err =
      cudaMemcpy(y_h, y_d, n / 2 * n / 2 * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy matrix y to host (error code %s)\n",
            cudaGetErrorString(err));
    exit(7);
  }

  // Free device memory
  err = cudaFree(x_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free x on device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(9);
  }

  err = cudaFree(y_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free y on device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(9);
  }

  return y_h;
}

__global__ void quadAddKernel(int *x, int *y, int n)
{
  int t_x = blockIdx.x * blockDim.x + threadIdx.x;
  int t_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (t_x > n / 2 || t_y > n / 2) {
    return;
  }
  int nw = 0, ne = 0, sw = 0, se = 0;
  int loc = t_x + t_y * n;
  nw = x[loc];
  ne = x[loc + n / 2];
  sw = x[loc + (n / 2) * n];
  se = x[loc + ((n / 2) * n) + (n / 2)];
  y[t_x + t_y * (n / 2)] = nw + ne + sw + se;
}

/**
 * x = 0, y = 4
 * 20
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 4 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 4 0 0 0 0 0 0 0 0 0 4 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 * 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 */
