/**
 * Isaac Denny
 * CSC 4310
 * March 8 2025
 * A small CUDA tester that adds 2 vectors from files and writes the result to a
 * file
 * Compile: nvcc cuda-vecadd.cu -o vecadd
 * Run: vecadd <a_vec_file> <b_vec_file> <output_file>
 */

#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * kernel for adding 1 element from each vector. to be run across threads on
 * device only
 * pre: no device computation, host-allocated c vector
 * post: c vector filled with results
 */
__global__ void vecAddKernel(float *a, float *b, float *c, int n);

/**
 * host interface function to set up memory on device and launch kernel
 * pre: no device computation, host-allocated c vector
 * post: c vector filled with results
 */
void vecAdd(float *a, float *b, float *c, int n);

/**
 * loads a float vector from a file with 1 line starting with length of vector.
 * vector should be freed after use
 * pre: name of unopened file
 * post: returned vector filled with file's floats
 */
float *loadVector(char *filename, int *numElements);

/**
 * outputs a float vector to a file with 1 line starting with length of vector.
 * pre: name of unopened file
 * post: floats written to file
 */
void outputVector(char *filename, float *vec, int numElements);

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "Usage: vecadd <a_vec_file> <b_vec_file> <output_file>\n");
    exit(1);
  }

  int numElementsA = 0, numElementsB = 0;
  float *a_h = loadVector(argv[1], &numElementsA);
  float *b_h = loadVector(argv[2], &numElementsB);

  if (numElementsA != numElementsB) {
    fprintf(stderr, "Invalid file configuration: vector sizes must match\n");
    exit(5);
  }
  float *c_h = (float *)malloc(numElementsA * sizeof(float));

  vecAdd(a_h, b_h, c_h, numElementsA);

  outputVector(argv[3], c_h, numElementsA);

  free(a_h);
  free(b_h);
  free(c_h);

  return 0;
}

__global__ void vecAddKernel(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

void vecAdd(float *a, float *b, float *c, int n) {
  float *a_d, *b_d, *c_d;

  // allocate memory on device
  cudaError_t err = cudaMalloc(&a_d, n * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n",
            cudaGetErrorString(err));
    exit(5);
  }

  err = cudaMalloc(&b_d, n * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n",
            cudaGetErrorString(err));
    exit(5);
  }
  err = cudaMalloc(&c_d, n * sizeof(float));
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n",
            cudaGetErrorString(err));
    exit(5);
  }

  // copy vectors to device
  err = cudaMemcpy(a_d, a, n * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector A to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(6);
  }

  err = cudaMemcpy(b_d, b, n * sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy vector B to device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(6);
  }

  // start the kernel
  int threadsPerBlock = 32;
  int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launched with %d of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, n);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(7);
  }

  // get output to the host
  err = cudaMemcpy(c, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy result data from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(8);
  }

  // free data on device
  err = cudaFree(a_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free a on device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(9);
  }
  err = cudaFree(b_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free b on device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(9);
  }
  err = cudaFree(c_d);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free c on device (error code %s)!\n",
            cudaGetErrorString(err));
    exit(9);
  }
}

float *loadVector(char *filename, int *numElements) {
  FILE *f;
  float *vec;

  f = fopen(filename, "r");
  if (f == NULL) {
    fprintf(stderr, "Failed to open file %s\n", filename);
    exit(2);
  }

  // read number of elements
  fscanf(f, "%d ", numElements);
  if (*numElements < 0) {
    fprintf(stderr, "Invalid file format %s\n", filename);
    exit(3);
  }

  // read file into vector
  vec = (float *)malloc(*numElements * sizeof(float));
  for (int i = 0; i < *numElements; i++) {
    fscanf(f, "%f ", &vec[i]);
  }

  fclose(f);
  return vec;
}

void outputVector(char *filename, float *vec, int numElements) {
  FILE *f;

  f = fopen(filename, "w");
  if (f == NULL) {
    fprintf(stderr, "Failed to open file for writing %s\n", filename);
    exit(10);
  }

  // write the size and then vector values
  fprintf(f, "%d ", numElements);
  for (int i = 0; i < numElements; i++) {
    fprintf(f, "%f ", vec[i]);
  }

  fclose(f);
}
