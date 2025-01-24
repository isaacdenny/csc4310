/**
 * Isaac Denny
 * Jan 14 2025
 * CSC4310
 * Sums randomly generated elements of an array in parallel using pragma
 * reduction Compile: g++ -Wall sumP2.cpp -o sumP2 Run: ./sumP2
 */
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

using namespace std;

#define MAX_VALUE 10

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("usage: sumP1 <length> <num_threads> <output-y/n>\n");
    return 1;
  }

  // parse arguments
  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  char *arg3 = argv[3];
  bool show_output = false;

  if (n <= 0) {
    printf("length must be greater than 0\n");
    return 2;
  }

  if (num_threads <= 0) {
    printf("num_threads must be greater than 0\n");
    return 3;
  }

  if (strcmp(arg3, "y") == 0) {
    show_output = true;
  } else if (strcmp(arg3, "n") == 0) {
    show_output = false;
  } else {
    cerr << "usage: sumP1 <array size> <num_threads> <output-y/n>" << endl;
    exit(2);
  }

  struct timeval t1, t2;
  double runtime;

  srand(time(NULL));
  // allocate and fill array
  int *nums = new int[n];
  for (int i = 0; i < n; i++) {
    nums[i] = rand() % MAX_VALUE;
  }

  gettimeofday(&t1, NULL);
  // sum elements
  int sum = 0;
#pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
  for (int i = 0; i < n; i++) {
    sum += nums[i];
  }
  gettimeofday(&t2, NULL);

  if (show_output) {
    for (int i = 0; i < n; i++) {
      printf("nums[%d] = %d\n", i, nums[i]);
    }
  }

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Time to sum numbers: %lf\n", runtime);
  printf("Sum: %d\n", sum);
  return 0;
}
