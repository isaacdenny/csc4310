/**
 * Isaac Denny
 * Jan 15 2025
 * CSC4310
 * Sums randomly generated elements of an array in parallel 
 * via 2 methods: omp manual and omp reduction
 * Compile: g++ -Wall sum_parallel.cpp -o sump -fopenmp -lpthread
 * Run: ./sump
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <sys/time.h>

using namespace std;

void gen_numbers(int numbers[], int how_many);
int gen_rand(int min, int max);
int sum_array(int nums[], int n);
int sum_manual_partition(int nums[], int n, int num_threads);
int sum_auto_partition(int nums[], int n, int num_threads);
void output_array(int nums[], int n);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "usage: sum <array size> <num_threads> <output-y/n>" << endl;
    exit(1);
  }

  int n = atoi(argv[1]);
  int num_threads = atoi(argv[2]);

  // data should be divisible evenly by thread count
  if (n % num_threads != 0) {
    cerr << "Error: data size should be evenly divisible by thread count"
         << endl;
    exit(2);
  }
  char *arg3 = argv[3];
  bool show_output = false;
  if (strcmp(arg3, "y") == 0) {
    show_output = true;
  } else if (strcmp(arg3, "n") == 0) {
    show_output = false;
  } else {
    cerr << "usage: sum <array size> <output-y/n>" << endl;
    exit(3);
  }

  struct timeval t1, t2;
  double runtime;

  int *numbers = new int[n];

  srand(time(NULL));

  gettimeofday(&t1, NULL);
  gen_numbers(numbers, n);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Time to generate numbers: %lf\n", runtime);

  if (show_output) {
    output_array(numbers, n);
  }

  gettimeofday(&t1, NULL);
  int sum = sum_manual_partition(numbers, n, num_threads);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Time to sum numbers (manual): %lf\n", runtime);

  printf("Sum 1: %d\n", sum);

  gettimeofday(&t1, NULL);
  sum = sum_auto_partition(numbers, n, num_threads);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Time to sum numbers (reduction): %lf\n", runtime);

  printf("Sum 2: %d\n", sum);

  delete[] numbers;
  return 0;
}

/**
 * Fills an array with random numbers using gen_rand
 * pre: empty array (garbage values)
 * post: array filled with min to max
 */
void gen_numbers(int numbers[], int how_many) {
  for (int i = 0; i < how_many; i++) {
    numbers[i] = gen_rand(0, 10);
  }
}

/**
 * Generates a random number with specified range
 * pre: min max
 * post: randomly generated number between min and max
 */
int gen_rand(int min, int max) {
  return min + (int)rand() / ((int)RAND_MAX / (max - min));
}

/*
 * Outputs the array to stdout
 * pre: none
 * post: array can be seen in stdout
 */
void output_array(int nums[], int n) {
  printf("--- Current Array ---\n");
  for (int i = 0; i < n; i++) {
    printf("num[%d] = %d\n", i, nums[i]);
  }
  printf("--- End ---\n");
}

/**
 *  Manually partitions the data for each thread
 *  pre: no threads up, no sum
 *  post: sum completed, threads rejoined
 */
int sum_manual_partition(int nums[], int n, int num_threads) {
  int sum = 0;
  int thread_work = n / num_threads;

#pragma omp parallel num_threads(num_threads)
  {
    /*
     * start index found by num_threads * thread_id (10 threads * 9 = 90, so
     * thread 9 starts at index 90
     */
    int start_index = thread_work * omp_get_thread_num();

    // local thread variable seems to outperform array accesses
    int thread_sum = 0;
    for (int i = 0; i < thread_work; i++) {
      thread_sum += nums[start_index + i];
    }

// avoid race condition with critical
#pragma omp critical
    {
      sum += thread_sum;
    }
  }

  return sum;
}

/**
 *  Partitions the data for each thread using omp reduction
 *  pre: no threads up
 *  post: nums summed and sum returned, threads rejoined
 */
int sum_auto_partition(int nums[], int n, int num_threads) {
  int sum = 0;
#pragma omp parallel for reduction(+ : sum) num_threads(num_threads)
  for (int i = 0; i < n; i++) {
    sum += nums[i];
  }
  return sum;
}
