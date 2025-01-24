/**
 * Isaac Denny
 * Jan 14 2025
 * CSC4310
 * Sums randomly generated elements of an array in parallel
 * Compile: g++ -Wall sumP1.cpp -o sumP1
 * Run: ./sumP1
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
  int length = atoi(argv[1]);
  int num_threads = atoi(argv[2]);
  char *arg3 = argv[3];
  bool show_output = false;

  if (length <= 0) {
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
  int *nums = new int[length];
  for (int i = 0; i < length; i++) {
    nums[i] = rand() % MAX_VALUE;
  }

  gettimeofday(&t1, NULL);
  // sum elements
  int sum = 0;

  int thread_work = length / num_threads;
  int left_over_work =
      length % num_threads; // 10 % 4 leaves 2 left over, we can add that to the
                            // last thread to make sure it's calculated
  if (show_output) {
    printf("thread_work: %d, left over: %d\n", thread_work, left_over_work);
  }

#pragma omp parallel num_threads(num_threads)
  {
    /*
     * start index found by num_threads * thread_id (10 threads * 9 = 90, so
     * thread 9 starts at index 90
     */
    int this_thread_work = thread_work;
    int start_index = thread_work * omp_get_thread_num();
    if (show_output) {
      printf("Thread start %d: %d\n", omp_get_thread_num(), start_index);
    }

    // if this is the last thread and there is leftover work, add that work to
    // this_thread_work
    if (omp_get_thread_num() + 1 == num_threads && left_over_work > 0) {
      if (show_output) {
        printf("adding leftover work to thread %d: %d\n", omp_get_thread_num(),
               left_over_work);
      }
      this_thread_work += left_over_work;
    }

    int thread_sum = 0;
    for (int i = 0; i < this_thread_work; i++) {
      thread_sum += nums[start_index + i];
    }

#pragma omp critical
    {
      if (show_output) {
        printf("Thread adding sum %d: %d\n", omp_get_thread_num(), thread_sum);
      }
      sum += thread_sum;
    }
  }
  gettimeofday(&t2, NULL);

  if (show_output) {
    for (int i = 0; i < length; i++) {
      printf("nums[%d] = %d\n", i, nums[i]);
    }
  }

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec - t1.tv_usec) / 1.e6;
  printf("Time to sum numbers: %lf\n", runtime);
  printf("Sum: %d\n", sum);
  return 0;
}
