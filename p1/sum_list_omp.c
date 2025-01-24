#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MAX_VALUE 100

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: sum_list <length> <num_threads>\n");
    return 1;
  }

  int length = atoi(argv[1]);
  int num_threads = atoi(argv[2]);

  if (length <= 0) {
    printf("length must be greater than 0\n");
    return 2;
  }

  if (num_threads <= 0) {
    printf("num_threads must be greater than 0\n");
    return 3;
  }

  srand(time(NULL));
  // allocate and fill array
  int *nums = malloc(length * sizeof(int));
  for (int i = 0; i < length; i++) {
    nums[i] = rand() % MAX_VALUE;
    printf("nums[%d] = %d\n", i, nums[i]);
  }

  // sum elements
  int sum = 0;
#pragma omp parallel num_threads(num_threads)
  {
    /*
     * start index found by num_threads * thread_id (10 threads * 9 = 90, so
     * thread 9 starts at index 90
     */
    int start_index = length / num_threads * omp_get_thread_num();
    printf("Thread start %d: %d\n", omp_get_thread_num(), start_index);

    int thread_sum = 0;
    for (int i = 0; i < length / num_threads; i++) {
      thread_sum += nums[start_index + i];
    }

#pragma omp critical
    {
      printf("Thread adding sum %d: %d\n", omp_get_thread_num(), thread_sum);
      sum += thread_sum;
    }
  }

  printf("Sum: %d\n", sum);
  return 0;
}
