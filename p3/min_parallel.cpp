/**
 * Isaac Denny
 * Jan 15 2025
 * CSC4310
 * Finds the min of randomly generated elements in an array in parallel
 * via 2 methods: omp manual decomposition and omp reduction
 * Compile: g++ -Wall min_parallel.cpp -o mmp -fopenmp -lpthread
 * Run: ./mp
 *
 * ---RESULTS---
 * Avg serial: .0140982500
 * Avg manual: .0036438500
 * Avg reduction: .0025528000
 *
 * Speedup serial/manual: 3.6496371512
 * Speedup serial/reduction: 5.3355856310
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <omp.h>
#include <sys/time.h>

using namespace std;

template <typename T> void gen_numbers(T numbers[], int how_many);
template <typename T> T  gen_rand(T min, T max);
template <typename T> T serial_partition(T nums[], int n);
template <typename T> T min_manual_partition(T nums[], int n, int num_threads);
template <typename T> T min_auto_partition(T nums[], int n, int num_threads);
template <typename T> void output_array(T nums[], int n);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    cerr << "usage: mp <array size> <num_threads> <output-y/n>" << endl;
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
    cerr << "usage: mp <array size> <output-y/n>" << endl;
    exit(3);
  }

  double *numbers = new double[n];
  double start, end, minim;

  srand(time(NULL));
  std::cout << std::fixed;

  start = omp_get_wtime();
  gen_numbers<double>(numbers, n);
  end = omp_get_wtime();
  std::cout << "Time to generate numbers: " << end - start << std::endl;

  if (show_output) {
    output_array<double>(numbers, n);
  }

  start = omp_get_wtime();
  minim = serial_partition<double>(numbers, n);
  end = omp_get_wtime();
  std::cout << "Time to min numbers (serial partition): " << end - start
            << std::endl;

  printf("Serial Min: %.2f\n", minim);

  start = omp_get_wtime();
  minim = min_manual_partition<double>(numbers, n, num_threads);
  end = omp_get_wtime();
  std::cout << "Time to min numbers (manual partition): " << end - start
            << std::endl;

  printf("Manual Min: %.2f\n", minim);

  start = omp_get_wtime();
  minim = min_auto_partition<double>(numbers, n, num_threads);
  end = omp_get_wtime();
  std::cout << "Time to min numbers (reduction): " << end - start << std::endl;

  printf("Reduction Min: %.2f\n", minim);

  delete[] numbers;
  return 0;
}

/**
 * Fills an array with random numbers using gen_rand
 * pre: empty array (garbage values)
 * post: array filled with min to max
 */
template <typename T> void gen_numbers(T numbers[], int how_many) {
  for (int i = 0; i < how_many; i++) {
    numbers[i] = gen_rand<T>(-1000.0, 1000.0);
  }
}

/**
 * Generates a random number with specified range
 * pre: min max
 * post: randomly generated number between min and max
 */
template <typename T> T gen_rand(T min, T max) {
  return min + (T)rand() / ((T)RAND_MAX / (max - min));
}

/*
 * Outputs the array to stdout
 * pre: none
 * post: array can be seen in stdout
 */
template <typename T> void output_array(T nums[], int n) {
  std::cout << "--- Current Array ---" << std::endl;
  std::cout << std::setprecision(2);
  for (int i = 0; i < n; i++) {
    std::cout << i << ": " << nums[i] << std::endl;
  }
  std::cout << "--- Current Array ---" << std::endl;
}


/**
 *  Serially find min in array
 *  pre: none
 *  post: min found, returned 
 */
template <typename T> T serial_partition(T nums[], int n) {
  T min = nums[0];
  for (int i = 0; i < n; i++) {
    if (nums[i] < min) {
      min = nums[i];
    }
  }
  return min;
}

/**
 *  Manually partitions the data for each thread
 *  pre: no threads up, no sum
 *  post: sum completed, threads rejoined
 */
template <typename T> T min_manual_partition(T nums[], int n, int num_threads) {
  T minim = 0;
  int thread_work = n / num_threads;
#pragma omp parallel num_threads(num_threads)
  {
    int start_index = thread_work * omp_get_thread_num();

    T thread_min = 0;
    for (int i = 0; i < thread_work; i++) {
      if (nums[start_index + i] < thread_min) {
        thread_min = nums[start_index + i];
      }
    }

#pragma omp critical
    {
      if (thread_min < minim) {
        minim = thread_min;
      }
    }
  }

  return minim;
}

/**
 *  Partitions the data for each thread using omp reduction
 *  pre: no threads up
 *  post: nums summed and sum returned, threads rejoined
 */
template <typename T> T min_auto_partition(T nums[], int n, int num_threads) {
  T minim = 0;
#pragma omp parallel for reduction(min:minim) num_threads(num_threads)
  for (int i = 0; i < n; i++) {
    if (nums[i] < minim) {
      minim = nums[i];
    }
  }
  return minim;
}
