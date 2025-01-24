/**
 * Isaac Denny
 * Jan 12 2025
 * CSC4310
 * Sums randomly generated elements of an array
 * Compile: g++ -Wall psum.cpp -o psum
 * Run: ./psum
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sys/time.h>

using namespace std;

void gen_numbers(int numbers[], int how_many);
int gen_rand(int min, int max);
int sum_array(int nums[], int n);
void output_array(int nums[], int n);

int main(int argc, char *argv[]) {
  if (argc != 3) {
    cerr << "sum <array size> <output-y/n>" << endl;
    exit(2);
  }

  int n = atoi(argv[1]);
  char *arg3 = argv[2];
  bool show_output = false;
  if (strcmp(arg3, "y") == 0) {
    show_output = true;
  } else if (strcmp(arg3, "n") == 0) {
    show_output = false;
  } else {
    cerr << "sum <array size> <output-y/n>" << endl;
    exit(2);
  }

  struct timeval t1, t2;
  double runtime;

  int *numbers = new int[n];

  srand(time(NULL));


  gettimeofday(&t1, NULL);
  gen_numbers(numbers, n);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1.e6;
  printf("Time to generate numbers: %lf\n", runtime); 

  if (show_output) {
    output_array(numbers, n);
  }

  gettimeofday(&t1, NULL);
  int sum = sum_array(numbers, n);
  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1.e6;
  printf("Time to sum numbers: %lf\n", runtime); 

  printf("Sum: %d\n", sum);

  return 0;
}

// Fills an array with random numbers using gen_rand
void gen_numbers(int numbers[], int how_many) {
  for (int i = 0; i < how_many; i++) {
    numbers[i] = gen_rand(0, 10);
  }
}

// Generates a random number with specified range
int gen_rand(int min, int max) {
  return min + (int)rand() / ((int)RAND_MAX / (max - min));
}

// Sums an array of size n, returns the sum
int sum_array(int nums[], int n) {
  int sum = 0;

  for (int i = 0; i < n; i++) {
    sum += nums[i];
  }

  return sum;
}

// Outputs the array to stdout
void output_array(int nums[], int n) {
  printf("--- Current Array ---\n");
  for (int i = 0; i < n; i++) {
    printf("num[%d] = %d\n", i, nums[i]);
  }
  printf("--- End ---\n");
}
