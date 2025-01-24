// g++ openmp_confirm.cpp -o oc -fopenmp -lpthread
#include "omp.h"
#include <cstdio>

using namespace std;

int main() {
#pragma omp parallel
  {
#pragma omp critical
    printf("hi from thread %d\n", omp_get_thread_num());
  }
  return 0;
}
