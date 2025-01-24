#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

void sqrt_pragma();
void sqrt_pragma_for();

int main() {
  sqrt_pragma();
  sqrt_pragma_for();
  return 0;
}

void sqrt_pragma() {
  int data[10] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
  int n = 10;
#pragma omp parallel
  {
#pragma omp for schedule(static, 3)
    for (int i = 0; i < n; i++) {
      printf("1: %d, %d\n", omp_get_thread_num(), i);
      data[i] = sqrt(abs(data[i]));
    }
  }

  for (int i = 0; i < n; i++) {
    printf("data[%d] = %d\n", i, data[i]);
  }
}

void sqrt_pragma_for() {
  int data[10] = {1, 4, 9, 16, 25, 36, 49, 64, 81, 100};
  int n = 10;
#pragma omp parallel for schedule(static, 3)
  for (int i = 0; i < n; i++) {
    printf("2: %d, %d\n", omp_get_thread_num(), i);
    data[i] = sqrt(abs(data[i]));
  }

  for (int i = 0; i < n; i++) {
    printf("data[%d] = %d\n", i, data[i]);
  }
}
