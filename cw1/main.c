#include <omp.h>
#include <stdio.h>
#include <sys/time.h>

int main() {
  int a[15] = {0, 1, 2, 2, 3, 4, 5, 5, 5};
  int counts[10] = {0};

  double runtime;
  struct timeval t1, t2;

  gettimeofday(&t1, NULL);
#pragma omp parallel for reduction(+ : counts) shared(a)
  for (int i = 0; i < 15; i++) {
    counts[a[i]]++;
  }

  for (int i = 0; i < 10; i++) {
    printf("%d\n", counts[i]);
  }
  gettimeofday(&t2, NULL);

  printf("%ld", t2.tv_sec - t1.tv_sec);
  return 0;
}
