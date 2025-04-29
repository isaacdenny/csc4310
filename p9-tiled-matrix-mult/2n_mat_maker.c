/**
* Isaac Denny
*	CSC 4310
*	4/16/2025
* Quick program to build matrices with 2^N dimensions
*/
#include <stdio.h>
#include <stdlib.h>

// probably doesnt totally work perfect but looks like it does
float gen_rand_float() {
	return (float)rand()  / RAND_MAX + (rand() % 100);
}

int main(int argc, char **argv) {
  FILE *output = fopen(argv[1], "w");
  if (output == NULL) {
    perror("file");
    return 1;
  }

  if (argc > 4) {
    srand(atoi(argv[4]));
  }

	int n = atoi(argv[2]);
	int dim = 1 << n;

  fprintf(output, "%d\n", dim);

  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      fprintf(output, "%6.2f", gen_rand_float());
    }
    fprintf(output, "\n");
  }
  return 0;
}

