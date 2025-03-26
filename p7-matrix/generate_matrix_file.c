
/**
 *	Isaac Denny
 *	CSC 4310
 *	15 March 2025
 *
 *	A serial generator for the matrix-quadrant adder: 
 *	writes a matrix to a file in the format
 *
 *	N
 *	x1 y1 ... N
 *	...
 *	xN yN ... N
 *
 *	where N is the dimensions of the square matrix
 *
 *	Compile: gcc -g -Wall gen_matrix_file.c -o gmf 
 *	Run: ./gmf <size> <output_file>
 */

#include <stdio.h>
#include <stdlib.h>

/**
 *	Writes a matrix to a file; the matrix should be freed by the user,
 *	the function returns >0 if failed.
 *	pre: filename unopened file
 *	post: matrix is written to file
 */
void writeMatrixToFile(char *filename, int *y, int n);

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "Usage: gmf <size> <output_file>\n");
    return 1;
  }

  int size = atoi(argv[1]);
  if (size < 0 || size % 2 != 0) {
    fprintf(stderr, "Usage: size must be even\n");
    return 2;
  }

  int* x = malloc(sizeof(int) * size * size);
  for(int i = 0; i < size * size; i++) {
    x[i] = rand() % 1000;
  }

  writeMatrixToFile(argv[2], x, size);

  free(x);
  return 0;
}


void writeMatrixToFile(char *filename, int *y, int n) {
  FILE *output_file = fopen(filename, "w");
  if (output_file == NULL) {
    perror("Error opening output file");
    exit(5);
  }

  fprintf(output_file, "%d\n", n);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(output_file, "%d ", y[i * n + j]);
    }

    fprintf(output_file, "\n");
  }

  fclose(output_file);
}

