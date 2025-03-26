/**
 *	Isaac Denny
 *	CSC 4310
 *	15 March 2025
 *
 *	A serial matrix-quadrant adder: adds the 4 quadrants of a
 *	matrix from a file and writes the result to a file in the format
 *
 *	N
 *	x1 y1 ... N
 *	...
 *	xN yN ... N
 *
 *	where N is the dimensions of the square matrix
 *
 *	Compile: gcc -g -Wall serial.c -o serial
 *	Run: ./serial <input_file> <output_file>
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/**
 *	Reads a matrix file into an array of ints and returns the array.
 *	The matrix should be freed by the user
 *	pre: filename unopened file
 *	post: file is closed and data is loaded into int* result
 */
int *readFileToMatrix(char *filename, int *n);

/**
 *	Writes a matrix to a file; the matrix should be freed by the user,
 *	the function returns >0 if failed.
 *	pre: filename unopened file
 *	post: matrix is written to file
 */
void writeMatrixToFile(char *filename, int *y, int n);

/**
 *	Adds 4 quadrants to a resulting matrix of size N/2xN/2,
 *	the function returns resulting matrix.
 *	pre: x is matrix
 *	post: quadrant sum returned
 */
int *addQuadrants(int *x, int n);

int main(int argc, char **argv) {

  if (argc != 3) {
    fprintf(stderr, "Usage: serial <input_file> <output_file>\n");
    return 1;
  }

  struct timeval t1, t2, t3, t4;
  double runtime;
  gettimeofday(&t1, NULL);

  int n = 0;
  int *x = readFileToMatrix(argv[1], &n);


  gettimeofday(&t3, NULL);
  int *y = addQuadrants(x, n);
  gettimeofday(&t4, NULL);
  writeMatrixToFile(argv[2], y, n);

  free(x);
  free(y);

  gettimeofday(&t2, NULL);

  runtime = t2.tv_sec - t1.tv_sec + (t2.tv_usec-t1.tv_usec)/1.e6;
  printf("Overall time: %lf\n", runtime); 


  runtime = t4.tv_sec - t3.tv_sec + (t4.tv_usec-t3.tv_usec)/1.e6;
  printf("Serial add quadrant time: %lf\n", runtime); 
  return 0;
}

int *readFileToMatrix(char *filename, int *n) {

  FILE *input_file = fopen(filename, "r");
  if (input_file == NULL) {
    perror("Error opening input file");
    exit(2);
  }

  int n_temp = 0;
  int num_read = fscanf(input_file, "%d\n", &n_temp);
  if (num_read < 1) {
    perror("Error reading from input file");
    exit(3);
  }

  if (n_temp <= 0) {
    fprintf(stderr, "Error: n cannot be 0");
    exit(4);
  }

  int *x = malloc(sizeof(int) * n_temp * n_temp);
  for (int i = 0; i < n_temp; i++) {
    for (int j = 0; j < n_temp; j++) {
      num_read = fscanf(input_file, "%d", &x[i * n_temp + j]);
    }

    // get rid of newline
    fscanf(input_file, "\n");
  }

  *n = n_temp;

  fclose(input_file);
  return x;
}

void writeMatrixToFile(char *filename, int *y, int n) {
  FILE *output_file = fopen(filename, "w");
  if (output_file == NULL) {
    perror("Error opening output file");
    exit(5);
  }

  fprintf(output_file, "%d\n", n / 2);

  for (int i = 0; i < n / 2; i++) {
    for (int j = 0; j < n / 2; j++) {
      fprintf(output_file, "%d ", y[i * (n / 2) + j]);
    }

    fprintf(output_file, "\n");
  }

  fclose(output_file);
}

int *addQuadrants(int *x, int n) {
  int *y = malloc(sizeof(int) * n / 2 * n / 2);

  int nw = 0, ne = 0, sw = 0, se = 0;
  for (int i = 0; i < n / 2; i++) {
    for (int j = 0; j < n / 2; j++) {
      nw = x[i * n + j];
      ne = x[(i * n) + n / 2 + j];
      sw = x[i * n + ((n / 2) * n) + j];
      se = x[(i * n) + ((n / 2) * n) + n / 2 + j];


      y[i * (n / 2) + j] = nw + ne + sw + se;
    }
  }

  return y;
}
