/**
 *
 * Compile: gcc -g -Wall matrix-mult-serial.c -o matrix-mult-serial
 * Run ./matrix-mult-serial <output_file> <input_file_a> <input_file_b>
 */
#include <stdio.h>
#include <stdlib.h>

struct {
  int n;
  int m;
  int *data;
} typedef matrix;

int read_file_to_matrix(char *filename, matrix *a);
int write_matrix_to_file(char *filename, matrix *a);
int multiply_matrices(matrix *a, matrix *b, matrix *c);

int main(int argc, char **argv) {

  if (argc != 4) {
    fprintf(stderr, "Usage: ./mult <output_file> <input_matrix_file_A> "
                    "<input_matrix_file_B>\n");
    return 1;
  }

  matrix *a = malloc(sizeof(matrix));
  int err = read_file_to_matrix(argv[2], a);
  if (err > 0) {
    fprintf(stderr, "Error reading file to matrix A: %d\n", err);
    return 2;
  }

  matrix *b = malloc(sizeof(matrix));
  err = read_file_to_matrix(argv[3], b);
  if (err > 0) {
    fprintf(stderr, "Error reading file to matrix B: %d\n", err);
    return 3;
  }

  matrix *c = malloc(sizeof(matrix));
  err = multiply_matrices(a, b, c);
  if (err > 0) {
    fprintf(stderr, "Error multiplying matrices: %d\n", err);
    return 4;
  }

  err = write_matrix_to_file(argv[1], c);
  if (err > 0) {
    fprintf(stderr, "Error writing matrix to file: %d\n", err);
    return 5;
  }

  free(a->data);
  free(b->data);
  free(c->data);
  free(a);
  free(b);
  free(c);

  return 0;
}

int read_file_to_matrix(char *filename, matrix *a) {
  FILE *input_file = fopen(filename, "r");
  if (input_file == NULL) {
    perror("Error reading file");
    return 1;
  }

  int n, m;
  fscanf(input_file, "%d %d", &n, &m);
  if (n < 0 || m < 0) {
    fprintf(stderr, "Invalid file format\n");
    return 2;
  }

  a->data = malloc(sizeof(int) * n * m);
  a->n = n;
  a->m = m;

  for (int i = 0; i < n * m; i++) {
    fscanf(input_file, "%d ", &(a->data[i]));
  }

  return 0;
}

int write_matrix_to_file(char *filename, matrix *a) {
  FILE *output_file = fopen(filename, "w");
  if (output_file == NULL) {
    perror("Error reading file");
    return 1;
  }

  fprintf(output_file, "%d %d\n", a->n, a->m);
  if (a->n < 0 || a->m < 0) {
    fprintf(stderr, "Invalid file format\n");
    return 2;
  }

  for (int i = 0; i < a->n; i++) {
    for (int j = 0; j < a->m; j++) {
      fprintf(output_file, "%6d ", a->data[i * a->m + j]);
    }
    fprintf(output_file, "\n");
  }

  return 0;
}

int multiply_matrices(matrix *a, matrix *b, matrix *c) {
  if (a->m != b->n) {
    fprintf(
        stderr,
        "Error multiplying matrices: Row count of A != Column count of B\n");
    return 1;
  }

  c->data = malloc(sizeof(int) * a->n * b->m);
  c->n = a->n;
  c->m = b->m;

  int sum;
  for (int i = 0; i < a->n; i++) {
    for (int j = 0; j < b->m; j++) {
      sum = 0;
      for (int k = 0; k < a->m; k++) {
        sum += a->data[i * a->m + k] * b->data[k * b->m + j];
      }

      c->data[i * c->m + j] = sum;
    }
  }
  return 0;
}
