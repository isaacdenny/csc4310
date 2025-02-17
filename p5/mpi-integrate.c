#include <stdlib.h>
#define MASTER 0
#include <mpi/mpi.h>
#include <stdio.h>
#include <unistd.h>

// y = 2x
double f(double x) { return x * 2; };
double integrate_range(double (*f)(double), double a, double b, double dx);

int main(int argc, char **argv) {
  int i, rank, size, tag = 99, dx = 1, a = 0, b = 100;
  char machine_name[256];
  MPI_Status status;

  printf("CSC-4310 - HPC\n");

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gethostname(machine_name, 255);

  if (rank == MASTER) {
    int start, end, width = (a + b) / size;
    for (i = 1; i < size; i++) {
      start = i * width;
      end = start + width;
      MPI_Send(&start, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      MPI_Send(&end, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      MPI_Send(&dx, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      printf("Sent range to process = %d: %d, %d\n", i, start, end);
    }

    double finalSum = 0, sumx = 0;
    finalSum += integrate_range(f, 0, width, dx);

    for (i = 1; i < size; i++) {
      MPI_Recv(&sumx, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
      finalSum += sumx;
      printf("Received sum from process = %d: %f\n", i, sumx);
    }

    printf("Final Sum = %f\n", finalSum);
  } else {
    int ax, bx, dx;
    double sum;
    MPI_Recv(&ax, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&bx, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&dx, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    char *message = malloc(512);
    sprintf(message, "===>Process %d running on %s received range: %d, %d",
            rank, machine_name, ax, bx);

    sum = integrate_range(f, ax, bx, dx);
    MPI_Send(&sum, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
    printf("	Sent data from %d: %f\n", rank, sum);
  }

  MPI_Finalize();
  return 0;
}

// function to integrate over passed function with trapezoid area
double integrate_range(double (*f)(double), double a, double b, double dx) {
  if (a > b) {
    return 0;
  }
  // area = (y2 + y1) / 2 * dx
  double sum = 0, area = 0, y2 = 0, y1 = 0;
  for (double i = a + dx; i <= b; i += dx) {
    y1 = f(i - dx);
    y2 = f(i);
    area = (y2 + y1) / 2 * dx;
    printf("  Area: %f, %f, %f, %f: %f\n", i - dx, i, y1, y2, area);
    sum += area;
  }

  return sum;
}
