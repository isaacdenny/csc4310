/*
 * Created by: Isaac Denny
 * Date: 2/21/2025
 * CSC 4310
 * A simple use of openMPI to integrate over functions
 * Compile: mpicc -g -Wall mpi-integrate.c -o integral -lm
 * Usage: mpiexec -n 4 -hostfile hosts integral <a> <b> <dx_percentage>
 */

#include <math.h>
#include <stdlib.h>
#define MASTER 0
#include <mpi/mpi.h>
#include <stdio.h>
#include <unistd.h>

// y = 2x
double starterFunction(double x) { return x * 2; };
// 0.5(x-4.5)^2 + 3(x-3)^3 - (x-3)^4 + 7.2
double crazyFunction(double x) {
  return 0.5 * pow(x - 4.5, 2) + 3 * pow(x - 3, 3) - pow(x - 3, 4) + 7.2;
};

/*
 * Integrates over a range starting at a, ending at b, with width dx 
 * @return area underneath the curve over the range
 * @param start point
 * @param end point
 * @param trapezoid width
 */
double integrate_range(double (*f)(double), double a, double b, double dx);

int main(int argc, char **argv) {
  int i, rank, size, tag = 99;
  char machine_name[256];
  MPI_Status status;

  if (argc != 4) {
    fprintf(stderr, "Usage: integral <a> <b> <dx_percentage>\n");
    return 1;
  }

  double a = atof(argv[1]);
  double b = atof(argv[2]);
  double dx_percentage = atof(argv[3]) / 100;

  if (a > b) {
    fprintf(stderr, "Error: b must be greater than a\n");
    return 2;
  } else if (dx_percentage < 0 || dx_percentage > 100) {
    fprintf(stderr, "Error: dx must be a percentage\n");
    return 1;
  }

  printf("CSC-4310 - HPC\n");

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gethostname(machine_name, 255);

  if (rank == MASTER) {
    double start, end, width = (b - a) / size;
    // calculate width of each trapezoid
    double dx = dx_percentage * width;
    for (i = 1; i < size; i++) {
      start = i * width + a;
      end = start + width;
      MPI_Send(&start, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      MPI_Send(&end, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      MPI_Send(&dx, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
      printf("Sent range to process = %d: %f, %f\n", i, start, end);
    }

    double finalSum = 0, sumx = 0;
    finalSum += integrate_range(starterFunction, 0, width, dx);

    for (i = 1; i < size; i++) {
      MPI_Recv(&sumx, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, &status);
      finalSum += sumx;
      printf("Received sum from process = %d: %f\n", i, sumx);
    }

    printf("Final Sum = %f\n", finalSum);
  } else {
    double ax, bx, dx;
    double sum;
    MPI_Recv(&ax, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&bx, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&dx, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, &status);
    printf("===>Process %d running on %s received range: %f, %f, %f\n", rank,
           machine_name, ax, bx, dx);

    sum = integrate_range(starterFunction, ax, bx, dx);
    MPI_Send(&sum, 1, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);
    printf("	Sent data from %d: %f\n", rank, sum);
  }

  MPI_Finalize();
  return 0;
}

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
    sum += area;
  }

  return sum;
}
