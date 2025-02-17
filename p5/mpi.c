#define MASTER 0
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char message[512];
  int i, rank, size, tag = 99;
  char machine_name[256];
  MPI_Status status;

  printf("CSC-4310 - HPC\n");

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  gethostname(machine_name, 255);

  if (rank == MASTER) {
    printf("Hello world from master process %d running on %s\n", rank,
           machine_name);
    for (i = 1; i < size; i++) {
      printf("Ready to recv a message.\n");
      MPI_Recv(message, 512, MPI_CHAR, i, tag, MPI_COMM_WORLD, &status);
      printf("Message from process = %d : %s\n", i, message);
    }
  } else {
    sprintf(message, "===>Hello world from process %d running on %s", rank,
            machine_name);
    printf("	Send a message from %d.\n", rank);
    sleep(20);
    MPI_Send(message, 512, MPI_CHAR, MASTER, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
