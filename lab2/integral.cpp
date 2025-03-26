// Copyright (C) 2025 José Enrique Vilca Campana
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as
// published by the Free Software Foundation, either version 3 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include <iostream>
#include <mpi.h>
#include <vector>

double f(double x) { return (x / 5) * x; }

double Trap(double left_endpt, double right_endpt, int trap_count, double base_len)
{

  double estimate = (f(left_endpt) + f(right_endpt)) / 2.0;
  for (int i = 1; i <= trap_count - 1; i++) {
    double x = left_endpt + i * base_len;
    estimate += f(x);
  }
  estimate *= base_len;

  return estimate;
}

nt main(int argc, char *argv[])
{
  int my_rank, comm_sz, n = 1024;
  double a = 0.0, b = 5.0;
  double total_int;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);


  double h = (b - a) / n;
  int local_n = n / comm_sz;

  double local_a = a + my_rank * local_n * h;
  double local_b = local_a + local_n * h;
  double local_int = Trap(local_a, local_b, local_n, h);

  if (my_rank != 0) {
    MPI_Send(&local_int, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    //
  } else {
    total_int = local_int;
    for (int source = 1; source < comm_sz; source++) {
      MPI_Recv(&local_int, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      total_int += local_int;
    }
  }

  if (my_rank == 0) {
    printf("With n = %d trapezoids, our estimate of the integral\n", n);
    printf("from %f to %f = %.15e\n", a, b, total_int);
  }

  MPI_Finalize();
  return 0;
}

/*
mpic++ integral.cpp -o integral.out
mpirun -np 3 ./integral.out
 */