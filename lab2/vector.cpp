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

#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <numeric>
#include <vector>

#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()

using namespace std;

template<class T, class U> std::ostream &operator<<(ostream &os, pair<T, U> v)
{
  return os << "(" << v.first << "," << v.second << ")";
}
template<class T> std::ostream &operator<<(ostream &os, vector<T> v)
{
  for (auto &&e : v) { os << e << " "; }
  return os;
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);// Initialize MPI

  int procs, mi_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);// Get total number of processes
  MPI_Comm_rank(MPI_COMM_WORLD, &mi_rank);// Get rank of current process

  MPI_Status status;
  // MPI_Status();// Get rank of current process

  const int v_sz = 1000;
  if (mi_rank == 0) {
    vector<int> vec(v_sz, 1);
    cout << "vec: " << vec << endl;

    for (int i = 1; i < procs; i++) {
      const int section_sz = v_sz / (procs - 1);
      vector<int> tmp((vec.begin() + (i - 1) * section_sz), (vec.begin() + i * section_sz));
      MPI_Send(tmp.data(), section_sz, MPI_INT, i, 0, MPI_COMM_WORLD);
    }

    int s = 0;

    for (size_t i = 1; i < procs; i++) {
      int tmp;
      MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
      s += tmp;
    }
    cout << "sumatoria:" << s << endl;

  } else {
    vector<int> tmp2(v_sz / (procs - 1));
    MPI_Recv(tmp2.data(), v_sz / (procs - 1), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    // cout << "tmp2: " << tmp2 << endl;
    int s = std::accumulate(all(tmp2), 0);

    cout << "before sending s: " << s << endl;
    MPI_Send(&s, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }

  MPI_Finalize();// Finalize MPI
  return 0;
}

/*

mpic++ vector.cpp -o vector.out
mpirun -np 3 ./vector.out
mpirun -np 5 ./vector.out

*/
