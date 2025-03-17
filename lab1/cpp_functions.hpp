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

//

#include <algorithm>
#include <cmath>
#include <vector>

using std::vector;

template<class T> vector<T> &book_example_efficient(const vector<vector<T>> M, const vector<T> x, auto size)
{
  vector<T> result(size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
  }
  return result;
}


template<class T> vector<T> &book_example_inefficient(const vector<vector<T>> M, const vector<T> x, auto size)
{
  vector<T> result(size);
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
  }
  return result;
}


template<class T> vector<vector<T>> &classic_matrix_mult(vector<vector<T>> A, vector<vector<T>> B)
{
  int n = A.size();// Rows in A
  int m = A[0].size();// Columns in A / Rows in B
  int p = B[0].size();// Columns in B

  vector<vector<T>> result(n, vector<T>(p, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < p; j++) {
      for (int k = 0; k < m; k++) { result[i][j] += A[i][k] * B[k][j]; }
    }
  }

  return result;
}
// Block Matrix Multiplication
template<typename T>
vector<vector<T>> block_multiply(const vector<vector<T>> &A, const vector<vector<T>> &B, auto blockSize)
{
  using std::min;
  using std::max;

  int n = A.size();// Rows of A
  int m = A[0].size();// Columns of A (Rows of B)
  int p = B[0].size();// Columns of B

  vector<vector<T>> result(n, vector<T>(p, 0));

  for (int i = 0; i < n; i += blockSize) {
    for (int j = 0; j < p; j += blockSize) {
      for (int k = 0; k < m; k += blockSize) {
        for (int ii = i; ii < min(i + blockSize, n); ii++) {
          for (int jj = j; jj < min(j + blockSize, p); jj++) {
            for (int kk = k; kk < min(k + blockSize, m); kk++) { result[ii][jj] += A[ii][kk] * B[kk][jj]; }
          }
        }
      }
    }
  }

  return result;
}