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

template<class T, class U>
void book_example_efficient(const vector<vector<T>> &M, const vector<T> &x, vector<U> &result)
{
  // vector<T> result(size);
  auto size = x.size();
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
  }
  // return result;
}


template<class T, class U>
void book_example_inefficient(const vector<vector<T>> &M, const vector<T> &x, vector<U> &result)
{
  // vector<T> result(size);
  auto size = x.size();
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
  }
}


template<class T, class U>
void classic_matrix_mult(vector<vector<T>> &A, vector<vector<T>> &B, vector<vector<U>> &result)
{
  int n = A.size();// Rows in A

  // vector<vector<T>> result(n, vector<T>(n, 0));

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) { result[i][j] += A[i][k] * B[k][j]; }
    }
  }
}

// Block Matrix Multiplication
template<class T, class U, class V>
void block_multiply(vector<vector<T>> &A, vector<vector<T>> &B, vector<vector<U>> &result, V blockSize)
{
  using std::min;
  using std::max;

  int n = A.size();

  // vector<vector<T>> result(n, vector<T>(n, 0));

  for (int i = 0; i < n; i += blockSize) {
    for (int j = 0; j < n; j += blockSize) {
      for (int k = 0; k < n; k += blockSize) {
        for (int ii = i; ii < min(i + blockSize, n); ii++) {
          for (int jj = j; jj < min(j + blockSize, n); jj++) {
            for (int kk = k; kk < min(k + blockSize, n); kk++) { result[ii][jj] += A[ii][kk] * B[kk][jj]; }
          }
        }
      }
    }
  }
}