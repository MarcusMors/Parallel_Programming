#include "../utils/new.hpp"
#include "cpp_functions.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>


#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()

using Data_type = int;
using Data_type2 = int;
using std::vector;

using int_config =
  utils::RNG_Config<std::random_device, std::default_random_engine, int, std::uniform_int_distribution<int>>;
using short_config = utils::RNG_Config<>;

auto min = -1024;
auto max = 1024;

utils::RNG<int_config> int_rng_base(max, min);
utils::RNG<short_config> short_rng_base(max, min);

// auto short_rng = [&] { return short_rng_base(); };
// auto int_rng = [&] { return int_rng_base(); };


void BMF_block_multiply(const std::size_t data_size)
{
  vector<vector<Data_type>> A(data_size, vector<Data_type>(data_size));
  vector<vector<Data_type>> B(data_size, vector<Data_type>(data_size));
  vector<vector<Data_type2>> result(data_size, vector<Data_type2>(data_size));

  for (auto &&v : A) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }
  for (auto &&v : B) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  block_multiply(A, B, result, 2);
}

int main()
{
  BMF_block_multiply(104);

  return 0;
}
/*
PAGE SIZE: 4096

L1d cache:                            128 KiB (4 instances) = 131072
L1i cache:                            128 KiB (4 instances) = 131072
L2 cache:                             1 MiB (4 instances) = 1048576
L3 cache:                             6 MiB (1 instance) = 6291456

short Byte size: 2
L1 cache:                            147.8
L2 cache:                            418.04

L3 cache:                            1024

int Byte size : 4
L1 cache:                            104.5
L2 cache:                            295.6

L3 cache:                            724.07

g++-14 -std=c++23 -fno-omit-frame-pointer -O0 -g3 BMF_book_example_both.cpp -o BMF_book_example_both.out

valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out ./BMF_book_example_both.out
cg_annotate cachegrind.out
kcachegrind cachegrind.out

g++-14 -std=c++23 -fno-omit-frame-pointer -O0 -g3 BMF_classic_matrix_mult.cpp -o BMF_classic_matrix_mult.out
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out ./BMF_classic_matrix_mult.out
cg_annotate cachegrind.out
kcachegrind cachegrind.out

g++-14 -std=c++23 -fno-omit-frame-pointer -O0 -g3 BMF_block_multiply.cpp  -o BMF_block_multiply.out
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out ./BMF_block_multiply.out
cg_annotate cachegrind.out
kcachegrind cachegrind.out

            .          template<class T, class U>
            8  (0.0%)  void book_example_efficient(const vector<vector<T>> &M, const vector<T> &x, vector<U> &result)
            .          {
            .            // vector<T> result(size);
            4  (0.0%)    auto size = x.size();
       80,006  (0.0%)    for (int i = 0; i < size; i++) {
8,960,096,000 (10.4%)      for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
            .            }
            3  (0.0%)    average(result);
            4  (0.0%)  }
*/
