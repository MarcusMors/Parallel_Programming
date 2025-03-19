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

void BMF_classic_matrix_mult(const std::size_t data_size)
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

  classic_matrix_mult(A, B, result);
}

int main()
{
  BMF_classic_matrix_mult(1000);

  return 0;
}