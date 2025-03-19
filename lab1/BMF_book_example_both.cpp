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


// function to benchmark
void BMF_book_example_efficient(const std::size_t data_size)
{
  vector<vector<Data_type>> M(data_size, vector<Data_type>(data_size));
  vector<Data_type> x(data_size);
  vector<Data_type2> result(data_size);

  for (auto &&v : M) {
    std::generate(all(v), [&] { return int_rng_base(); });
  }

  std::generate(all(x), [&] { return int_rng_base(); });
  book_example_efficient(M, x, result);
}

void BMF_book_example_inefficient(const std::size_t data_size)
{
  vector<vector<Data_type>> M(data_size, vector<Data_type>(data_size));
  vector<Data_type> x(data_size);
  vector<Data_type2> result(data_size);

  for (auto &&v : M) {
    std::generate(all(v), [&] { return int_rng_base(); });
  }

  std::generate(all(x), [&] { return int_rng_base(); });
  book_example_inefficient(M, x, result);
}

int main()
{
  BMF_book_example_efficient(16000);
  BMF_book_example_inefficient(16000);

  return 0;
}