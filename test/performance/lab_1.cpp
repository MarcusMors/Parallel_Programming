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
#include "../../lab1/cpp_functions.hpp"
#include "../../utils/new.hpp"
#include <algorithm>
#include <benchmark/benchmark.h>
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
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  std::generate(all(x), [&] { return short_rng_base(); });

  std::fstream file;
  file.open("data.txt");
  for (auto &&v : M) {
    for (auto &&i : v) { file << i << " "; }
    file << "\n";
  }
  file.close();

  book_example_efficient(M, x, result);
}

void BMF_book_example_inefficient(const std::size_t data_size)
{
  vector<vector<Data_type>> M(data_size, vector<Data_type>(data_size));
  vector<Data_type> x(data_size);
  vector<Data_type2> result(data_size);

  for (auto &&v : M) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  std::generate(all(x), [&] { return short_rng_base(); });
  book_example_inefficient(M, x, result);
}

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

// A wrapper used by the framework
static void BM_book_example_efficient(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_book_example_efficient(state.range(0)); }
}
static void BM_book_example_inefficient(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_book_example_inefficient(state.range(0)); }
}

static void BM_classic_matrix_mult(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_classic_matrix_mult(state.range(0)); }
}

static void BM_block_multiply(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_block_multiply(state.range(0)); }
}

BENCHMARK(BM_book_example_efficient)//
  ->Arg(500000)
  ->Arg(550000)
  ->Arg(600000)
  ->Arg(650000)
  ->Arg(700000)
  ->Arg(750000)
  ->Arg(800000)
  ->Arg(850000)
  ->Arg(900000)
  ->Arg(950000)
  ->Arg(1000000)
  //
  ;

BENCHMARK(BM_book_example_inefficient)//
  ->Arg(500000)
  ->Arg(550000)
  ->Arg(600000)
  ->Arg(650000)
  ->Arg(700000)
  ->Arg(750000)
  ->Arg(800000)
  ->Arg(850000)
  ->Arg(900000)
  ->Arg(950000)
  ->Arg(1000000);
//
;
BENCHMARK(BM_classic_matrix_mult)//
  ->Arg(500000)
  ->Arg(550000)
  ->Arg(600000)
  ->Arg(650000)
  ->Arg(700000)
  ->Arg(750000)
  ->Arg(800000)
  ->Arg(850000)
  ->Arg(900000)
  ->Arg(950000)
  ->Arg(1000000)
  //
  ;

// BENCHMARK(BM_block_multiply)
//   ->Arg(500000)
//   ->Arg(550000)
//   ->Arg(600000)
//   ->Arg(650000)
//   ->Arg(700000)
//   ->Arg(750000)
//   ->Arg(800000)
//   ->Arg(850000)
//   ->Arg(900000)
//   ->Arg(950000)
//   ->Arg(1000000);

BENCHMARK_MAIN();
