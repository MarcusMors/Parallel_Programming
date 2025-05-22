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

void BMF_classic_matrix_mult_int(const std::size_t data_size)
{
  vector<vector<int>> A(data_size, vector<int>(data_size));
  vector<vector<int>> B(data_size, vector<int>(data_size));
  vector<vector<int>> result(data_size, vector<int>(data_size));

  for (auto &&v : A) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }
  for (auto &&v : B) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  classic_matrix_mult(A, B, result);
}

void BMF_classic_matrix_mult_short(const std::size_t data_size)
{
  vector<vector<short>> A(data_size, vector<short>(data_size));
  vector<vector<short>> B(data_size, vector<short>(data_size));
  vector<vector<short>> result(data_size, vector<short>(data_size));

  for (auto &&v : A) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }
  for (auto &&v : B) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  classic_matrix_mult(A, B, result);
}

void BMF_block_multiply_int(const std::size_t data_size, const int cache_size)
{
  vector<vector<int>> A(data_size, vector<int>(data_size));
  vector<vector<int>> B(data_size, vector<int>(data_size));
  vector<vector<int>> result(data_size, vector<int>(data_size));

  for (auto &&v : A) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }
  for (auto &&v : B) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  block_multiply(A, B, result, cache_size);
}
void BMF_block_multiply_short(const std::size_t data_size, const int cache_size)
{
  vector<vector<short>> A(data_size, vector<short>(data_size));
  vector<vector<short>> B(data_size, vector<short>(data_size));
  vector<vector<short>> result(data_size, vector<short>(data_size));

  for (auto &&v : A) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }
  for (auto &&v : B) {
    std::generate(all(v), [&] { return short_rng_base(); });
  }

  block_multiply(A, B, result, cache_size);
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

// =================================================================================

static void BM_classic_matrix_mult_int(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_classic_matrix_mult_int(state.range(0)); }
}
// /*********************************************************************************
static void BM_classic_matrix_mult_short(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_classic_matrix_mult_short(state.range(0)); }
}
// =================================================================================

static void BM_block_multiply_int_L1(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_block_multiply_int(state.range(0), 104); }
}
static void BM_block_multiply_int_L2(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_block_multiply_int(state.range(0), 295); }
}

// /*********************************************************************************

static void BM_block_multiply_short_L1(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_block_multiply_short(state.range(0), 147); }
}
static void BM_block_multiply_short_L2(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_block_multiply_short(state.range(0), 418); }
}

// =================================================================================

// BENCHMARK(BM_book_example_efficient)->DenseRange(1000, 16000, 1000)->Iterations(5);

// BENCHMARK(BM_book_example_inefficient)->DenseRange(1000, 16000, 1000)->Iterations(5);
BENCHMARK(BM_classic_matrix_mult_int)->DenseRange(100, 1000, 100)->Iterations(5);
// BENCHMARK(BM_classic_matrix_mult_short)->DenseRange(100, 1000, 100)->Iterations(5);

BENCHMARK(BM_block_multiply_int_L1)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_block_multiply_int_L2)->DenseRange(100, 1000, 100)->Iterations(5);
// BENCHMARK(BM_block_multiply_short_L1)->DenseRange(100, 1000, 100)->Iterations(5);
// BENCHMARK(BM_block_multiply_short_L2)->DenseRange(100, 1000, 100)->Iterations(5);

BENCHMARK_MAIN();


/*
./lab_1.out --benchmark_min_time=5 --benchmark_report_aggregates_only=true --benchmark_format=json > lab_1_results.json
./lab_1.out --benchmark_report_aggregates_only=true --benchmark_format=json > lab_1_results.json

benchplot -t central_tendency --title "int type book example benchmark results"  int_book_lab_1_results.json
*/


/*
PAGE SIZE: 4096

short Byte size: 2
int Byte size : 4
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
*/
