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
#include "../../../ipp-source-use/ch6/omp_nbody_basic.h"
#include "../../../ipp-source-use/ch6/omp_nbody_red.h"

//
// #include "../../utils/new.hpp"

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


int n_steps = 1000;
double delta_t = 0.1;
int output_freq = 1000;
char g_i = 'g';


void main_do_omp_basic(int n, int thread_count, int n_steps, double delta_t, int output_freq, char g_i)
{
  int step; /* Current step               */
  int part; /* Current particle           */
  double t; /* Current Time               */
  struct particle_s *curr; /* Current state of system    */
  vect_t *forces; /* Forces on each particle    */
  double start, finish; /* For timings                */

  curr = malloc(n * sizeof(struct particle_s));
  forces = malloc(n * sizeof(vect_t));
  if (g_i == 'i')
    Get_init_cond(curr, n);
  else
    Gen_init_cond(curr, n);

  start = omp_get_wtime();
#ifndef NO_OUTPUT
  Output_state(0, curr, n);
#endif
#pragma omp parallel num_threads(thread_count) default(none) \
  shared(curr, forces, thread_count, delta_t, n, n_steps, output_freq) private(step, part, t)
  for (step = 1; step <= n_steps; step++) {
    t = step * delta_t;
//    memset(forces, 0, n*sizeof(vect_t));
#pragma omp for
    for (part = 0; part < n; part++) Compute_force(part, forces, curr, n);
#pragma omp for
    for (part = 0; part < n; part++) Update_part(part, forces, curr, n, delta_t);
#ifndef NO_OUTPUT
#pragma omp single
    if (step % output_freq == 0) Output_state(t, curr, n);
#endif
  }

  finish = omp_get_wtime();
  printf("Elapsed time = %e seconds\n", finish - start);

  free(curr);
  free(forces);
}

void main_do_omp_reduced(int n, int thread_count, int n_steps, double delta_t, int output_freq, char g_i)
{
  //
  int step; /* Current step                     */
  int part; /* Current particle                 */
  double t; /* Current Time                     */
  struct particle_s *curr; /* Current state of system          */
  vect_t *forces; /* Forces on each particle          */
  double start, finish; /* For timing                       */
  vect_t *loc_forces; /* Forces computed by each thread   */
  curr = malloc(n * sizeof(struct particle_s));
  forces = malloc(n * sizeof(vect_t));
  loc_forces = malloc(thread_count * n * sizeof(vect_t));
  if (g_i == 'i')
    Get_init_cond(curr, n);
  else
    Gen_init_cond(curr, n);

  start = omp_get_wtime();
#ifndef NO_OUTPUT
  Output_state(0, curr, n);
#endif
#pragma omp parallel num_threads(thread_count) default(none) \
  shared(curr, forces, thread_count, delta_t, n, n_steps, output_freq, loc_forces) private(step, part, t)
  {
    int my_rank = omp_get_thread_num();
    int thread;

    for (step = 1; step <= n_steps; step++) {
      t = step * delta_t;
//       memset(loc_forces + my_rank*n, 0, n*sizeof(vect_t));
#pragma omp for
      for (part = 0; part < thread_count * n; part++) loc_forces[part][X] = loc_forces[part][Y] = 0.0;
#ifdef DEBUG
#pragma omp single
      {
        printf("Step %d, after memset loc_forces = \n", step);
        for (part = 0; part < thread_count * n; part++)
          printf("%d %e %e\n", part, loc_forces[part][X], loc_forces[part][Y]);
        printf("\n");
      }
#endif
      /* Particle n-1 will have all forces computed after call to
       * Compute_force(n-2, . . .) */
#pragma omp for schedule(static, 1)
      for (part = 0; part < n - 1; part++) Compute_force(part, loc_forces + my_rank * n, curr, n);
#pragma omp for
      for (part = 0; part < n; part++) {
        forces[part][X] = forces[part][Y] = 0.0;
        for (thread = 0; thread < thread_count; thread++) {
          forces[part][X] += loc_forces[thread * n + part][X];
          forces[part][Y] += loc_forces[thread * n + part][Y];
        }
      }
#pragma omp for
      for (part = 0; part < n; part++) Update_part(part, forces, curr, n, delta_t);
#ifndef NO_OUTPUT
      if (step % output_freq == 0) {
#pragma omp single
        Output_state(t, curr, n);
      }
#endif
    } /* for step */
  } /* pragma omp parallel */
  finish = omp_get_wtime();
  printf("Elapsed time = %e seconds\n", finish - start);

  free(curr);
  free(forces);
  free(loc_forces);
  return 0;
}

void BMF_nbody_simple(const std::size_t data_size, int thread_count)
{
  int n = data_size;

  main_do_omp_basic(n, thread_count, n_steps, delta_t, output_freq, g_i);
}

void BMF_nbody_reduced(const std::size_t data_size, int thread_count)
{
  int n = data_size;

  main_do_omp_reduced(n, thread_count, n_steps, delta_t, output_freq, g_i);
}

// A wrappers used by the framework
// =================================================================================

static void BM_nbody_simple_4(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_simple(state.range(0)), 4; }
}
static void BM_nbody_reduced_4(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_reduced(state.range(0), 4); }
}
// =================================================================================

static void BM_nbody_simple_3(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_simple(state.range(0)), 3; }
}
static void BM_nbody_reduced_3(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_reduced(state.range(0), 3); }
}
// =================================================================================

static void BM_nbody_simple_2(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_simple(state.range(0)), 2; }
}
static void BM_nbody_reduced_2(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_reduced(state.range(0), 2); }
}
// =================================================================================

static void BM_nbody_simple_1(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_simple(state.range(0)), 1; }
}
static void BM_nbody_reduced_1(benchmark::State &state)
{
  for (auto &&_ : state) { BMF_nbody_reduced(state.range(0), 1); }
}
// =================================================================================

BENCHMARK(BM_nbody_simple_4)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_simple_3)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_simple_2)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_simple_1)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_reduced_4)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_reduced_3)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_reduced_2)->DenseRange(100, 1000, 100)->Iterations(5);
BENCHMARK(BM_nbody_reduced_1)->DenseRange(100, 1000, 100)->Iterations(5);

BENCHMARK_MAIN();


/*
./one_process_nbody_bm.out --benchmark_min_time=5 --benchmark_report_aggregates_only=true --benchmark_format=json >
one_process_nbody_bm_results.json

./one_process_nbody_bm.out --benchmark_report_aggregates_only=true --benchmark_format=json >
one_process_nbody_bm_results.json

benchplot -t central_tendency --title "int type book example benchmark results"
int_book_one_process_nbody_bm_results.json
*/
