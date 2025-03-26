
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

using ull = unsigned long long;
using ul = unsigned long;
using ui = unsigned;
using ill = long long;

namespace utils {


using namespace std;

template<class T_Seed = std::random_device,
  class T_Engine = std::default_random_engine,
  class T_intType = short int,
  class T_Distribution = std::uniform_int_distribution<T_intType>>
struct RNG_Config
{
  using Seed = T_Seed;
  using Engine = T_Engine;
  // supported int Types // check https://en.cppreference.com/w/cpp/header/random
  // short, int, long, long long,
  // ui short, ui int, ui long, or ull
  using intType = T_intType;
  using Distribution = T_Distribution;
};

template<class Config = RNG_Config<>> class RNG
{
private:
  using Seed = typename Config::Seed;
  using Engine = typename Config::Engine;
  using intType = typename Config::intType;
  using Distribution = typename Config::Distribution;

  Seed seed;
  Engine engine{ seed() };
  const intType m_max{ std::numeric_limits<intType>::max() };
  const intType m_min{ std::numeric_limits<intType>::min() };
  Distribution distribution = Distribution(m_min, m_max);

public:
  intType max() const { return m_max; }
  intType min() const { return m_min; }

  RNG(intType t_max, intType t_min) : m_max{ t_max }, m_min{ t_min }, distribution{ Distribution(m_min, m_max) } {}
  RNG() {}

  // To avoid copying the seed, which is an error because random_device is not copyable, we delete copy.


  intType operator()() { return distribution(engine); }
};


}// namespace utils

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using std::vector;

const std::string msg = "Errase:";


template<class T> void average(const vector<T> &vec)
{
  double avg = 0.0;
  size_t n = vec.size();

  for (size_t i = 0; i < n; ++i) {
    avg += (vec[i] - avg) / (i + 1);// Prevents overflow
  }

  std::cerr << msg + std::to_string(avg);
}
template<class T> void average(const vector<vector<T>> &matrix)
{
  double avg = 0.0;
  size_t count = 0;

  for (const auto &row : matrix) {
    for (int num : row) {
      ++count;
      avg += (num - avg) / count;// Incremental averaging formula
    }
  }
  std::cerr << msg + std::to_string(avg);
}


template<class T, class U>
void book_example_efficient(const vector<vector<T>> &M, const vector<T> &x, vector<U> &result)
{
  // vector<T> result(size);
  auto size = x.size();
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) { result[i] += M[i][j] * x[j]; }
  }
  average(result);
}


template<class T, class U>
void book_example_inefficient(const vector<vector<T>> &M, const vector<T> &x, vector<U> &result)
{
  // vector<T> result(size);
  auto size = x.size();
  for (int j = 0; j < size; j++) {
    for (int i = 0; i < size; i++) { result[i] += M[i][j] * x[j]; }
  }
  average(result);
}


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
  BMF_book_example_efficient(1000);
  BMF_book_example_inefficient(1000);

  return 0;
}