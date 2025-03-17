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

template<typename T> T bit_number(ui Bytes)
{
  // static_assert(sizeof(T) < Bytes, "the type isn't able to hold that much Bytes");
  if (sizeof(T) < Bytes) {
    std::cerr << "the type isn't able to hold " << Bytes << " Bytes\n";
    std::cerr << "sizeof(T)\t: " << sizeof(T) << "\n";
    std::cerr << "Bytes\t\t: " << Bytes << " \n";
    return 0;
  }
  T n{ 1 };
  for (ui i = 0; i <= Bytes; i++) { n <<= 3; }

  return n;
}

template<typename T> std::string to_str(const T &number)
{
  std::stringstream ss;
  ss << number;
  std::string str;
  ss >> str;
  return str;
}

template<typename T>
concept Integer = std::is_integral_v<T>;

template<Integer T> constexpr char get_sign()
{
  if constexpr (std::is_unsigned_v<T>) {
    return 'u';
  } else {
    return 'i';
  }
}

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


// pass cout
// pass fout
// pass stringstream

// use these to fill vectors or something

// namespace utils
