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
struct Random_config
{
  using Seed = T_Seed;
  using Engine = T_Engine;
  // supported int Types // check https://en.cppreference.com/w/cpp/header/random
  // short, int, long, long long,
  // ui short, ui int, ui long, or ull
  using intType = T_intType;
  using Distribution = T_Distribution;
};

template<class It_type, class Stream, class Config = Random_config<>>
void random_data_generator(auto data_size, Stream &stream)
{
  using Seed = typename Config::Seed;
  using Engine = typename Config::Engine;
  using intType = typename Config::intType;
  using Distribution = typename Config::Distribution;

  Seed seed;
  Engine engine{ seed() };

  if (data_size == 0) { return; }
  if (std::vector<intType> v{}; v.max_size() < data_size) { return; }

  const intType max = std::numeric_limits<intType>::max();
  const intType min = std::numeric_limits<intType>::min();
  Distribution distribution(min, max);
  auto generate_random_number = [&]() { return distribution(engine); };

  const std::string type_size = to_str(sizeof(intType));
  const std::string sign = to_str(get_sign<intType>());
  const std::string data_size_str = to_str(data_size);

  const std::string file_name = "uniform_distribution__" + data_size_str + "__" + sign + type_size + "B.txt";


  std::ofstream out;
  out.open(file_name);

  for (It_type i = 0; i < data_size; i++) {
    const intType rand = generate_random_number();

    if constexpr (std::is_same_v < remove_cvref_t<Stream>,
      ostream >> or std::is_same_v < remove_cvref_t<Stream>,
      fstream >> or std::is_same_v < remove_cvref_t<Stream>,
      std::ostream & >>) {
      stream << rand << ' ';
    } else {
      stream.push_back(rand);
    }
  }

  if constexpr (std::is_same_v < remove_cvref_t<Stream>,
    ostream >> or std::is_same_v < remove_cvref_t<Stream>,
    fstream >>) {
  } else if constexpr (std::is_same_v < remove_cvref_t<Stream>, stringstream >>) {
    stream << file_name;
  } else {
    stream.push_back();
  }


  if constexpr (std::is_same_v < remove_cvref_t<Stream>,
    ostream >> or std::is_same_v < remove_cvref_t<Stream>,
    fstream >>) {
    stream.close();
  }
}


// pass cout
// pass fout
// pass stringstream

// use these to fill vectors or something

}// namespace utils
