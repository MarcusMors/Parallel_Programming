#ifndef __UTILITY_H__
#define __UTILITY_H__

// Copyright (C) 2022 José Enrique Vilca Campana
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
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace utils {
// template<class T> std::istream &operator>>(std::istream &is, std::vector<T> &t_vector);
template<class T> std::istream &operator>>(std::istream &is, std::vector<T> &t_vector)
{
  typename std::vector<T>::size_type sz;// NOLINT initialization
  if (is >> sz) {
    // char comma;// NOLINT initialization
    t_vector.reserve(sz);
    char comma{};
    if (is >> comma) {
      while (sz > 0) {
        T elem;// NOLINT initialization
        if (is >> elem) { t_vector.push_back(elem); }
        if (is >> comma) { ; }
        --sz;
      }

      return is;
    }
  }

  is.setstate(std::ios_base::failbit);
  return is;
}
// template<class T>std::ostream &operator<<(std::ostream &os, std::vector<T> t_vector);
template<class T> std::ostream &operator<<(std::ostream &os, std::vector<T> t_vector)
{
  for (auto &&elem : t_vector) { os << elem << ", "; }
  return os;
}
template<class int_type> std::string int_type_and_data_size(const std::size_t t_data_size);
template<class Data_type> std::vector<Data_type> fill_vector_with_random_data(const std::size_t data_size);

}// namespace utils

template<class int_type> std::string utils::int_type_and_data_size(const std::size_t t_data_size)
{
  const std::string var_size = std::to_string(sizeof(int_type));
  const std::string data_size = std::to_string(t_data_size);

  return std::string{ var_size + "b-" + data_size + "b" };
}


template<class Data_type> std::vector<Data_type> utils::fill_vector_with_random_data(const std::size_t data_size)
{
  using utils::operator>>;
  const std::string directory_path{ "/home/marcus/+projects/Parallel_Programming/data" };
  const std::string distribution{ "uniform_distribution" };
  const std::string prefix{ "/" + distribution + "__" };
  const std::string type{ "" };
  const int data_size_begin{ 10000 };
  const int data_size_step{ 10000 };
  // type
  // std::string var_size_and_data_size = int_type_and_data_size<Data_type>(data_size);

  // const std::string source{ directory_path + prefix + var_size_and_data_size + ".csv" };
  const std::string source{ directory_path + prefix + std::to_string(data_size) + ".csv" };
  std::ifstream ifs{ source };
  if (!ifs) { std::cerr << "couldn't open " << source << " for reading\n"; }
  if (std::vector<Data_type>().max_size() <= data_size) { std::cerr << "max_size reached"; }

  std::vector<Data_type> vec;
  // std::cout << "BEFORE" << std::endl;
  ifs >> vec;
  // std::cout << "AFTER" << std::endl;

  return vec;
}

#endif// __UTILITY_H__
