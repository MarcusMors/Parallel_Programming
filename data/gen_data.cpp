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
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>

#include "../utils/gen_data.hpp"

using ull = unsigned long long;
using ul = unsigned long;
using ui = unsigned;
using ill = long long;

int main()
{
  using it_type = ull;
  // const ui Bytes = 1;
  // const it_type data_size = bit_number<it_type>(Bytes);
  // const it_type data_size = std::numeric_limits<it_type>::max();
  // const it_type data_size = 500'000;


  for (it_type data_size = 500'000; data_size <= 1'000'000; data_size += 50'000) {
    fill_file_with_random_data<it_type>(data_size);
  }

  // it_type data_size = 500'000;
  // for (it_type i = 0; data_size <= 1'000'000; i++) {
  //   data_size += 50'000;
  //   fill_file_with_random_data<it_type>(data_size);
  // }


  return 0;
}
