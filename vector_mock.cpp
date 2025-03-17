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

#include "utils/new.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int main()
{
  const int size = 1000;
  vector<int> v(size);
  utils::RNG<utils::RNG_Config<>> rng_base;
  auto rng = [&rng_base]() { return rng_base(); };

  std::generate(v.begin(), v.end(), rng);

  for (auto i : v) { cout << i << " "; }

  return 0;
}