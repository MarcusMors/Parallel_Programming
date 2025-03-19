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

#include <iostream>

using namespace std;

int main()
{
  cout << "short: " << sizeof(short) << "\n";
  cout << "int: " << sizeof(int) << "\n";
  // cout << sizeof(short)<<"\n";
  // cout << sizeof(short)<<"\n";
  return 0;
}


/*
PAGE SIZE: 4096

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
