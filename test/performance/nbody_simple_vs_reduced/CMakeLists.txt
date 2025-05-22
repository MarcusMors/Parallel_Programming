cmake_minimum_required(VERSION 3.16)
project(omp_nbody CXX)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Debug flag (Set to ON to enable debug mode)
option(ENABLE_DEBUG "Enable debugging flags" ON)

if (ENABLE_DEBUG)
    message(STATUS "Building in Debug mode with extra flags")
    set(CMAKE_CXX_FLAGS "-O0")
    # set(CMAKE_CXX_FLAGS "-O0 -g3 -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined -Wall -Wextra -Wshadow -Wnon-virtual-dtor -pedantic -Wold-style-cast -Wformat=2")
else()
    message(STATUS "Building in Release mode with optimization")
    set(CMAKE_CXX_FLAGS "-O3")
endif()

# Define the source file (relative to this directory)
set(SOURCE_FILE one_process_nbody_bm.cpp)

# Include directories (relative to the test/performance directory)
# include_directories(../../lab1 ../../utils)

# Link Google Benchmark
find_package(benchmark REQUIRED)
find_package(Threads REQUIRED)
# find_package(m REQUIRED)

# Define the executable
add_executable(lab_1.out ${SOURCE_FILE})

# Link libraries
target_link_libraries(lab_1.out PRIVATE benchmark::benchmark Threads::Threads m)


# [==
# --benchmark_format=csv > benchmark_results.csv
# ==]