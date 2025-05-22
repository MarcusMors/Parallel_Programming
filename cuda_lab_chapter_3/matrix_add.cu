/**
 * Copyright (C) 2025 José Enrique Vilca Campana
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <cstdio>
#include <cuda_runtime.h>

// PART B:
__global__ void matrix_add_kernel_element(float *c, const float *a, const float *b, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        c[row * n + col] = a[row * n + col] + b[row * n + col];
    }
}

// Execution configuration:
// dim3 block_size(16, 16);
// dim3 grid_size((n + 15) / 16, (n + 15) / 16);


// PART A:

void matrix_add_host(float *h_c, const float *h_a, const float *h_b, int n) {
    float *d_a, *d_b, *d_c;
    size_t size = n * n * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy input data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Example kernel launch for Part B (one element per thread)
    dim3 block_size(16, 16);
    dim3 grid_size((n + 15) / 16, (n + 15) / 16);
    matrix_add_kernel_element<<<grid_size, block_size>>>(d_c, d_a, d_b, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}


// Part C
__global__ void matrix_add_kernel_row(float *c, const float *a, const float *b, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        for (int col = 0; col < n; col++) {
            c[row * n + col] = a[row * n + col] + b[row * n + col];
        }
    }
}

// Execution configuration:
// int block_size = 256;
// int grid_size = (n + block_size - 1) / block_size;

// PART D
__global__ void matrix_add_kernel_column(float *c, const float *a, const float *b, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int row = 0; row < n; row++) {
            c[row * n + col] = a[row * n + col] + b[row * n + col];
        }
    }
}

// Execution configuration:
// int block_size = 256;
// int grid_size = (n + block_size - 1) / block_size;


/* 
nvcc matrix_add.cu -o matrix_add && ./matrix_add
*/
