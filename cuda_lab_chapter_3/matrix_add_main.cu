#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <sys/time.h>

#define CUDA_CHECK(call)                                                                          \
  {                                                                                               \
    cudaError_t err = call;                                                                       \
    if (err != cudaSuccess) {                                                                     \
      fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(1);                                                                                    \
    }                                                                                             \
  }

// Kernels (B, C, D)
__global__ void matrix_add_kernel_element(float *c, const float *a, const float *b, int n)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) { c[row * n + col] = a[row * n + col] + b[row * n + col]; }
}

__global__ void matrix_add_kernel_row(float *c, const float *a, const float *b, int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n) {
    for (int col = 0; col < n; col++) { c[row * n + col] = a[row * n + col] + b[row * n + col]; }
  }
}

__global__ void matrix_add_kernel_column(float *c, const float *a, const float *b, int n)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n) {
    for (int row = 0; row < n; row++) { c[row * n + col] = a[row * n + col] + b[row * n + col]; }
  }
}

// Host function (supports all kernels via 'mode' argument)
void matrix_add_host(float *h_c, const float *h_a, const float *h_b, int n, char mode)
{
  float *d_a, *d_b, *d_c;
  size_t size = n * n * sizeof(float);

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_a, size));
  CUDA_CHECK(cudaMalloc(&d_b, size));
  CUDA_CHECK(cudaMalloc(&d_c, size));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

  // Launch kernel based on mode
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));

  switch (mode) {
  case 'B': {
    dim3 block_size(16, 16);
    dim3 grid_size((n + 15) / 16, (n + 15) / 16);
    matrix_add_kernel_element<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());// Added error check
    break;
  }
  case 'C': {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    matrix_add_kernel_row<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());// Added error check
    break;
  }
  case 'D': {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    matrix_add_kernel_column<<<grid_size, block_size>>>(d_c, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());// Added error check
    break;
  }
  default:
            fprintf(stderr, "Invalid mode. Use B/C/D.\n");
            exit(1);
  }

  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  // Measure kernel time
  float milliseconds = 0;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
  printf("Kernel time: %.3f ms\n", milliseconds);

  // Copy result back
  CUDA_CHECK(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

  // Cleanup
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

int main(int argc, char **argv)
{
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <matrix_size> <mode (B/C/D)>\n", argv[0]);
    return 1;
  }

  int n = atoi(argv[1]);
  char mode = argv[2][0];

  // Allocate and initialize host matrices
  float *h_a = (float *)malloc(n * n * sizeof(float));
  float *h_b = (float *)malloc(n * n * sizeof(float));
  float *h_c = (float *)malloc(n * n * sizeof(float));

  // Initialize A and B (e.g., A[i][j] = i + j, B[i][j] = i - j)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      h_a[i * n + j] = i + j;
      h_b[i * n + j] = i - j;
    }
  }

  // Run kernel
  matrix_add_host(h_c, h_a, h_b, n, mode);

  // Validate result (C should be A + B = 2i)
  bool valid = true;
  const float tolerance = 1e-6;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float expected = 2.0f * i;
      float actual = h_c[i * n + j];
      if (fabs(actual - expected) > tolerance) {
        valid = false;
        break;
      }
    }
    if (!valid) break;
  }

  free(h_a);
  free(h_b);
  free(h_c);
  return 0;
}