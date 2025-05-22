#include <cstdio>

__global__ void matrixVectorMulKernel(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += B[i * N + j] * C[j];
        }
        A[i] = sum;
    }
}

void matrixVectorMultiply(float *A, float *B, float *C, int N) {
    float *d_A, *d_B, *d_C;

    // Asignar memoria en el dispositivo
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copiar datos al dispositivo
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, N * sizeof(float), cudaMemcpyHostToDevice);

    // Configurar y lanzar kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    matrixVectorMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copiar resultado al host
    cudaMemcpy(A, d_A, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Liberar memoria del dispositivo
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}