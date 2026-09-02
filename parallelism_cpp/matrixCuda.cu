#include <chrono>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;
using namespace std::chrono;

const int N = 2048;
const int M = 2048;
const int P = 2048;

__global__ void multiply_matrixes(const long long* A, const long long* B, long long* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < P) {
        long long sum = 0;
        for (int k = 0; k < M; ++k) {
            sum += A[row * M + k] * B[k * P + col];
        }
        C[row * P + col] = sum;
    }
}

void initialize_matrix(long long* matrix, size_t size, int value_start) {
    for (size_t i = 0; i < size; ++i) {
        matrix[i] = static_cast<long long>(value_start) + i;
    }
}

int main() {
    long long *A, *B, *C;

    cudaMallocManaged(&A, static_cast<size_t>(N) * M * sizeof(long long));
    cudaMallocManaged(&B, static_cast<size_t>(M) * P * sizeof(long long));
    cudaMallocManaged(&C, static_cast<size_t>(N) * P * sizeof(long long));

    initialize_matrix(A, static_cast<size_t>(N) * M, 1);
    initialize_matrix(B, static_cast<size_t>(M) * P, 2);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaDeviceSynchronize(); 
    auto start = high_resolution_clock::now();

    multiply_matrixes<<<numBlocks, threadsPerBlock>>>(A, B, C);

    cudaDeviceSynchronize();

    auto end = high_resolution_clock::now();
    duration<double> duration_sec = end - start;

    cout << "C[0, 0] = " << C[0] << endl;
    cout << "C[N-1, P-1] = " << C[(static_cast<size_t>(N) - 1) * P + (P - 1)] << endl;

    cout << "Exec time (CUDA Kernel): " << duration_sec.count() << " sec" << endl;

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}