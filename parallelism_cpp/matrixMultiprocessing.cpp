#include <chrono>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;
using namespace std::chrono;

const int N = 2048;
const int M = 2048;
const int P = 2048;

void initialize(vector<long long> &matrix, int size, long long valve_start){
    for (int i = 0; i < size; ++i){
        matrix[i] = valve_start + i;
    }
}

int main(){
    vector<long long> A(N*M);
    vector<long long> B(M*P);
    vector<long long> C(N*P, 0);

    initialize(A, N*M, 1);
    initialize(B, M*P, 2);

    auto start = high_resolution_clock::now();

    // only parallelizable code
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        for (int k = 0; k < M; ++k){
            long long r = A[i * M + k];
            for (int j = 0; j < P; ++j){
                C[i * P + j] += r * B[k * P + j];
            }
        }
    }

    auto end = high_resolution_clock::now();
    duration<double> duration_sec = end - start;

    cout<< "C[0, 0] = " << C[0] << endl;
    cout<< "C[N-1, P-1] = " << C[(N-1) * P + (P - 1)] << endl;

    cout << "Exec time (OpenMP): " << duration_sec.count() << " sec" << endl;

    return 0;
}