#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
using namespace std;

// Error checking defines
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

// Error checking function implementations
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    #ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    #endif
}

inline void __cudaCheckError(const char *file, const int line) {
    #ifdef CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    
    err = cudaDeviceSynchronize();  
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                file, line, cudaGetErrorString(err));
        exit(-1);
    }
    #endif
}

// Function to generate random array
int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 1000000;
    }
    return array;
}

// Device functions for parallel quicksort
__device__ void swap(int& a, int& b) {
    int temp = a;
    a = b;
    b = temp;
}

__device__ int partition(int* data, int left, int right) {
    int pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            swap(data[i], data[j]);
        }
    }
    swap(data[i + 1], data[right]);
    return i + 1;
}

// Global quicksort kernel
__global__ void quicksortKernel(int* data, int* workList, int* workListSize) {
    while (*workListSize > 0) {
        if (threadIdx.x == 0) {
            int workIndex = atomicSub(workListSize, 1);
            if (workIndex > 0) {
                int left = workList[workIndex * 2];
                int right = workList[workIndex * 2 + 1];
                
                if (left < right) {
                    int pivotIdx = partition(data, left, right);
                    
                    if (pivotIdx - 1 > left) {
                        int newIndex = atomicAdd(workListSize, 1);
                        workList[newIndex * 2] = left;
                        workList[newIndex * 2 + 1] = pivotIdx - 1;
                    }
                    
                    if (right > pivotIdx + 1) {
                        int newIndex = atomicAdd(workListSize, 1);
                        workList[newIndex * 2] = pivotIdx + 1;
                        workList[newIndex * 2 + 1] = right;
                    }
                }
            }
        }
        __syncthreads();
    }
}

int main(int argc, char* argv[]) {
    int* array;
    int size, seed;
    int *d_array, *d_workList, *d_workListSize;
    
    // Parse command line arguments
    if(argc < 3) {
        fprintf(stderr, "usage: %s [number of random integers to generate] [seed value for random number generation]\n", argv[0]);
        exit(-1);
    }
    
    // Convert command line arguments
    {
        stringstream ss1(argv[1]);
        ss1 >> size;
    }
    {
        stringstream ss1(argv[2]);
        ss1 >> seed;
    }
    
    // Generate random array
    array = makeRandArray(size, seed);
    
    // Create CUDA timer
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);
    
    // Allocate device memory
    CudaSafeCall(cudaMalloc(&d_array, size * sizeof(int)));
    CudaSafeCall(cudaMalloc(&d_workList, size * 2 * sizeof(int)));
    CudaSafeCall(cudaMalloc(&d_workListSize, sizeof(int)));
    
    // Copy data to device
    CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize work list
    int workListSize = 1;
    int initialWorkList[2] = {0, size - 1};
    CudaSafeCall(cudaMemcpy(d_workList, initialWorkList, 2 * sizeof(int), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(d_workListSize, &workListSize, sizeof(int), cudaMemcpyHostToDevice));
    
    // Calculate grid and block dimensions
    int threadsPerBlock = 256;
    int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
    numBlocks = max(numBlocks, 4);  // Ensure at least 1024 threads
    
    // Launch kernel
    quicksortKernel<<<numBlocks, threadsPerBlock>>>(d_array, d_workList, d_workListSize);
    CudaCheckError();
    
    // Copy result back to host
    CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));
    
    // Stop and destroy timer
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);
    
    // Print execution time
    fprintf(stderr, "Total time in seconds: %f\n", timeTotal / 1000.0);
    
    // Cleanup
    delete[] array;
    cudaFree(d_array);
    cudaFree(d_workList);
    cudaFree(d_workListSize);
    
    return 0;
}