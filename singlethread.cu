#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// Enable error checking
#define CUDA_CHECK_ERROR
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 )
    do
    {
        if ( cudaSuccess != err )
        {
            fprintf( stderr,
                "cudaSafeCall() failed at %s:%i : %s\n",
                file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
#pragma warning( pop )
#endif
    return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef CUDA_CHECK_ERROR
#pragma warning( push )
#pragma warning( disable: 4127 )
    do
    {
        cudaError_t err = cudaGetLastError();
        if ( cudaSuccess != err )
        {
            fprintf( stderr,
                "cudaCheckError() failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
        err = cudaDeviceSynchronize();
        if( cudaSuccess != err )
        {
            fprintf( stderr,
                "cudaCheckError() with sync failed at %s:%i : %s.\n",
                file, line, cudaGetErrorString( err ) );
            exit( -1 );
        }
    } while ( 0 );
#pragma warning( pop )
#endif
    return;
}

// Single-threaded bubble sort kernel
__global__ void bubbleSortKernel(int* arr, int n)
{
    // Since this is single-threaded, we only want one thread to do the work
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < n-1; i++)
        {
            for (int j = 0; j < n-i-1; j++)
            {
                if (arr[j] > arr[j+1])
                {
                    // Swap elements
                    int temp = arr[j];
                    arr[j] = arr[j+1];
                    arr[j+1] = temp;
                }
            }
        }
    }
}

int* makeRandArray(const int size, const int seed) {
    srand(seed);
    int* array = new int[size];
    for(int i = 0; i < size; i++) {
        array[i] = std::rand() % 1000000;
    }
    return array;
}

int main(int argc, char* argv[])
{
    int* array;
    int size, seed;

    // Modified argument checking as per PDF requirements
    if(argc < 3) {
        fprintf(stderr, "usage: %s [number of random integers to generate] [seed value for random number generation]\n", 
                argv[0]);
        exit(-1);
    }

    // Parse arguments
    {
        std::stringstream ss1(argv[1]);
        ss1 >> size;
    }
    {
        std::stringstream ss1(argv[2]);
        ss1 >> seed;
    }

    // Get the random numbers
    array = makeRandArray(size, seed);

    // Create device array and copy input data to it
    int* d_array;
    CudaSafeCall(cudaMalloc(&d_array, size * sizeof(int)));
    CudaSafeCall(cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice));

    // Create CUDA timer
    cudaEvent_t startTotal, stopTotal;
    float timeTotal;
    cudaEventCreate(&startTotal);
    cudaEventCreate(&stopTotal);
    cudaEventRecord(startTotal, 0);

    // Launch kernel with single thread
    bubbleSortKernel<<<1, 1>>>(d_array, size);
    CudaCheckError();

    // Copy result back to host
    CudaSafeCall(cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost));

    // Stop timer and calculate elapsed time
    cudaEventRecord(stopTotal, 0);
    cudaEventSynchronize(stopTotal);
    cudaEventElapsedTime(&timeTotal, startTotal, stopTotal);
    cudaEventDestroy(startTotal);
    cudaEventDestroy(stopTotal);

    fprintf(stderr, "Total time in seconds: %f\n", timeTotal/1000.0);

    // Cleanup
    delete[] array;
    cudaFree(d_array);
    return 0;
}