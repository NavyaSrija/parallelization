# Parallelization 
This project implements and compares three different CUDA-based sorting algorithms:
1. Thrust Library Sort
2. Single-threaded CUDA Bubble Sort
3. Multi-threaded CUDA Quicksort

## Requirements

- NVIDIA CUDA Toolkit
- NVIDIA GPU with CUDA support
- GNU Make
- C++ compiler with CUDA support (nvcc)

## Project Structure

```
.
├── Makefile
├── thrust.cu         # Thrust library implementation
├── singlethread.cu   # Single-threaded bubble sort
└── multithread.cu    # Multi-threaded quicksort
```

## Building the Project

To compile all implementations:
```bash
make
```

## Running the Tests

The project includes predefined test cases for each implementation:

### Run all tests:
```bash
make run
```

### Run specific implementation tests:
```bash
make run_thrust    # Run Thrust tests
make run_single    # Run Single-threaded tests
make run_multi     # Run Multi-threaded tests
```

## Test Cases and Results

### Thrust Implementation
Tests with arrays of size:
- 10,000,000 elements (~0.015 seconds)
- 20,000,000 elements (~0.029 seconds)
- 30,000,000 elements (~0.042 seconds)
- 40,000,000 elements (~0.054 seconds)
- 50,000,000 elements (~0.067 seconds)

### Single-threaded Implementation
Tests with arrays of size:
- 2,000 elements (~0.069 seconds)
- 5,000 elements (~0.384 seconds)
- 6,000 elements (~0.544 seconds)
- 7,000 elements (~0.740 seconds)
- 8,000 elements (~0.985 seconds)
- 9,000 elements (~1.219 seconds)
- 10,000 elements (~1.504 seconds)
- 15,000 elements (~3.365 seconds)
- 20,000 elements

### Multi-threaded Implementation
Tests with arrays of size:
- 50,000 elements (~0.027 seconds)
- 60,000 elements (~0.027 seconds)
- 70,000 elements (~0.027 seconds)
- 80,000 elements (~0.027 seconds)
- 90,000 elements (~0.027 seconds)
- 100,000 elements (~0.027 seconds)
- 10,000 elements (~0.036 seconds)

## Implementation Details

### Thrust Implementation (thrust.cu)
- Uses NVIDIA's Thrust library for sorting
- Highly optimized parallel implementation
- Best performance for large datasets

### Single-threaded Implementation (singlethread.cu)
- Implements bubble sort algorithm
- Runs on a single CUDA thread
- Good for understanding basic CUDA operations
- Limited by single-thread performance

### Multi-threaded Implementation (multithread.cu)
- Implements parallel quicksort algorithm
- Uses multiple CUDA threads and blocks
- Good balance of implementation complexity and performance
- Efficient for medium to large datasets

## Error Handling
All implementations include CUDA error checking and will report any issues during execution.

## Cleanup
To remove all compiled executables:
```bash
make clean
```

## Performance Notes
- Thrust implementation shows the best performance for large datasets
- Single-threaded implementation is limited by the bubble sort algorithm and single thread execution
- Multi-threaded implementation shows consistent performance across different input sizes


