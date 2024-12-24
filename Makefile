# Makefile for CS456 Project - CUDA Sorting Implementations

# Compiler settings
CC=nvcc
CFLAGS=-O3

# Executables
THRUST=thrust
SINGLE=singlethread
MULTI=multithread

# Test parameters
SEED=42

# Default target
all: $(THRUST) $(SINGLE) $(MULTI)

# Compilation targets
$(THRUST): thrust.cu
	$(CC) $(CFLAGS) $< -o $@

$(SINGLE): singlethread.cu
	$(CC) $(CFLAGS) $< -o $@

$(MULTI): multithread.cu
	$(CC) $(CFLAGS) $< -o $@

# Run targets
run: run_thrust run_single run_multi

run_thrust: $(THRUST)
	@echo "Running Thrust implementation:"
	@echo "Testing with 10,000,000 elements:"
	./$(THRUST) 10000000 $(SEED)
	@echo "Testing with 20,000,000 elements:"
	./$(THRUST) 20000000 $(SEED)
	@echo "Testing with 30,000,000 elements:"
	./$(THRUST) 30000000 $(SEED)
	@echo "Testing with 40,000,000 elements:"
	./$(THRUST) 40000000 $(SEED)
	@echo "Testing with 50,000,000 elements:"
	./$(THRUST) 50000000 $(SEED)

run_single: $(SINGLE)
	@echo "Running Single-threaded implementation:"
	@echo "Testing with 2,000 elements:"
	./$(SINGLE) 2000 $(SEED)
	@echo "Testing with 5,000 elements:"
	./$(SINGLE) 5000 $(SEED)
	@echo "Testing with 6,000 elements:"
	./$(SINGLE) 6000 $(SEED)
	@echo "Testing with 7,000 elements:"
	./$(SINGLE) 7000 $(SEED)
	@echo "Testing with 8,000 elements:"
	./$(SINGLE) 8000 $(SEED)
	@echo "Testing with 9,000 elements:"
	./$(SINGLE) 9000 $(SEED)
	@echo "Testing with 10,000 elements:"
	./$(SINGLE) 10000 $(SEED)
	@echo "Testing with 15,000 elements:"
	./$(SINGLE) 15000 $(SEED)
	@echo "Testing with 20,000 elements:"
	./$(SINGLE) 20000 $(SEED)

run_multi: $(MULTI)
	@echo "Running Multi-threaded implementation:"
	@echo "Testing with 50,000 elements:"
	./$(MULTI) 50000 $(SEED)
	@echo "Testing with 60,000 elements:"
	./$(MULTI) 60000 $(SEED)
	@echo "Testing with 70,000 elements:"
	./$(MULTI) 70000 $(SEED)
	@echo "Testing with 80,000 elements:"
	./$(MULTI) 80000 $(SEED)
	@echo "Testing with 90,000 elements:"
	./$(MULTI) 90000 $(SEED)
	@echo "Testing with 100,000 elements:"
	./$(MULTI) 100000 $(SEED)
	@echo "Testing with 10,000 elements:"
	./$(MULTI) 10000 $(SEED)

# Clean target
clean:
	rm -f $(THRUST) $(SINGLE) $(MULTI)

# Help target
help:
	@echo "Makefile for CS456 Project - CUDA Sorting Implementations"
	@echo "Usage:"
	@echo "  make          - Compile all programs"
	@echo "  make run      - Run all implementations with their test cases"
	@echo "  make run_thrust  - Run thrust implementation tests"
	@echo "  make run_single  - Run single-threaded implementation tests"
	@echo "  make run_multi   - Run multi-threaded implementation tests"
	@echo "  make clean    - Remove all compiled executables"
	@echo "  make help     - Show this help message"

.PHONY: all run run_thrust run_single run_multi clean help