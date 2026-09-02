CXX = g++
NVCC = nvcc

CXXFLAGS = -O2 -Wall
OMPFLAGS = -fopenmp
NVCCFLAGS = -O2 -arch=sm_80

BIN_DIR = bin

all: create_bin_dir $(BIN_DIR)/matrix $(BIN_DIR)/matrixMultiprocessing $(BIN_DIR)/matrixCuda

create_bin_dir:
	@mkdir -p $(BIN_DIR)

# Sequencial
$(BIN_DIR)/matrix: matrix.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# OpenMP
$(BIN_DIR)/matrixMultiprocessing: matrixMultiprocessing.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $< -o $@

# CUDA (GPU)
$(BIN_DIR)/matrixCuda: matrixCuda.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

matrix: $(BIN_DIR)/matrix
matrixMultiprocessing: $(BIN_DIR)/matrixMultiprocessing
matrixCuda: $(BIN_DIR)/matrixCuda

clean:
	rm -rf $(BIN_DIR)

.PHONY: all clean create_bin_dir matrix matrixMultiprocessing matrixCuda