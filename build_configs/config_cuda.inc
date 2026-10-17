CXX = g++
CXXFLAGS = -std=c++17 -O2
OPENMP_FLAGS = -fopenmp
CUDA_ROOT ?= /usr/local/cuda
NVCC = $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 75
NVCCFLAGS = -std=c++17 -O2 --extended-lambda -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
