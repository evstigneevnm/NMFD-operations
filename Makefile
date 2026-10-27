CONFIG_FILE ?= build_configs/config_cpu.inc
include $(CONFIG_FILE)

BUILD_DIR ?= build
SCFD_ROOT ?= contrib/scfd
BOOST_ROOT ?=
CUDA_ROOT ?= /usr/local/cuda
NVCC ?= $(CUDA_ROOT)/bin/nvcc
HIPCC ?= hipcc
HIPFLAGS ?= -std=c++17 -O2
HIP_LANGUAGE ?= hip
HIP_EXTRA_OBJECTS ?=
CUDA_ARCH ?= 75
NVCCFLAGS ?= -std=c++17 -O2 --extended-lambda -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
OMP_NUM_THREADS ?= 4
EXTRA_FLAGS ?=
MPICXX ?= mpicxx
MPIEXEC ?= mpiexec
MPIEXEC_FLAGS ?=
MPI_NP ?= 2
MPI_CXXFLAGS ?=
MPI_LDFLAGS ?=
CPPFLAGS += -Iinclude -I$(SCFD_ROOT)/include
ifneq ($(strip $(BOOST_ROOT)),)
CPPFLAGS += -I$(BOOST_ROOT)/include
endif
HEADERS := $(shell find include test -name '*.h' -o -name '*.cuh')
CPU_TESTS = $(BUILD_DIR)/hierarchy.bin $(BUILD_DIR)/glued.bin $(BUILD_DIR)/algebra_float.bin $(BUILD_DIR)/algebra_double.bin $(BUILD_DIR)/product.bin $(BUILD_DIR)/product_lifetime.bin $(BUILD_DIR)/block_operator.bin $(BUILD_DIR)/two_block_access.bin $(BUILD_DIR)/matrix_market.bin $(BUILD_DIR)/small_dense.bin $(BUILD_DIR)/io.bin $(BUILD_DIR)/backend.bin $(BUILD_DIR)/dense.bin
CUDA_TESTS = $(BUILD_DIR)/hierarchy_cuda.bin $(BUILD_DIR)/glued_cuda.bin $(BUILD_DIR)/algebra_cuda_float.bin $(BUILD_DIR)/algebra_cuda_double.bin $(BUILD_DIR)/product_cuda.bin $(BUILD_DIR)/backend_cuda.bin $(BUILD_DIR)/dense_cuda.bin

.PHONY: all test test-cpu test-hierarchy test-glued test-algebra test-io test-multiprecision test-backend test-cuda check-headers
all: $(CPU_TESTS)
test: test-cpu
test-cpu: test-hierarchy test-glued test-algebra test-io test-backend test-rect test-dense

.PHONY: test-dense test-dense-cuda
$(BUILD_DIR)/dense.bin: test/test_dense_solve.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

test-dense: $(BUILD_DIR)/dense.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $<

$(BUILD_DIR)/dense_cuda.bin: test/test_dense_solve.cpp $(HEADERS) $(SCFD_ROOT)/include/scfd/external_libraries/cusolver_wrap.h $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DTEST_DENSE_CUDA -x cu -c $< -o $(BUILD_DIR)/dense_cuda.o
	$(NVCC) $(NVCCFLAGS) $(BUILD_DIR)/dense_cuda.o $(BUILD_DIR)/high_precision_cuda.o -lcublas -lcusolver -o $@

test-dense-cuda: $(BUILD_DIR)/dense_cuda.bin
	$<

.PHONY: test-rect test-rect-mpi
$(BUILD_DIR)/rect.bin: test/test_rect_vector_space.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT $< -o $@

$(BUILD_DIR)/rect_mpi.bin: test/test_rect_vector_space.cpp $(HEADERS) | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $(MPI_CXXFLAGS) \
		-DSCFD_ARRAYS_ENABLE_INDEX_SHIFT -DTEST_RECT_MPI $< $(MPI_LDFLAGS) -o $@

$(BUILD_DIR)/rect_cuda.bin: test/test_rect_vector_space.cpp $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DSCFD_ARRAYS_ENABLE_INDEX_SHIFT -DTEST_RECT_CUDA -x cu $< -o $@

test-rect: $(BUILD_DIR)/rect.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $<

test-rect-mpi: $(BUILD_DIR)/rect_mpi.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) -n $(MPI_NP) $<

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/hierarchy.bin: test/test_nmfd_vector_space_hierarchy.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/backend.bin: test/test_backend_utilities.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

test-backend: $(BUILD_DIR)/backend.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$(BUILD_DIR)/backend.bin

$(BUILD_DIR)/glued.bin: test/test_glued_operations.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/algebra_%.bin: test/test_scfd_vector_operations_cpu.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) -DSCALAR_TYPE=$* $< -o $@

$(BUILD_DIR)/product.bin: test/test_product_vector_space.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/product_lifetime.bin: test/test_product_vector_space_lifetime.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/block_operator.bin: test/test_block_operator.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

$(BUILD_DIR)/two_block_access.bin: test/test_two_block_access.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $< -o $@

.PHONY: test-block
test-block: $(BUILD_DIR)/block_operator.bin $(BUILD_DIR)/two_block_access.bin
	$(BUILD_DIR)/block_operator.bin
	$(BUILD_DIR)/two_block_access.bin

$(BUILD_DIR)/product_mpi.bin: test/test_product_vector_space_mpi.cpp $(HEADERS) | $(BUILD_DIR)
	$(MPICXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $(OPENMP_FLAGS) $(MPI_CXXFLAGS) $< $(MPI_LDFLAGS) -o $@

.PHONY: test-product test-product-mpi test-product-cuda test-product-hip
test-product: $(BUILD_DIR)/product.bin $(BUILD_DIR)/product_lifetime.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(BUILD_DIR)/product.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(BUILD_DIR)/product_lifetime.bin

test-product-mpi: $(BUILD_DIR)/product_mpi.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(MPIEXEC) $(MPIEXEC_FLAGS) -n $(MPI_NP) $<

$(BUILD_DIR)/matrix_market.bin: test/test_matrix_market.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $< -o $@

$(BUILD_DIR)/small_dense.bin: test/test_small_dense_linalg.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $< -o $@

$(BUILD_DIR)/io.bin: test/test_io.cpp test/io_header_link.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) test/test_io.cpp test/io_header_link.cpp -o $@

$(BUILD_DIR)/multiprecision.bin: test/test_cpu_vector_operations_var_prec.cpp $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) $< -o $@

test-hierarchy: $(BUILD_DIR)/hierarchy.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$(BUILD_DIR)/hierarchy.bin

test-glued: $(BUILD_DIR)/glued.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$(BUILD_DIR)/glued.bin

test-algebra: $(BUILD_DIR)/algebra_float.bin $(BUILD_DIR)/algebra_double.bin test-product test-block
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$(BUILD_DIR)/algebra_float.bin
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$(BUILD_DIR)/algebra_double.bin

test-io: $(BUILD_DIR)/matrix_market.bin $(BUILD_DIR)/small_dense.bin $(BUILD_DIR)/io.bin
	./$(BUILD_DIR)/matrix_market.bin
	./$(BUILD_DIR)/small_dense.bin
	./$(BUILD_DIR)/io.bin

test-multiprecision: $(BUILD_DIR)/multiprecision.bin
	./$(BUILD_DIR)/multiprecision.bin

$(BUILD_DIR)/high_precision_cuda.o: include/nmfd/operations/blas1/high_precision/cuda/gpu_reduction_ogita_kernels.cu $(HEADERS) | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -c $< -o $@

$(BUILD_DIR)/hierarchy_cuda.bin: test/test_nmfd_vector_space_hierarchy.cpp $(HEADERS) $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DTEST_VECTOR_BACKEND_CUDA -x cu -c $< -o $(BUILD_DIR)/hierarchy_cuda.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_FLAGS) $(BUILD_DIR)/hierarchy_cuda.o $(BUILD_DIR)/high_precision_cuda.o -o $@

$(BUILD_DIR)/backend_cuda.bin: test/test_backend_utilities.cpp $(HEADERS) $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DTEST_BACKEND_CUDA -x cu -c $< -o $(BUILD_DIR)/backend_cuda.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_FLAGS) $(BUILD_DIR)/backend_cuda.o $(BUILD_DIR)/high_precision_cuda.o -o $@

$(BUILD_DIR)/glued_cuda.bin: test/test_glued_operations.cpp $(HEADERS) $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DTEST_GLUED_CUDA -x cu -c $< -o $(BUILD_DIR)/glued_cuda.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_FLAGS) $(BUILD_DIR)/glued_cuda.o $(BUILD_DIR)/high_precision_cuda.o -lcublas -o $@

$(BUILD_DIR)/product_cuda.bin: test/test_product_vector_space.cpp $(HEADERS) $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DTEST_PRODUCT_CUDA -x cu -c $< -o $(BUILD_DIR)/product_cuda.o
	$(NVCC) $(NVCCFLAGS) $(EXTRA_FLAGS) $(BUILD_DIR)/product_cuda.o $(BUILD_DIR)/high_precision_cuda.o -o $@

test-product-cuda: $(BUILD_DIR)/product_cuda.bin
	$<

$(BUILD_DIR)/product_hip.bin: test/test_product_vector_space.cpp $(HEADERS) $(HIP_EXTRA_OBJECTS) | $(BUILD_DIR)
	$(HIPCC) $(CPPFLAGS) $(HIPFLAGS) $(EXTRA_FLAGS) -DTEST_PRODUCT_HIP -x $(HIP_LANGUAGE) -c $< -o $(BUILD_DIR)/product_hip.o
	$(HIPCC) $(HIPFLAGS) $(EXTRA_FLAGS) $(BUILD_DIR)/product_hip.o $(HIP_EXTRA_OBJECTS) -o $@

test-product-hip: $(BUILD_DIR)/product_hip.bin
	$<

$(BUILD_DIR)/algebra_cuda_%.bin: test/test_scfd_vector_operations.cu $(HEADERS) $(BUILD_DIR)/high_precision_cuda.o
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(EXTRA_FLAGS) -DSCALAR_TYPE=$* $< $(BUILD_DIR)/high_precision_cuda.o -o $@

test-cuda: $(CUDA_TESTS) $(BUILD_DIR)/rect_cuda.bin
	./$(BUILD_DIR)/product_cuda.bin
	./$(BUILD_DIR)/dense_cuda.bin
	./$(BUILD_DIR)/rect_cuda.bin
	./$(BUILD_DIR)/backend_cuda.bin
	./$(BUILD_DIR)/hierarchy_cuda.bin
	./$(BUILD_DIR)/glued_cuda.bin
	./$(BUILD_DIR)/algebra_cuda_float.bin
	./$(BUILD_DIR)/algebra_cuda_double.bin

check-headers:
	@set -e; for header in $$(find include -name '*.h' | sort); do \
	  case "$$header" in */cuda/*|*/hip/*|*/dense_operations_cuda_hip.h) continue;; esac; \
	  printf '#include <%s>\n' "$${header#include/}" | $(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) -x c++ -fsyntax-only -; \
	done
