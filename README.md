# NMFD-operations

C++17 vector operations, spaces, operator utilities, compensated reductions,
small dense algebra, and Matrix Market/CSR utilities. SCFD supplies storage and
serial, OpenMP, CUDA, and HIP backend primitives. No Newton or application headers
are required by the standalone include tree.

```sh
git clone --recurse-submodules https://github.com/evstigneevnm/NMFD-operations.git
cd NMFD-operations
make -j2 test-cpu
make check-headers test-multiprecision BOOST_ROOT=/usr/local/boost
make -j2 test-cuda CONFIG_FILE=build_configs/config_cuda.inc CUDA_ARCH=75
```

`CONFIG_FILE`, `BUILD_DIR`, `SCFD_ROOT`, `BOOST_ROOT`, and compiler flags can be
overridden. CPU tests need a C++17 compiler and OpenMP. Multiprecision tests need
Boost headers; LAPACK and device dense solvers are optional consumers' dependencies.
CUDA tests require a toolkit and a GPU matching `CUDA_ARCH`. HIP headers are
preserved but HIP execution has not been validated in this extraction.

Public headers retain `nmfd/operations/...` paths. Algebra-only
`dense_vector_operations` uses passed-vector extents; `dense_vector_space` adds
allocation and dimensions. `scfd_vector_space` preserves activation-time allocation
and normalized norms. The global `scfd_vector_operations` alias is retained for
existing callers. Instances own mutable scratch and are not concurrently reusable.

Missing SCFD scalar/complex traits, indexed reductions, deterministic random fill,
and scoped host-view write-back live in internal helpers. Native SCFD scalar traits
supply basic real math. Compensated GPU reductions require the kernel object built
by the Makefile; ordinary header-only consumers do not need it on CPU.

`nmfd/detail/vector_wrap.h` reuses the established NMFD wrapper implementation.
Solver repositories still carry their older headers until their dependency
migration; do not mix incompatible operations trees by changing include order.

`glued_vector`, `glued_vector_operations`, and `glued_vector_space` form homogeneous
fixed-count block products. Component dimensions may differ. Algebra and factories
use the same NMFD hierarchy as dense spaces; complex products conjugate the first
argument, and norms are real. The historical `norm2_sq` metric sums each component's
normalized squared norm, rather than dividing by the combined dimension.

`glued_matrix_operator` applies rectangular blocks and validates domain/image
dimensions at setup. It retains its operations and blocks through shared ownership.
Repeated `apply()` calls reuse cached spaces and do not allocate geometry. Inputs
and outputs must not overlap. Coefficient-only updates reuse the geometry; call
`prepare()` after changing block layouts through shared aliases. Mutable `comp()`
access invalidates the cache automatically. Instances are not concurrently reusable
while being modified.

Historical project-dependent HYPRE adapters remain in `legacy/`, outside the public
include tree. Other inherited operator utilities are retained without claiming that
all their optional methods are implemented.

CI separates algebra, hierarchy, glued operations, I/O, multiprecision, and sanitizer checks.
The optional CUDA dispatch job requires a self-hosted runner labeled `cuda`.
Preserve copyright notices and the repository's GPLv2 license.
