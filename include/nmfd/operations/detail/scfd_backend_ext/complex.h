#ifndef __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__
#define __COMMON_SCFD_BACKEND_EXT_COMPLEX_H__

#include <complex>
#include <type_traits>
#include <scfd/utils/scalar_traits.h>

#if defined( __CUDACC__ ) || defined( __HIPCC__ ) || defined( NMFD_OPERATIONS_ENABLE_THRUST_COMPLEX )
#    include <scfd/utils/thrust_complex_scalar_traits.h>
#    define COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX 1
#endif

#if defined( __CUDACC__ ) && !defined( __HIPCC__ )
#    include <scfd/utils/cucomplex_scalar_traits.h>
#    include <cufft.h>
#    define COMMON_SCFD_BACKEND_EXT_HAS_CUFFT_COMPLEX 1
#endif

namespace scfd
{
namespace backend
{
template <class Ordinal>
struct cuda;
template <class Ordinal>
struct hip;
template <class Ordinal>
struct omp;
template <class Ordinal>
struct serial_cpu;
}
}

namespace nmfd
{
namespace operations
{
namespace detail
{
namespace scfd_backend_ext
{

template <class Backend, class Real>
struct complex
{
    using type = std::complex<Real>;
};

template <class>
struct dependent_false : std::false_type
{
};

#ifdef COMMON_SCFD_BACKEND_EXT_HAS_THRUST_COMPLEX
template <class BackendOrdinal, class Real>
struct complex<scfd::backend::cuda<BackendOrdinal>, Real>
{
    using type = thrust::complex<Real>;
};
template <class BackendOrdinal, class Real>
struct complex<scfd::backend::hip<BackendOrdinal>, Real>
{
    using type = thrust::complex<Real>;
};
#else
template <class BackendOrdinal, class Real>
struct complex<scfd::backend::cuda<BackendOrdinal>, Real>
{
    static_assert( dependent_false<Real>::value, "CUDA complex type requires thrust::complex." );
};
template <class BackendOrdinal, class Real>
struct complex<scfd::backend::hip<BackendOrdinal>, Real>
{
    static_assert( dependent_false<Real>::value, "HIP complex type requires thrust::complex." );
};
#endif

template <class Backend, class Real>
using complex_t = typename complex<Backend, Real>::type;

template <class T>
struct is_complex_type : std::integral_constant<bool, scfd::utils::scalar_traits<T>::is_complex>
{
};

// Compatibility names only; arithmetic is implemented once, in SCFD.
template <class Complex>
struct complex_value_traits : scfd::utils::scalar_traits<Complex>
{
    static_assert( is_complex_type<Complex>::value, "Unsupported complex type." );
    using complex_type = Complex;
    using real_type    = typename scfd::utils::scalar_traits<Complex>::component_type;
};

template <class Complex>
using complex_real_t = typename complex_value_traits<Complex>::real_type;

}
}
}
}

#endif
