#ifndef __COMMON_SCFD_BACKEND_EXT_MATH_H__
#define __COMMON_SCFD_BACKEND_EXT_MATH_H__

#include <scfd/utils/scalar_traits.h>
#include <nmfd/operations/detail/scfd_backend_ext/complex.h>

namespace nmfd
{
namespace operations
{
namespace detail
{
namespace scfd_backend_ext
{

// Backend selection belongs to operations; scalar semantics belong to SCFD.
template <class Backend, class T>
using math = scfd::utils::scalar_traits<T>;

template <class Backend, class T>
using scalar_traits = scfd::utils::scalar_traits<T>;

template <class T>
struct max_op
{
    __DEVICE_TAG__ T operator()( const T &a, const T &b ) const
    {
        return a > b ? a : b;
    }
};

}
}
}
}

#endif
