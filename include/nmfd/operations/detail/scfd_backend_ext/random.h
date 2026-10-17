#ifndef __COMMON_SCFD_BACKEND_EXT_RANDOM_H__
#define __COMMON_SCFD_BACKEND_EXT_RANDOM_H__

#include <cstddef>

#include <scfd/utils/device_tag.h>

#include <nmfd/operations/detail/scfd_backend_ext/math.h>

namespace nmfd
{
namespace operations
{
namespace detail
{
namespace scfd_backend_ext
{

template <class Backend, class T, class Ordinal>
struct random_fill
{
    template <class ForEach>
    static void uniform01( ForEach &for_each, Ordinal size, T *output, std::size_t seed )
    {
        for_each(
            [=] __DEVICE_TAG__( Ordinal i ) {
                output[i] = scalar_traits<Backend, T>::random_scalar( static_cast<std::size_t>( i ), seed );
            },
            size
        );
        for_each.wait();
    }
};

} // namespace scfd_backend_ext
} // namespace detail
} // namespace operations
} // namespace nmfd

#endif
