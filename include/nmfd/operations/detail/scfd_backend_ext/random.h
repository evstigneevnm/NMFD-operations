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

template <class T>
__DEVICE_TAG__ typename scfd::utils::scalar_traits<T>::abs_type random_value( std::size_t i, std::size_t seed )
{
    using abs_type    = typename scfd::utils::scalar_traits<T>::abs_type;
    std::size_t value = ( i + 1u ) * ( seed + 7919u );
    value ^= value >> 13u;
    value *= 1274126177u;
    return static_cast<abs_type>( value % 100000u ) / static_cast<abs_type>( 100000u );
}

template <class T>
__DEVICE_TAG__ T random_scalar( std::size_t i, std::size_t seed )
{
    using traits_type = scfd::utils::scalar_traits<T>;
    if constexpr ( traits_type::is_complex )
        return traits_type::make( random_value<T>( i, seed ), random_value<T>( i + 17u, seed + 23u ) );
    else
        return random_value<T>( i, seed );
}

template <class Backend, class T, class Ordinal>
struct random_fill
{
    template <class ForEach>
    static void uniform01( ForEach &for_each, Ordinal size, T *output, std::size_t seed )
    {
        for_each(
            [=] __DEVICE_TAG__( Ordinal i ) { output[i] = random_scalar<T>( static_cast<std::size_t>( i ), seed ); },
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
