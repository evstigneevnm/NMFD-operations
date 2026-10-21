#ifndef __NMFD_OPERATIONS_BACKEND_BOOST_MULTIPRECISION_H__
#define __NMFD_OPERATIONS_BACKEND_BOOST_MULTIPRECISION_H__

#include <boost/multiprecision/cpp_bin_float.hpp>
#include <nmfd/operations/backend/math.h>

namespace nmfd
{
namespace operations
{
namespace detail
{
namespace scfd_backend_ext
{

// Optional host scalar support; native backend headers do not depend on Boost.
template <class Backend, class NumberBackend, boost::multiprecision::expression_template_option Expressions>
struct math<Backend, boost::multiprecision::number<NumberBackend, Expressions>>
{
    using T = boost::multiprecision::number<NumberBackend, Expressions>;
    static T zero()
    {
        return T( 0 );
    }
    static T abs( const T &x )
    {
        return boost::multiprecision::abs( x );
    }
    static bool is_finite( const T &x )
    {
        return boost::multiprecision::isfinite( x );
    }
    static T sqrt( const T &x )
    {
        return boost::multiprecision::sqrt( x );
    }
    static T hypot( const T &x, const T &y )
    {
        return boost::multiprecision::hypot( x, y );
    }
    static T sin( const T &x )
    {
        return boost::multiprecision::sin( x );
    }
    static T cos( const T &x )
    {
        return boost::multiprecision::cos( x );
    }
};

} // namespace scfd_backend_ext
} // namespace detail
} // namespace operations
} // namespace nmfd

#endif
