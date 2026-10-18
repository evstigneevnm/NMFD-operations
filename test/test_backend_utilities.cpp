#include <cstdio>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <scfd/backend/serial_cpu.h>
#ifdef TEST_BACKEND_CUDA
#    include <scfd/backend/cuda.h>
#else
#    include <scfd/backend/omp.h>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/backend/complex.h>
#include <nmfd/operations/backend/host_view.h>
#include <nmfd/operations/backend/math.h>
#include <nmfd/operations/scfd_vector_space.h>

namespace
{

unsigned int checks = 0;

void require( bool passed )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( "Public backend utility check failed" );
}

template <class Backend, class T>
void test()
{
    using math_type      = nmfd::operations::backend::math<Backend, T>;
    using complex_type   = nmfd::operations::backend::complex_t<Backend, T>;
    using complex_traits = nmfd::operations::backend::complex_value_traits<complex_type>;
    using space_type     = nmfd::operations::scfd_vector_space<Backend, T>;
    using view_traits    = nmfd::operations::backend::host_view_traits<typename space_type::vector_type>;
    static_assert( nmfd::operations::backend::is_complex_type<complex_type>::value );
    static_assert( std::is_same_v<nmfd::operations::backend::complex_real_t<complex_type>, T> );

    require( math_type::sqrt( T( 4 ) ) == T( 2 ) && math_type::abs( T( -3 ) ) == T( 3 ) );
    require( math_type::sin( T( 0 ) ) == T( 0 ) && math_type::cos( T( 0 ) ) == T( 1 ) );
    require( math_type::is_finite( std::numeric_limits<T>::max() ) );
    require( !math_type::is_finite( std::numeric_limits<T>::quiet_NaN() ) );
    require( !math_type::is_finite( std::numeric_limits<T>::infinity() ) );
    const auto value = complex_traits::make( T( 3 ), T( 4 ) );
    require( complex_traits::real( value ) == T( 3 ) && complex_traits::imag( value ) == T( 4 ) );
    require( complex_traits::abs_sq( value ) == T( 25 ) );

    space_type                            space( 3 );
    nmfd::detail::vector_wrap<space_type> vector( space );
    vector.start_use();
    space.assign_scalar( T( 2 ), *vector );
    {
        typename view_traits::scoped_view_type view( *vector, true, false );
        require( view( 0 ) == T( 2 ) && view.size() == 3 );
    }
    {
        typename view_traits::scoped_view_type view( *vector, true, true );
        view( 0 ) = T( 5 );
    }
    require( space.get_value_at_point( 0, *vector ) == T( 5 ) );
}

}

int main()
{
    try
    {
#ifdef TEST_BACKEND_CUDA
        using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
        backend_type::init_device();
#else
        using backend_type = scfd::backend::omp<std::ptrdiff_t>;
#endif
        test<scfd::backend::serial_cpu<std::ptrdiff_t>, float>();
        test<scfd::backend::serial_cpu<std::ptrdiff_t>, double>();
        test<backend_type, float>();
        test<backend_type, double>();
        std::printf( "Public backend utilities: %u checks, 0 failures\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "%s\n", error.what() );
        return 1;
    }
}
