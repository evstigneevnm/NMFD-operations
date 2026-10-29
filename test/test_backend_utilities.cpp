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
void test_complex_contract()
{
    using complex_type   = nmfd::operations::backend::complex_t<Backend, T>;
    using traits_type    = scfd::utils::scalar_traits<complex_type>;
    using space_type     = nmfd::operations::scfd_vector_space<Backend, complex_type>;
    using abs_space_type = nmfd::operations::scfd_vector_space<Backend, T>;
    static_assert( std::is_same_v<typename space_type::norm_type, typename traits_type::abs_type> );
    static_assert( std::is_same_v<typename space_type::vector_of_abs_type, typename abs_space_type::vector_type> );
    space_type                                space( 2 );
    abs_space_type                            abs_space( 2 );
    nmfd::detail::vector_wrap<space_type>     x( space ), y( space );
    nmfd::detail::vector_wrap<abs_space_type> magnitudes( abs_space );
    x.start_use();
    y.start_use();
    magnitudes.start_use();
    const complex_type values[] = { traits_type::make( T( 3 ), T( 4 ) ), traits_type::make( T( -5 ), T( 12 ) ) };
    space.set( values, *x );
    space.make_abs_copy( *x, *magnitudes );
    require( abs_space.get_value_at_point( 0, *magnitudes ) == T( 5 ) );
    require( abs_space.get_value_at_point( 1, *magnitudes ) == T( 13 ) );
    require( space.get_value_at_point( 0, *x ) == values[0] && space.get_value_at_point( 1, *x ) == values[1] );
    for ( bool high_precision : { false, true } )
    {
        if ( high_precision )
            space.use_high_precision();
        require( space.asum( *x ) == T( 24 ) );
        require( space.norm_sq( *x ) == T( 194 ) );
        const auto dot = space.scalar_prod( *x, *x );
        require( traits_type::real( dot ) == T( 194 ) && traits_type::imag( dot ) == T( 0 ) );
        // Preserve the existing component-sum conventions during trait consolidation.
        require( space.norm1( *x ) == T( 24 ) && space.norm_inf( *x ) == T( 17 ) );
    }
    space.assign_scalar( traits_type::make( T( -2 ), T( 1 ) ), *y );
    const auto dot = space.scalar_prod( *x, *y );
    require( traits_type::real( dot ) == T( 20 ) && traits_type::imag( dot ) == T( 30 ) );
    space.assign_scalar( traits_type::zero(), *x );
    space.make_abs_copy( *x, *magnitudes );
    require( abs_space.norm( *magnitudes ) == T( 0 ) );
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
#ifndef TEST_BACKEND_CUDA
        test_complex_contract<scfd::backend::serial_cpu<std::ptrdiff_t>, float>();
        test_complex_contract<scfd::backend::serial_cpu<std::ptrdiff_t>, double>();
#endif
        test_complex_contract<backend_type, float>();
        test_complex_contract<backend_type, double>();
        std::printf( "Public backend utilities: %u checks, 0 failures\n", checks );
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::fprintf( stderr, "%s\n", error.what() );
        return 1;
    }
}
