#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#if defined( TEST_DENSE_CUDA )
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/cusolver_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#elif defined( TEST_DENSE_HIP )
#    include <scfd/backend/hip.h>
#    include <scfd/external_libraries/hipblas_wrap_singleton_impl.h>
#    include <scfd/external_libraries/hipsolver_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#else
#    include <scfd/backend/serial_cpu.h>
#    include <scfd/backend/omp.h>
#    include <nmfd/operations/dense_operations_base.h>
#endif
#include <nmfd/detail/vector_wrap.h>

namespace
{
int  checks   = 0;
int  failures = 0;
void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

template <class T, class Backend>
void test_solve()
{
#if defined( TEST_DENSE_CUDA ) || defined( TEST_DENSE_HIP )
    using operations_type = nmfd::operations::dense_operations_cuda_hip<T, Backend>;
#else
    using operations_type = nmfd::operations::dense_operations<T, Backend>;
#endif
    operations_type operations( 3 );
    using space_type = typename operations_type::vector_space_type;
    space_type                            space( 3 );
    typename operations_type::matrix_type matrix;
    matrix.init( 3, 3 );
    nmfd::detail::vector_wrap<space_type, false, false> b( space ), x( space );
    b.init();
    b.start_use();
    x.init();
    x.start_use();
    const T entries[3][3] = { { 0, 2, 1 }, { 1, -2, -3 }, { -1, 1, 2 } };
    const T exact[3]      = { 1, -2, 3 };
    const T tolerance     = T( 100 ) * std::numeric_limits<T>::epsilon();
    auto    a_view        = matrix.create_view( false );
    auto    b_view        = ( *b ).create_view( false );
    for ( int i = 0; i < 3; ++i )
    {
        b_view( i ) = T( 0 );
        for ( int j = 0; j < 3; ++j )
        {
            a_view( i, j ) = entries[i][j];
            b_view( i ) += entries[i][j] * exact[j];
        }
    }
    a_view.release( true );
    b_view.release( true );
    for ( int repeat = 0; repeat < 3; ++repeat )
    {
        require( operations.solve( matrix, *b, *x ), "pivoted nonsymmetric solve" );
        auto result   = ( *x ).create_view( true );
        auto original = matrix.create_view( true );
        auto rhs      = ( *b ).create_view( true );
        for ( int i = 0; i < 3; ++i )
        {
            require( std::abs( result( i ) - exact[i] ) <= tolerance, "known solution" );
            T residual = -rhs( i );
            for ( int j = 0; j < 3; ++j )
            {
                require( original( i, j ) == entries[i][j], "matrix preserved" );
                residual += original( i, j ) * result( j );
            }
            require( std::abs( residual ) <= tolerance * T( 10 ), "original system residual" );
        }
    }
    require( operations.solve( matrix, *b, *b ), "in-place RHS solve" );
    auto in_place = ( *b ).create_view( true );
    for ( int i = 0; i < 3; ++i )
        require( std::abs( in_place( i ) - exact[i] ) <= tolerance, "in-place solution" );
    in_place.release( false );

    for ( int mode = 0; mode < 3; ++mode )
    {
        auto singular = matrix.create_view( false );
        for ( int i = 0; i < 3; ++i )
            for ( int j = 0; j < 3; ++j )
                singular( i, j ) = i == j ? T( 1 ) : T( 0 );
        singular( 2, 2 ) = mode == 0 ? T( 0 ) : std::numeric_limits<T>::quiet_NaN();
        if ( mode == 2 )
            singular( 2, 2 ) = std::numeric_limits<T>::infinity();
        singular.release( true );
        operations.assign_scalar( T( 7 ), *x );
        require( !operations.solve( matrix, *b, *x ), "singular/nonfinite matrix fails" );
        auto unchanged = ( *x ).create_view( true );
        for ( int i = 0; i < 3; ++i )
            require( unchanged( i ) == T( 7 ), "failed solve preserves output" );
    }
    operations.assign_zero_matrix( matrix );
    auto identity = matrix.create_view( false );
    for ( int i = 0; i < 3; ++i )
        identity( i, i ) = T( 1 );
    identity.release( true );
    operations.assign_scalar( std::numeric_limits<T>::infinity(), *b );
    require( !operations.solve( matrix, *b, *x ), "nonfinite RHS fails" );
    operations.assign_scalar( T( 2 ), *b );
    require( operations.solve( matrix, *b, *x ), "recovery after numerical failure" );
    typename operations_type::matrix_type rectangular;
    rectangular.init( 2, 3 );
    bool threw = false;
    try
    {
        operations.solve( rectangular, *b, *x );
    }
    catch ( const std::invalid_argument & )
    {
        threw = true;
    }
    require( threw, "invalid dimensions are a configuration error" );
}

template <class Backend>
void test_backend()
{
    test_solve<float, Backend>();
    test_solve<double, Backend>();
}
}

int main()
{
#if defined( TEST_DENSE_CUDA )
    scfd::backend::cuda<std::ptrdiff_t>::init_device();
    scfd::cublas_wrap   blas( false, true );
    scfd::cusolver_wrap solver( &blas, true );
    test_backend<scfd::backend::cuda<std::ptrdiff_t>>();
#elif defined( TEST_DENSE_HIP )
    scfd::backend::hip<std::ptrdiff_t>::init_device();
    scfd::hipblas_wrap   blas( false, true );
    scfd::hipsolver_wrap solver( &blas, true );
    test_backend<scfd::backend::hip<std::ptrdiff_t>>();
#else
    test_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>();
    test_backend<scfd::backend::omp<std::ptrdiff_t>>();
#endif
    std::cout << "Dense solves: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
