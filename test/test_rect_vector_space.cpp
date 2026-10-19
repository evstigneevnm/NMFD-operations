#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/rect_vector_space.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/backend/omp.h>
#ifdef TEST_RECT_CUDA
#    include <scfd/backend/cuda.h>
#endif
#ifdef TEST_RECT_MPI
#    include <scfd/communication/mpi_wrap.h>
#else
#    include <scfd/communication/trivial_platform.h>
#endif

namespace
{

unsigned checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class Backend, class T, class Comm>
void test_space( Comm comm )
{
    using space_type                                = nmfd::rect_vector_space<T, 2, 2, Backend, Comm>;
    using vector_type                               = typename space_type::vector_type;
    using idx_type                                  = typename space_type::idx_nd_type;
    const T                               tolerance = T( 128 ) * std::numeric_limits<T>::epsilon();
    const int                             nx        = comm.myid + 2;
    space_type                            space( idx_type{ nx, 3 }, comm, false, 1, 2 );
    nmfd::detail::vector_wrap<space_type> x( space ), y( space );
    {
        typename vector_type::view_type view( *x, false );
        for ( int i = -1; i <= nx; ++i )
            for ( int j = -1; j <= 3; ++j )
                for ( int c = 0; c < 2; ++c )
                    view( i, j, c ) = std::numeric_limits<T>::quiet_NaN();
        view.release();
    }
    space.assign_scalar( T( comm.myid + 1 ), *x );
    space.assign_scalar( T( -2 ), *y );
    T sum = 0, squared = 0, count = 0;
    for ( int rank = 0; rank < comm.num_procs; ++rank )
    {
        const T entries = T( ( rank + 2 ) * 3 * 2 );
        count += entries;
        sum += entries * T( rank + 1 );
        squared += entries * T( ( rank + 1 ) * ( rank + 1 ) );
    }
    require( space.get_stencil() == 1 && space.get_max_stencil_order() == 2, "Halo metadata" );
    require( ( *x ).size() == ( nx + 2 ) * 5, "Halo allocation" );
    require( ( *x ).template get_index0<0>() == -1 && ( *x ).template get_index0<1>() == -1, "Halo origins" );
    require( space.is_valid_number( *x ), "Ghost poison must not affect validity" );
    require( std::abs( space.sum( *x ) - sum ) <= tolerance * sum, "Global sum" );
    require( std::abs( space.scalar_prod( *x, *y ) + T( 2 ) * sum ) <= tolerance * sum, "Global dot" );
    require( std::abs( space.norm_sq( *x ) - squared ) <= tolerance * squared, "Global norm squared" );
    require( std::abs( space.norm( *x ) - std::sqrt( squared ) ) <= tolerance * squared, "Global norm" );
    require( std::abs( space.norm2_sq( *x ) - squared / count ) <= tolerance * squared, "Global normalization" );
    require( std::abs( space.norm_l2( *x ) - std::sqrt( squared / count ) ) <= tolerance * squared, "Global RMS" );
    require( std::abs( space.asum( *y ) - T( 2 ) * count ) <= tolerance * count, "Global absolute sum" );
    require( space.norm_inf( *x ) == T( comm.num_procs ), "Global maximum" );
    space.assign_lin_comb( T( 2 ), *x, T( 3 ), *y, *y );
    require(
        std::abs( space.sum( *y ) - ( T( 2 ) * sum - T( 6 ) * count ) ) <= tolerance * count, "Local affine update"
    );
    {
        typename vector_type::view_type view( *x, true );
        require( std::isnan( view( -1, 0, 0 ) ), "Algebra must not overwrite ghosts" );
        if ( comm.myid == comm.num_procs - 1 )
            view( 0, 0, 0 ) = std::numeric_limits<T>::infinity();
        view.release();
    }
    require( !space.is_valid_number( *x ), "One invalid rank invalidates every rank" );
    space.assign_scalar( std::numeric_limits<T>::max(), *x );
    require( space.is_valid_number( *x ), "Finite entries remain valid when their sum overflows" );

    space_type                            empty( idx_type{ comm.myid == 0 ? 0 : nx, 3 }, comm );
    nmfd::detail::vector_wrap<space_type> z( empty );
    empty.assign_scalar( T( 2 ), *z );
    require( empty.is_valid_number( *z ), "Empty rank participates in validity collective" );
    require( empty.norm_inf( *z ) == T( comm.num_procs == 1 ? 0 : 2 ), "Empty rank maximum" );
    require( empty.norm_l2( *z ) == T( comm.num_procs == 1 ? 0 : 2 ), "Empty rank normalization" );
    space_type                            all_empty( idx_type{ 0, 3 }, comm );
    nmfd::detail::vector_wrap<space_type> none( all_empty );
    require( all_empty.sum( *none ) == 0 && all_empty.norm_l2( *none ) == 0, "Globally empty space" );

    bool threw = false;
    try
    {
        space_type invalid( idx_type{ comm.myid == 0 ? -1 : nx, 3 }, comm );
    }
    catch ( const std::invalid_argument & )
    {
        threw = true;
    }
    require( threw, "Invalid local range is rejected on every rank" );
    threw = false;
    try
    {
        space_type invalid( idx_type{ std::numeric_limits<int>::max(), 2 }, comm );
    }
    catch ( const std::invalid_argument & )
    {
        threw = true;
    }
    require( threw, "Reject overflowing storage before allocation" );
}

[[maybe_unused]]
void test_compatibility()
{
    using backend_type = scfd::backend::serial_cpu<>;
    nmfd::rect_vector_space<double, 2, 1, backend_type, int> old_ordinal( { 2, 3 }, true );
    nmfd::detail::vector_wrap<decltype( old_ordinal )>       x( old_ordinal );
    old_ordinal.assign_scalar( 3., *x );
    require( old_ordinal.sum( *x ) == 18., "Ordinal-only compatibility" );
    const scfd::communication::trivial_comm<backend_type::memory_type> comm;
    const double                                                       input[]   = { 2., -3. };
    double                                                             output[2] = {};
    comm.all_reduce_max( input, output, 2 );
    require( output[0] == input[0] && output[1] == input[1], "Trivial maximum identity" );
    comm.all_reduce_min( input, output, 2 );
    require(
        output[0] == input[0] && output[1] == input[1] && comm.all_reduce_min( 4. ) == 4., "Trivial minimum identity"
    );
}

}

int main( int argc, char *argv[] )
{
    try
    {
#ifdef TEST_RECT_MPI
        scfd::communication::mpi_wrap platform( argc, argv );
        const auto                    comm = platform.comm_world();
        test_space<scfd::backend::serial_cpu<>, float>( comm );
        test_space<scfd::backend::serial_cpu<>, double>( comm );
        test_space<scfd::backend::omp<>, double>( comm );
#elif defined( TEST_RECT_CUDA )
        using backend_type = scfd::backend::cuda<>;
        backend_type::init_device();
        scfd::communication::trivial_platform<typename backend_type::memory_type> platform( argc, argv );
        test_space<backend_type, float>( platform.comm_world() );
        test_space<backend_type, double>( platform.comm_world() );
#else
        scfd::communication::trivial_platform<scfd::backend::serial_cpu<>::memory_type> platform( argc, argv );
        test_space<scfd::backend::serial_cpu<>, float>( platform.comm_world() );
        test_space<scfd::backend::serial_cpu<>, double>( platform.comm_world() );
        test_space<scfd::backend::omp<>, double>( platform.comm_world() );
        test_compatibility();
#endif
        std::cout << "Rectangular space: " << checks << " checks passed per rank\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
