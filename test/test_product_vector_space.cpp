#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <scfd/backend/omp.h>
#include <scfd/backend/serial_cpu.h>
#if defined( TEST_PRODUCT_CUDA )
#    include <scfd/backend/cuda.h>
#elif defined( TEST_PRODUCT_HIP )
#    include <scfd/backend/hip.h>
#endif

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/scfd_vector_operations.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <scfd/static_vec/vec.h>

namespace
{

std::size_t checks   = 0;
std::size_t failures = 0;

void require( bool condition, const std::string &message )
{
    ++checks;
    if ( !condition )
    {
        ++failures;
        std::cout << "FAIL " << message << '\n';
    }
}

void require_close( double actual, double expected, double tolerance, const std::string &message )
{
    require( std::abs( actual - expected ) <= tolerance, message );
}

template <class Backend>
void run_case( const std::string &label )
{
    using component_space_type = scfd_vector_operations<Backend, double>;
    using product_space_type   = nmfd::operations::two_block_vector_space<component_space_type>;
    using vector_type          = typename product_space_type::vector_type;
    using multivector_type     = typename product_space_type::multivector_type;

    component_space_type component_space( 3 );
    product_space_type   space( component_space, component_space );

    vector_type x;
    vector_type y;
    space.init_vector( x );
    space.init_vector( y );
    space.start_use_vector( x );
    space.start_use_vector( y );

    const std::vector<double> host_x{ 1.0, 2.0, 3.0, -4.0, 5.0, -6.0 };
    const std::vector<double> host_y{ 2.0, -1.0, 0.5, 3.0, 1.0, -2.0 };
    space.set( host_x.data(), x, host_x.size() );
    space.set( host_y.data(), y, host_y.size() );

    require( space.get_default_size() == 6, label + " flattened size" );
    require_close( space.norm_sq( x ), 91.0, 1.0e-13, label + " norm squared" );
    require_close( space.scalar_prod( x, y ), 6.5, 1.0e-13, label + " dot" );

    space.assign_lin_comb( 2.0, x, -1.0, y, y );
    std::vector<double> actual( 6 );
    space.get( y, actual.data(), actual.size() );
    const std::vector<double> expected{ 0.0, 5.0, 5.5, -11.0, 9.0, -10.0 };
    for ( std::size_t index = 0; index < actual.size(); ++index )
    {
        require_close(
            actual[index], expected[index], 1.0e-13, label + " linear combination " + std::to_string( index )
        );
    }

    multivector_type basis;
    space.init_multivector( basis, 2 );
    space.start_use_multivector( basis, 2 );
    space.assign( x, basis, 2, 0 );
    space.assign( y, basis, 2, 1 );
    space.assign_scalar( 0.0, y );
    space.add_lin_comb( 3.0, basis, 2, 0, 0.0, y );
    require_close( space.scalar_prod( basis, 2, 0, y ), 273.0, 1.0e-12, label + " multivector column operations" );

    space.stop_use_multivector( basis, 2 );
    space.free_multivector( basis, 2 );
    require( basis.empty(), label + " multivector release" );

    require( space.is_valid_number( x ), label + " finite product state" );
    for ( auto *component : { &x.first, &x.second } )
    {
        for ( double invalid : { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity() } )
        {
            component_space.set_value_at_point( invalid, 0, *component );
            require( !space.is_valid_number( x ), label + " nonfinite entry in either block" );
        }
        component_space.assign_scalar( 0.0, *component );
        require( space.is_valid_number( x ), label + " restored finite product state" );
    }

    space.stop_use_vector( y );
    space.stop_use_vector( x );
    space.free_vector( y );
    space.free_vector( x );
}

template <class Backend>
void run_heterogeneous_case( const std::string &label )
{
    using state_space_type = scfd_vector_operations<Backend, double>;
    using small_space_type = nmfd::operations::static_vector_space<double, 3, scfd::static_vec::vec<double, 3>>;
    using space_type       = nmfd::operations::product_vector_space<state_space_type, small_space_type>;
    using pair_type        = nmfd::operations::pair_vector_space<state_space_type, small_space_type>;
    static_assert( std::is_same_v<typename space_type::vector_type, typename pair_type::vector_type> );

    auto      state = std::make_shared<state_space_type>( 2 );
    auto      small = std::make_shared<small_space_type>();
    auto      space = std::make_shared<space_type>( state, small );
    pair_type pair( state, small );
    require( pair.first() == state && pair.second() == small, label + " owning pair accessors" );
    const auto &const_pair = pair;
    require( const_pair.first() == state && const_pair.second() == small, label + " const pair accessors" );

    nmfd::detail::vector_wrap<space_type, false, false> x( *space ), y( *space );
    x.init();
    x.start_use();
    y.init();
    y.start_use();
    const double input[]{ 1., -2., 3., -4., 5. };
    space->set( input, *x, 5 );
    double output[5]{};
    space->get( *x, output, 5 );
    for ( std::size_t i = 0; i < 5; ++i )
        require_close( output[i], input[i], 0., label + " heterogeneous host transfer" );
    require(
        space->get_default_size() == 5 && space->get_vector_size() == 5 && space->get_size( *x ) == 5,
        label + " heterogeneous dimensions"
    );
    require_close( space->norm_sq( *x ), 55., 1.e-12, label + " heterogeneous squared norm" );
    require_close( pair.norm_l2( *x ), std::sqrt( 55. ), 1.e-12, label + " legacy pair metric" );
    require_close( space->norm_l2( *x ), pair.norm_l2( *x ), 1.e-12, label + " product metric aliases" );
    require_close( space->norm2_sq( *x ), 55., 1.e-12, label + " unnormalized squared alias" );
    require_close( space->sum( *x ), 3., 1.e-12, label + " sum" );
    require_close( space->asum( *x ), 15., 1.e-12, label + " absolute sum" );
    require_close( space->norm_inf( *x ), 5., 1.e-12, label + " infinity norm" );
    space->assign( *x, *y );
    space->add_lin_comb( 2., *x, -1., *x, 0., *y );
    space->add_lin_comb( -1., *x, *y );
    require_close( space->norm( *y ), 0., 1.e-12, label + " aliased combinations" );
    space->assign( *x, *y );
    require_close( space->normalize( *y ), std::sqrt( 55. ), 1.e-12, label + " normalization factor" );
    require_close( space->norm( *y ), 1., 1.e-12, label + " unit norm" );
    space->set_value_at_point( 7., 2, *y );
    require_close( space->get_value_at_point( 2, *y ), 7., 0., label + " access across block boundary" );
    ( *y ).second[2] = std::numeric_limits<double>::infinity();
    require( !space->is_valid_number( *y ), label + " invalid small block" );

    typename space_type::vector_type a, b;
    space->init_vectors( a, b );
    space->start_use_vectors( a, b );
    space->assign_scalar( 2., a );
    space->assign( a, b );
    require_close( space->scalar_prod( a, b ), 20., 1.e-12, label + " batched lifecycle" );
    space->stop_use_vectors( a, b );
    space->free_vectors( a, b );

    using tail_type   = nmfd::operations::static_vector_space<double, 1, scfd::static_vec::vec<double, 1>>;
    using nested_type = nmfd::operations::product_vector_space<space_type, tail_type>;
    nested_type                                          nested( space, std::make_shared<tail_type>() );
    nmfd::detail::vector_wrap<nested_type, false, false> z( nested );
    z.init();
    z.start_use();
    nested.assign_scalar( 2., *z );
    require( nested.size() == 6, label + " nested dimensions" );
    require_close( nested.norm_sq( *z ), 24., 1.e-12, label + " nested metric" );
}

} // namespace

int main()
{
#if defined( TEST_PRODUCT_CUDA )
    scfd::backend::cuda<>::init_device();
    run_case<scfd::backend::cuda<>>( "cuda" );
    run_heterogeneous_case<scfd::backend::cuda<>>( "cuda" );
#elif defined( TEST_PRODUCT_HIP )
    scfd::backend::hip<>::init_device();
    run_case<scfd::backend::hip<>>( "hip" );
    run_heterogeneous_case<scfd::backend::hip<>>( "hip" );
#else
    run_case<scfd::backend::serial_cpu<std::ptrdiff_t>>( "serial" );
    run_case<scfd::backend::omp<std::ptrdiff_t>>( "omp" );
    run_heterogeneous_case<scfd::backend::serial_cpu<int>>( "serial" );
    run_heterogeneous_case<scfd::backend::omp<std::ptrdiff_t>>( "omp" );
#endif

    std::cout << "Checks: " << checks << ", failures: " << failures << '\n';
    if ( failures != 0 )
    {
        std::cout << "FAILED\n";
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
