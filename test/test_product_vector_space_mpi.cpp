#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/rect_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <scfd/backend/serial_cpu.h>
#include <scfd/communication/mpi_wrap.h>
#include <scfd/static_vec/vec.h>

int main( int argc, char *argv[] )
{
    try
    {
        scfd::communication::mpi_wrap platform( argc, argv );
        auto                          comm = platform.comm_world();
        using state_type     = nmfd::rect_vector_space<double, 1, 1, scfd::backend::serial_cpu<>, decltype( comm )>;
        using parameter_type = nmfd::operations::static_vector_space<double, 2, scfd::static_vec::vec<double, 2>>;
        using space_type     = nmfd::operations::product_vector_space<state_type, parameter_type>;
        auto       state     = std::make_shared<state_type>( typename state_type::idx_nd_type{ comm.myid + 2 }, comm );
        auto       parameter = std::make_shared<parameter_type>();
        space_type space( state, parameter );
        nmfd::detail::vector_wrap<space_type, false, false> x( space );
        x.init();
        x.start_use();
        state->assign_scalar( double( comm.myid + 1 ), ( *x ).first );
        parameter->assign_scalar( 3., ( *x ).second );
        double squared = 18., sum = 6.;
        for ( int rank = 0; rank < comm.num_procs; ++rank )
        {
            squared += double( rank + 2 ) * ( rank + 1 ) * ( rank + 1 );
            sum += double( rank + 2 ) * ( rank + 1 );
        }
        const bool passed = space.size() == std::size_t( comm.myid + 4 ) && space.get_size( *x ) == space.size() &&
                            std::abs( space.scalar_prod( *x, *x ) - squared ) < 1.e-12 * squared &&
                            std::abs( space.norm_sq( *x ) - squared ) < 1.e-12 * squared &&
                            std::abs( space.sum( *x ) - sum ) < 1.e-12 * sum && space.is_valid_number( *x );
        if ( comm.all_reduce_sum( int( !passed ) ) != 0 )
            throw std::runtime_error( "Product reductions must count replicated parameters once" );
        if ( comm.myid == 0 )
            std::cout << "Product MPI: dimensions and global reductions passed on " << comm.num_procs << " ranks\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
