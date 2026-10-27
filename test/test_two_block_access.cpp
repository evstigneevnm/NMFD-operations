#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/dense_operations_base.h>
#include <nmfd/operations/two_block_access.h>
#include <scfd/backend/serial_cpu.h>

namespace
{

using base_operations = nmfd::operations::dense_operations<double, scfd::backend::serial_cpu<>>;

struct checked_operations : base_operations
{
    bool fail = false;

    bool add_matrix_vector_prod(
        double alpha, const matrix_type &matrix, const vector_type &in, double beta, vector_type &out
    ) const
    {
        if ( fail )
            return false;
        base_operations::add_matrix_vector_prod( alpha, matrix, in, beta, out );
        return true;
    }
};

int checks = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

template <class Operations>
void check_access()
{
    using operator_type = nmfd::operations::glued_matrix_operator<Operations, 2>;
    using space_type    = typename operator_type::vector_space_type;
    using access_type   = nmfd::operations::two_block_access<space_type, operator_type>;
    static_assert( !std::is_default_constructible_v<access_type> );
    static_assert(
        std::is_same_v<
            decltype( access_type::template component<0>( std::declval<const typename space_type::vector_type &>() ) ),
            const typename Operations::vector_type &>
    );
    std::weak_ptr<Operations>  weak_operations;
    std::optional<access_type> access;
    {
        auto operations = std::make_shared<Operations>();
        weak_operations = operations;
        access.emplace( operations );
        typename operator_type::block_array_type blocks;
        for ( int i = 0; i < 2; ++i )
            for ( int j = 0; j < 2; ++j )
            {
                blocks[i][j]   = std::make_shared<typename Operations::matrix_type>();
                const int rows = i == 0 ? 2 : 1, cols = j == 0 ? 2 : 1;
                operations->init_matrix( rows, cols, *blocks[i][j] );
                auto matrix = blocks[i][j]->create_view( false );
                for ( int row = 0; row < rows; ++row )
                    for ( int col = 0; col < cols; ++col )
                        matrix( row, col ) = row + 1 + 2 * ( col + 1 ) + i + j;
                matrix.release( true );
            }
        operator_type                                       op( operations, std::move( blocks ) );
        auto                                                space = op.get_dom_space();
        nmfd::detail::vector_wrap<space_type, false, false> in( *space ), out( *space );
        in.init();
        in.start_use();
        out.init();
        out.start_use();
        space->assign_scalar( 3., *in );
        require( &access->template space<0>( *space ) == &space->space_comp( 0 ), "Component-space reference" );
        require( &access->template component<0>( *in ) == &( *in ).comp( 0 ), "No vector conversion" );
        require( access->template block<0, 0>( op ) == op.comp_ptr( 0, 0 ), "No matrix copy" );
        const auto owners = operations.use_count();
        require( access->template apply_block<0, 1>( op, ( *in ).comp( 1 ), ( *out ).comp( 0 ) ), "B action" );
        auto result = ( *out ).comp( 0 ).create_view( true );
        require( result( 0 ) == 12. && result( 1 ) == 15., "Rectangular B action values" );
        result.release( false );
        require( operations.use_count() == owners, "No retained handles during application" );
        if constexpr ( std::is_same_v<Operations, checked_operations> )
        {
            operations->fail = true;
            require(
                !access->template apply_block<0, 1>( op, ( *in ).comp( 1 ), ( *out ).comp( 0 ) ),
                "Boolean failure status"
            );
        }
    }
    require( !weak_operations.expired(), "Access retains operations after matrix destruction" );
    access.reset();
    require( weak_operations.expired(), "Access releases operations without a cycle" );
    bool rejected = false;
    try
    {
        access_type invalid( nullptr );
    }
    catch ( const std::invalid_argument & )
    {
        rejected = true;
    }
    require( rejected, "Null operations rejected" );
}

}

int main()
{
    try
    {
        check_access<base_operations>();
        check_access<checked_operations>();
        std::cout << "Two-block access: " << checks << " checks passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
