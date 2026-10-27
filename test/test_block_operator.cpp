#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

#include <nmfd/operations/block_operator.h>
#include <nmfd/operations/product_vector_space.h>
#include <nmfd/operations/static_vector_space.h>
#include <scfd/static_mat/mat.h>
#include <scfd/static_vec/vec.h>

namespace
{

template <int Rows, int Columns>
struct checked_block
{
    scfd::static_mat::mat<double, Rows, Columns> matrix;
    bool                                         fail = false;

    bool apply( const scfd::static_vec::vec<double, Columns> &in, scfd::static_vec::vec<double, Rows> &out ) const
    {
        if ( fail )
            return false;
        out = matrix * in;
        return true;
    }
};

struct zero_block
{
    void apply( const scfd::static_vec::vec<double, 1> &, scfd::static_vec::vec<double, 1> &out ) const
    {
        out[0] = 0.;
    }
};

}

int main()
{
    using first_type    = nmfd::operations::static_vector_space<double, 2, scfd::static_vec::vec<double, 2>>;
    using second_type   = nmfd::operations::static_vector_space<double, 1, scfd::static_vec::vec<double, 1>>;
    using space_type    = nmfd::operations::product_vector_space<first_type, second_type>;
    using operator_type = nmfd::operations::block_operator<
        space_type, checked_block<2, 2>, checked_block<2, 1>, checked_block<1, 2>, zero_block>;

    // Nonsymmetric 3x3 system with E=0; no scalar-border elimination is needed.
    auto space = std::make_shared<space_type>( std::make_shared<first_type>(), std::make_shared<second_type>() );
    auto a     = std::make_shared<checked_block<2, 2>>();
    auto b     = std::make_shared<checked_block<2, 1>>();
    auto c     = std::make_shared<checked_block<1, 2>>();
    auto e     = std::make_shared<zero_block>();
    a->matrix  = { 2., 1., -1., 3. };
    b->matrix  = { 4., -2. };
    c->matrix  = { 3., 5. };
    std::weak_ptr<space_type>          weak_space = space;
    std::weak_ptr<checked_block<2, 2>> weak_a     = a;
    int                                checks = 0, failures = 0;
    const auto                         require = [&]( bool condition, const char *message ) {
        ++checks;
        if ( !condition )
        {
            ++failures;
            std::cerr << "FAIL " << message << '\n';
        }
    };
    for ( int missing = 0; missing < 5; ++missing )
    {
        bool rejected = false;
        try
        {
            operator_type invalid(
                missing == 0 ? nullptr : space, missing == 1 ? nullptr : a, missing == 2 ? nullptr : b,
                missing == 3 ? nullptr : c, missing == 4 ? nullptr : e
            );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "null dependencies are rejected before workspace activation" );
    }
    {
        operator_type           op( space, a, b, c, e );
        space_type::vector_type in{ { 1., -2. }, { 3. } }, out;
        require( op.apply( in, out ), "void and bool block actions" );
        require( out.first[0] == 12. && out.first[1] == -13. && out.second[0] == -7., "assembled action" );
        require( op.block_11() == a && op.block_12() == b && op.block_21() == c && op.block_22() == e, "typed blocks" );
        require( op.get_dom_space() == space && op.get_im_space() == space, "domain and image" );
        for ( auto *fail : { &a->fail, &b->fail, &c->fail } )
        {
            *fail = true;
            require( !op.apply( in, out ), "block failure propagates" );
            *fail = false;
        }
        a->matrix( 0, 0 ) = 4.;
        require( op.apply( in, out ) && out.first[0] == 14., "changed coefficients use unchanged block handles" );
        bool rejected = false;
        try
        {
            op.apply( in, in );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected, "aliasing is explicitly rejected" );
        a.reset();
        b.reset();
        c.reset();
        e.reset();
        space.reset();
        require( !weak_space.expired() && !weak_a.expired(), "operator retains dependencies" );
    }
    require( weak_space.expired() && weak_a.expired(), "dependencies release without cycles" );
    std::cout << "Block operator: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
