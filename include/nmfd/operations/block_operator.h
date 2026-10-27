#ifndef NMFD_OPERATIONS_BLOCK_OPERATOR_H
#define NMFD_OPERATIONS_BLOCK_OPERATOR_H

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <nmfd/detail/vector_wrap.h>

namespace nmfd
{
namespace operations
{

// Matrix-free [A B; C E] on a product space. Blocks may return void or bool.
template <class ProductSpace, class A, class B, class C, class E>
class block_operator
{
public:
    using vector_space_type = ProductSpace;
    using scalar_type       = typename ProductSpace::scalar_type;
    using vector_type       = typename ProductSpace::vector_type;

    block_operator(
        std::shared_ptr<ProductSpace> space, std::shared_ptr<const A> a, std::shared_ptr<const B> b,
        std::shared_ptr<const C> c, std::shared_ptr<const E> e
    )
        : space_( space ? std::move( space ) : throw std::invalid_argument( "Block operator: null space" ) ),
          a_( std::move( a ) ), b_( std::move( b ) ), c_( std::move( c ) ), e_( std::move( e ) ), work_( *space_ )
    {
        if ( !a_ || !b_ || !c_ || !e_ )
            throw std::invalid_argument( "Block operator: null block" );
        work_.init();
        work_.start_use();
    }

    const std::shared_ptr<const A> &block_11() const
    {
        return a_;
    }
    const std::shared_ptr<const B> &block_12() const
    {
        return b_;
    }
    const std::shared_ptr<const C> &block_21() const
    {
        return c_;
    }
    const std::shared_ptr<const E> &block_22() const
    {
        return e_;
    }

    const std::shared_ptr<ProductSpace> &get_dom_space() const
    {
        return space_;
    }
    const std::shared_ptr<ProductSpace> &get_im_space() const
    {
        return space_;
    }

    // Input and output must not alias. A failed action leaves output unspecified.
    bool apply( const vector_type &in, vector_type &out ) const
    {
        if ( &in == &out )
            throw std::invalid_argument( "Block operator requires distinct input and output" );
        if ( !apply_block( *a_, in.first, out.first ) || !apply_block( *b_, in.second, ( *work_ ).first ) ||
             !apply_block( *c_, in.first, out.second ) || !apply_block( *e_, in.second, ( *work_ ).second ) )
            return false;
        space_->first_space().add_lin_comb( scalar_type( 1 ), ( *work_ ).first, scalar_type( 1 ), out.first );
        space_->second_space().add_lin_comb( scalar_type( 1 ), ( *work_ ).second, scalar_type( 1 ), out.second );
        return true;
    }

private:
    template <class Block, class Input, class Output>
    static bool apply_block( const Block &block, const Input &in, Output &out )
    {
        if constexpr ( std::is_void<decltype( block.apply( in, out ) )>::value )
        {
            block.apply( in, out );
            return true;
        }
        else
            return static_cast<bool>( block.apply( in, out ) );
    }

    std::shared_ptr<ProductSpace>                                 space_;
    std::shared_ptr<const A>                                      a_;
    std::shared_ptr<const B>                                      b_;
    std::shared_ptr<const C>                                      c_;
    std::shared_ptr<const E>                                      e_;
    mutable nmfd::detail::vector_wrap<ProductSpace, false, false> work_;
};

}
}

#endif
