#ifndef NMFD_OPERATIONS_TWO_BLOCK_ACCESS_H
#define NMFD_OPERATIONS_TWO_BLOCK_ACCESS_H

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <nmfd/operations/glued_matrix_operator.h>

namespace nmfd
{
namespace operations
{

// Product vectors and typed operator blocks require no retained access state.
template <class VectorSpace, class LinearOperator>
class two_block_access
{
public:
    using first_space_type  = typename VectorSpace::first_space_type;
    using second_space_type = typename VectorSpace::second_space_type;

    template <std::size_t I>
    static const auto &space( const VectorSpace &value )
    {
        static_assert( I < 2, "Two-block space index must be 0 or 1" );
        if constexpr ( I == 0 )
            return value.first_space();
        else
            return value.second_space();
    }

    template <std::size_t I, class Vector>
    static auto &component( Vector &value )
    {
        static_assert( I < 2, "Two-block component index must be 0 or 1" );
        if constexpr ( I == 0 )
            return value.first;
        else
            return value.second;
    }

    template <std::size_t I, std::size_t J>
    static decltype( auto ) block( const LinearOperator &op )
    {
        static_assert( I < 2 && J < 2, "Two-block matrix indices must be 0 or 1" );
        if constexpr ( I == 0 && J == 0 )
            return op.block_11();
        else if constexpr ( I == 0 && J == 1 )
            return op.block_12();
        else if constexpr ( I == 1 && J == 0 )
            return op.block_21();
        else
            return op.block_22();
    }

    template <std::size_t I, std::size_t J, class Input, class Output>
    bool apply_block( const LinearOperator &op, const Input &in, Output &out ) const
    {
        const auto &handle = block<I, J>( op );
        const auto &action = *handle;
        if constexpr ( std::is_void<decltype( action.apply( in, out ) )>::value )
        {
            action.apply( in, out );
            return true;
        }
        else
        {
            static_assert(
                std::is_convertible<decltype( action.apply( in, out ) ), bool>::value,
                "Block action must return void or a bool-convertible status"
            );
            return static_cast<bool>( action.apply( in, out ) );
        }
    }
};

// The operations must be the same computational dependency used by the glued operator.
template <class Operations, std::size_t N>
class two_block_access<
    glued_vector_space<typename Operations::vector_space_type, N>, glued_matrix_operator<Operations, N>>
{
    static_assert( N == 2, "Two-block access requires exactly two glued components" );
    using space_type    = glued_vector_space<typename Operations::vector_space_type, N>;
    using operator_type = glued_matrix_operator<Operations, N>;

public:
    using first_space_type  = typename Operations::vector_space_type;
    using second_space_type = first_space_type;

    explicit two_block_access( std::shared_ptr<Operations> operations ) : operations_( std::move( operations ) )
    {
        if ( !operations_ )
            throw std::invalid_argument( "Glued block access requires operations" );
    }

    template <std::size_t I>
    static const auto &space( const space_type &value )
    {
        static_assert( I < 2, "Two-block space index must be 0 or 1" );
        return value.space_comp( I );
    }

    template <std::size_t I, class Vector>
    static auto &component( Vector &value )
    {
        static_assert( I < 2, "Two-block component index must be 0 or 1" );
        return value.comp( I );
    }

    template <std::size_t I, std::size_t J>
    static std::shared_ptr<const typename Operations::matrix_type> block( const operator_type &op )
    {
        static_assert( I < 2 && J < 2, "Two-block matrix indices must be 0 or 1" );
        return op.comp_ptr( I, J );
    }

    template <std::size_t I, std::size_t J>
    bool apply_block(
        const operator_type &op, const typename Operations::vector_type &in, typename Operations::vector_type &out
    ) const
    {
        static_assert( I < 2 && J < 2, "Two-block matrix indices must be 0 or 1" );
        using T = typename Operations::scalar_type;
        operations_->assign_scalar( T( 0 ), out );
        if constexpr ( std::is_void<decltype( operations_->add_matrix_vector_prod(
                           T( 1 ), op.comp( I, J ), in, T( 1 ), out
                       ) )>::value )
        {
            operations_->add_matrix_vector_prod( T( 1 ), op.comp( I, J ), in, T( 1 ), out );
            return true;
        }
        else
        {
            static_assert(
                std::is_convertible<
                    decltype( operations_->add_matrix_vector_prod( T( 1 ), op.comp( I, J ), in, T( 1 ), out ) ),
                    bool>::value,
                "Matrix action must return void or a bool-convertible status"
            );
            return static_cast<bool>( operations_->add_matrix_vector_prod( T( 1 ), op.comp( I, J ), in, T( 1 ), out ) );
        }
    }

private:
    std::shared_ptr<Operations> operations_;
};

}
}

#endif
