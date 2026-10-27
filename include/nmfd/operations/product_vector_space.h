#ifndef __NMFD_OPERATIONS_PRODUCT_VECTOR_SPACE_H__
#define __NMFD_OPERATIONS_PRODUCT_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace nmfd
{
namespace operations
{

namespace detail
{

template <class Space, class = void>
struct product_ordinal
{
    using type = std::ptrdiff_t;
};

template <class Space>
struct product_ordinal<Space, std::void_t<typename Space::ordinal_type>>
{
    using type = typename Space::ordinal_type;
};

template <class Space, class = void>
struct product_norm
{
    using type = typename Space::scalar_type;
};

template <class Space>
struct product_norm<Space, std::void_t<typename Space::norm_type>>
{
    using type = typename Space::norm_type;
};

}

template <class VectorSpace1, class VectorSpace2>
class product_vector_space
{
public:
    using first_space_type   = VectorSpace1;
    using second_space_type  = VectorSpace2;
    using scalar_type        = typename first_space_type::scalar_type;
    using norm_type          = typename detail::product_norm<first_space_type>::type;
    using ordinal_type       = typename detail::product_ordinal<first_space_type>::type;
    using Ord                = ordinal_type;
    using first_vector_type  = typename first_space_type::vector_type;
    using second_vector_type = typename second_space_type::vector_type;
    using vector_type        = std::pair<first_vector_type, second_vector_type>;
    using multivector_type   = std::vector<vector_type>;
    using scalar1_type       = scalar_type;
    using scalar2_type       = typename second_space_type::scalar_type;
    using vector1_type       = first_vector_type;
    using vector2_type       = second_vector_type;

    static_assert(
        std::is_same<scalar_type, typename second_space_type::scalar_type>::value,
        "product vector spaces must have the same scalar type"
    );
    static_assert(
        std::is_same<norm_type, typename detail::product_norm<second_space_type>::type>::value,
        "product vector spaces must have the same norm type"
    );
    // Compatibility path: both component spaces must outlive this borrowed space.
    product_vector_space( const first_space_type &first, const second_space_type &second )
        : first_( &first ), second_( &second )
    {
    }

    product_vector_space( std::shared_ptr<first_space_type> first, std::shared_ptr<second_space_type> second )
        : first_owner_( std::move( first ) ), second_owner_( std::move( second ) ), first_( first_owner_.get() ),
          second_( second_owner_.get() )
    {
        if ( !first_ || !second_ )
            throw std::invalid_argument( "Product vector space requires nonnull components" );
    }

    const first_space_type &first_space() const
    {
        return *first_;
    }

    const second_space_type &second_space() const
    {
        return *second_;
    }

    std::size_t first_size() const
    {
        return component_size( *first_, 0 );
    }

    std::size_t second_size() const
    {
        return component_size( *second_, 0 );
    }

    std::size_t get_default_size() const
    {
        return first_size() + second_size();
    }

    std::size_t size() const
    {
        return get_default_size();
    }

    std::size_t get_vector_size() const
    {
        return size();
    }

    std::size_t get_size( const vector_type &vector ) const
    {
        return first_->get_size( vector.first ) + second_->get_size( vector.second );
    }

    void init_vector( vector_type &vector ) const
    {
        first_->init_vector( vector.first );
        try
        {
            second_->init_vector( vector.second );
        }
        catch ( ... )
        {
            first_->free_vector( vector.first );
            throw;
        }
    }

    void start_use_vector( vector_type &vector ) const
    {
        first_->start_use_vector( vector.first );
        try
        {
            second_->start_use_vector( vector.second );
        }
        catch ( ... )
        {
            first_->stop_use_vector( vector.first );
            throw;
        }
    }

    void stop_use_vector( vector_type &vector ) const
    {
        second_->stop_use_vector( vector.second );
        first_->stop_use_vector( vector.first );
    }

    void free_vector( vector_type &vector ) const
    {
        second_->free_vector( vector.second );
        first_->free_vector( vector.first );
    }

    void init_vectors() const
    {
    }

    template <class... Vectors>
    void init_vectors( vector_type &first, Vectors &...rest ) const
    {
        init_vector( first );
        try
        {
            init_vectors( rest... );
        }
        catch ( ... )
        {
            free_vector( first );
            throw;
        }
    }

    void start_use_vectors() const
    {
    }

    template <class... Vectors>
    void start_use_vectors( vector_type &first, Vectors &...rest ) const
    {
        start_use_vector( first );
        try
        {
            start_use_vectors( rest... );
        }
        catch ( ... )
        {
            stop_use_vector( first );
            throw;
        }
    }

    template <class... Vectors>
    void stop_use_vectors( Vectors &...vectors ) const
    {
        ( stop_use_vector( vectors ), ... );
    }

    template <class... Vectors>
    void free_vectors( Vectors &...vectors ) const
    {
        ( free_vector( vectors ), ... );
    }

    void init_multivector( multivector_type &vectors, ordinal_type count ) const
    {
        const std::size_t size_count = checked_count( count );
        // As for init_vector(), existing elements must not own live component allocations.
        vectors.clear();
        vectors.resize( size_count );
        std::size_t initialized = 0;
        try
        {
            for ( ; initialized < size_count; ++initialized )
                init_vector( vectors[initialized] );
        }
        catch ( ... )
        {
            while ( initialized > 0 )
                free_vector( vectors[--initialized] );
            vectors.clear();
            throw;
        }
    }

    void start_use_multivector( multivector_type &vectors, ordinal_type count ) const
    {
        const std::size_t size_count = checked_count( count );
        check_multivector_size( vectors, size_count );
        std::size_t started = 0;
        try
        {
            for ( ; started < size_count; ++started )
                start_use_vector( vectors[started] );
        }
        catch ( ... )
        {
            while ( started > 0 )
                stop_use_vector( vectors[--started] );
            throw;
        }
    }

    void stop_use_multivector( multivector_type &vectors, ordinal_type count ) const
    {
        const std::size_t size_count = checked_count( count );
        check_multivector_size( vectors, size_count );
        for ( std::size_t index = size_count; index > 0; --index )
            stop_use_vector( vectors[index - 1] );
    }

    void free_multivector( multivector_type &vectors, ordinal_type count ) const
    {
        const std::size_t size_count = checked_count( count );
        check_multivector_size( vectors, size_count );
        for ( std::size_t index = size_count; index > 0; --index )
            free_vector( vectors[index - 1] );
        vectors.clear();
    }

    void assign_scalar( scalar_type value, vector_type &destination ) const
    {
        first_->assign_scalar( value, destination.first );
        second_->assign_scalar( value, destination.second );
    }

    void add_mul_scalar( scalar_type value, scalar_type scale, vector_type &vector ) const
    {
        first_->add_mul_scalar( value, scale, vector.first );
        second_->add_mul_scalar( value, scale, vector.second );
    }

    void assign( const vector_type &source, vector_type &destination ) const
    {
        first_->assign( source.first, destination.first );
        second_->assign( source.second, destination.second );
    }

    void assign_lin_comb( scalar_type source_scale, const vector_type &source, vector_type &destination ) const
    {
        first_->assign_lin_comb( source_scale, source.first, destination.first );
        second_->assign_lin_comb( source_scale, source.second, destination.second );
    }

    void assign_lin_comb(
        scalar_type first_scale, const vector_type &first, scalar_type second_scale, const vector_type &second,
        vector_type &destination
    ) const
    {
        first_->assign_lin_comb( first_scale, first.first, second_scale, second.first, destination.first );
        second_->assign_lin_comb( first_scale, first.second, second_scale, second.second, destination.second );
    }

    void add_lin_comb(
        scalar_type source_scale, const vector_type &source, scalar_type destination_scale, vector_type &destination
    ) const
    {
        first_->add_lin_comb( source_scale, source.first, destination_scale, destination.first );
        second_->add_lin_comb( source_scale, source.second, destination_scale, destination.second );
    }

    void add_lin_comb( scalar_type scale, const vector_type &source, vector_type &destination ) const
    {
        add_lin_comb( scale, source, scalar_type( 1 ), destination );
    }

    void add_lin_comb(
        scalar_type first_scale, const vector_type &first, scalar_type second_scale, const vector_type &second,
        scalar_type destination_scale, vector_type &destination
    ) const
    {
        first_->add_lin_comb(
            first_scale, first.first, second_scale, second.first, destination_scale, destination.first
        );
        second_->add_lin_comb(
            first_scale, first.second, second_scale, second.second, destination_scale, destination.second
        );
    }

    void scale( scalar_type value, vector_type &vector ) const
    {
        first_->scale( value, vector.first );
        second_->scale( value, vector.second );
    }

    scalar_type scalar_prod( const vector_type &first, const vector_type &second ) const
    {
        const auto first_value  = first_->scalar_prod( first.first, second.first );
        const auto second_value = second_->scalar_prod( first.second, second.second );
        return first_value + second_value;
    }

    scalar_type scalar_prod_l2( const vector_type &first, const vector_type &second ) const
    {
        return scalar_prod( first, second );
    }

    norm_type norm_sq( const vector_type &vector ) const
    {
        const auto first_value  = first_->norm_sq( vector.first );
        const auto second_value = second_->norm_sq( vector.second );
        return first_value + second_value;
    }

    norm_type norm( const vector_type &vector ) const
    {
        return std::sqrt( norm_sq( vector ) );
    }

    // Preserve the pair-space metric: these aliases do not normalize either block.
    norm_type norm2_sq( const vector_type &vector ) const
    {
        return norm_sq( vector );
    }

    norm_type norm_l2_sq( const vector_type &vector ) const
    {
        return norm_sq( vector );
    }

    norm_type norm_l2( const vector_type &vector ) const
    {
        return norm( vector );
    }

    norm_type norm2( const vector_type &vector ) const
    {
        return norm( vector );
    }

    scalar_type sum( const vector_type &vector ) const
    {
        const auto first_value  = first_->sum( vector.first );
        const auto second_value = second_->sum( vector.second );
        return first_value + second_value;
    }

    norm_type asum( const vector_type &vector ) const
    {
        const auto first_value  = first_->asum( vector.first );
        const auto second_value = second_->asum( vector.second );
        return first_value + second_value;
    }

    norm_type norm1( const vector_type &vector ) const
    {
        return asum( vector );
    }

    norm_type norm_l1( const vector_type &vector ) const
    {
        return asum( vector );
    }

    norm_type norm_inf( const vector_type &vector ) const
    {
        const auto first_value  = first_->norm_inf( vector.first );
        const auto second_value = second_->norm_inf( vector.second );
        if ( first_value != first_value )
            return first_value;
        if ( second_value != second_value )
            return second_value;
        return std::max( first_value, second_value );
    }

    norm_type norm_l_inf( const vector_type &vector ) const
    {
        return norm_inf( vector );
    }

    norm_type normalize( vector_type &vector ) const
    {
        const auto value = norm( vector );
        if ( value > norm_type( 0 ) )
            scale( scalar_type( 1 ) / value, vector );
        return value;
    }

    void make_abs_copy( const vector_type &source, vector_type &destination ) const
    {
        first_->make_abs_copy( source.first, destination.first );
        second_->make_abs_copy( source.second, destination.second );
    }

    void make_abs( vector_type &vector ) const
    {
        first_->make_abs( vector.first );
        second_->make_abs( vector.second );
    }

    void max_pointwise( scalar_type value, const vector_type &source, vector_type &destination ) const
    {
        first_->max_pointwise( value, source.first, destination.first );
        second_->max_pointwise( value, source.second, destination.second );
    }

    void max_pointwise( scalar_type value, vector_type &vector ) const
    {
        first_->max_pointwise( value, vector.first );
        second_->max_pointwise( value, vector.second );
    }

    void min_pointwise( scalar_type value, const vector_type &source, vector_type &destination ) const
    {
        first_->min_pointwise( value, source.first, destination.first );
        second_->min_pointwise( value, source.second, destination.second );
    }

    void min_pointwise( scalar_type value, vector_type &vector ) const
    {
        first_->min_pointwise( value, vector.first );
        second_->min_pointwise( value, vector.second );
    }

    void mul_pointwise( vector_type &vector, scalar_type scale, const vector_type &source ) const
    {
        first_->mul_pointwise( vector.first, scale, source.first );
        second_->mul_pointwise( vector.second, scale, source.second );
    }

    void mul_pointwise(
        scalar_type first_scale, const vector_type &first, scalar_type second_scale, const vector_type &second,
        vector_type &destination
    ) const
    {
        first_->mul_pointwise( first_scale, first.first, second_scale, second.first, destination.first );
        second_->mul_pointwise( first_scale, first.second, second_scale, second.second, destination.second );
    }

    void div_pointwise( vector_type &vector, scalar_type scale, const vector_type &source ) const
    {
        first_->div_pointwise( vector.first, scale, source.first );
        second_->div_pointwise( vector.second, scale, source.second );
    }

    void div_pointwise(
        scalar_type first_scale, const vector_type &first, scalar_type second_scale, const vector_type &second,
        vector_type &destination
    ) const
    {
        first_->div_pointwise( first_scale, first.first, second_scale, second.first, destination.first );
        second_->div_pointwise( first_scale, first.second, second_scale, second.second, destination.second );
    }

    bool is_valid_number( const vector_type &vector ) const
    {
        // Both components may perform collectives, so never short-circuit the second check.
        const bool first_valid  = first_->is_valid_number( vector.first );
        const bool second_valid = second_->is_valid_number( vector.second );
        return first_valid && second_valid;
    }

    void
    assign( const multivector_type &source, ordinal_type capacity, ordinal_type column, vector_type &destination ) const
    {
        assign( source[checked_column( source, capacity, column )], destination );
    }

    void
    assign( const vector_type &source, multivector_type &destination, ordinal_type capacity, ordinal_type column ) const
    {
        assign( source, destination[checked_column( destination, capacity, column )] );
    }

    scalar_type scalar_prod(
        const multivector_type &source, ordinal_type capacity, ordinal_type column, const vector_type &vector
    ) const
    {
        return scalar_prod( source[checked_column( source, capacity, column )], vector );
    }

    scalar_type scalar_prod_l2(
        const multivector_type &source, ordinal_type capacity, ordinal_type column, const vector_type &vector
    ) const
    {
        return scalar_prod( source, capacity, column, vector );
    }

    void add_lin_comb(
        scalar_type source_scale, const multivector_type &source, ordinal_type capacity, ordinal_type column,
        scalar_type destination_scale, vector_type &destination
    ) const
    {
        add_lin_comb(
            source_scale, source[checked_column( source, capacity, column )], destination_scale, destination
        );
    }

    void set( const scalar_type *host, vector_type &destination, std::size_t count ) const
    {
        if ( count != get_default_size() )
            throw std::invalid_argument( "product_vector_space::set size mismatch" );
        first_->set( host, destination.first, first_size() );
        second_->set( host + first_size(), destination.second, second_size() );
    }

    void get( const vector_type &source, scalar_type *host, std::size_t count ) const
    {
        if ( count != get_default_size() )
            throw std::invalid_argument( "product_vector_space::get size mismatch" );
        first_->get( source.first, host, first_size() );
        second_->get( source.second, host + first_size(), second_size() );
    }

    void set_value_at_point( scalar_type value, std::size_t index, vector_type &vector ) const
    {
        if ( index < first_size() )
        {
            first_->set_value_at_point( value, index, vector.first );
            return;
        }
        const std::size_t second_index = index - first_size();
        if ( second_index >= second_size() )
            throw std::out_of_range( "product_vector_space::set_value_at_point" );
        second_->set_value_at_point( value, second_index, vector.second );
    }

    scalar_type get_value_at_point( std::size_t index, const vector_type &vector ) const
    {
        if ( index < first_size() )
            return first_->get_value_at_point( index, vector.first );
        const std::size_t second_index = index - first_size();
        if ( second_index >= second_size() )
            throw std::out_of_range( "product_vector_space::get_value_at_point" );
        return second_->get_value_at_point( second_index, vector.second );
    }

protected:
    std::shared_ptr<first_space_type>  first_owner_;
    std::shared_ptr<second_space_type> second_owner_;

private:
    template <class Space>
    static auto component_size( const Space &space, int ) -> decltype( space.get_default_size() )
    {
        return space.get_default_size();
    }

    template <class Space>
    static auto component_size( const Space &space, long ) -> decltype( space.size() )
    {
        return space.size();
    }

    static std::size_t checked_count( ordinal_type count )
    {
        if ( count < ordinal_type{} )
            throw std::invalid_argument( "product_vector_space requires a nonnegative count" );
        return static_cast<std::size_t>( count );
    }

    static void check_multivector_size( const multivector_type &vectors, std::size_t count )
    {
        if ( vectors.size() < count )
            throw std::out_of_range( "product_vector_space multivector capacity" );
    }

    static std::size_t checked_column( const multivector_type &vectors, ordinal_type capacity, ordinal_type column )
    {
        const std::size_t size_capacity = checked_count( capacity );
        const std::size_t size_column   = checked_count( column );
        check_multivector_size( vectors, size_capacity );
        if ( size_column >= size_capacity )
            throw std::out_of_range( "product_vector_space multivector column" );
        return size_column;
    }

    const first_space_type  *first_;
    const second_space_type *second_;
};

template <class VectorSpace>
using two_block_vector_space = product_vector_space<VectorSpace, VectorSpace>;

} // namespace operations
} // namespace nmfd

#endif
