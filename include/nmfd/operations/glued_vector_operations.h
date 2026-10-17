#ifndef __NMFD_GLUED_VECTOR_OPERATIONS_H__
#define __NMFD_GLUED_VECTOR_OPERATIONS_H__

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <nmfd/operations/default_multivector_operations_base.h>
#include <nmfd/operations/glued_vector.h>

namespace nmfd
{
namespace operations
{
namespace detail
{

template <class Operations, class = void>
struct glued_ordinal
{
    using type = std::ptrdiff_t;
};

template <class Operations>
struct glued_ordinal<Operations, std::void_t<typename Operations::ordinal_type>>
{
    using type = typename Operations::ordinal_type;
};

template <class Operations, class = void>
struct glued_precision
{
    static void high( Operations &operations )
    {
        operations.use_high_precision();
    }
    static void regular( Operations &operations )
    {
        operations.use_standard_precision();
    }
};

template <class Operations>
struct glued_precision<
    Operations, std::void_t<
                    decltype( std::declval<Operations &>().set_high_precision() ),
                    decltype( std::declval<Operations &>().set_regular_precision() )>>
{
    static void high( Operations &operations )
    {
        operations.set_high_precision();
    }
    static void regular( Operations &operations )
    {
        operations.set_regular_precision();
    }
};

}

template <class VectorOperations, std::size_t n>
class glued_vector_operations
    : public default_multivector_operations_base<
          glued_vector_operations<VectorOperations, n>, typename VectorOperations::scalar_type,
          glued_vector<typename VectorOperations::vector_type, n>,
          typename detail::glued_ordinal<VectorOperations>::type,
          std::vector<glued_vector<typename VectorOperations::vector_type, n>>,
          vector_operations_base<
              typename VectorOperations::scalar_type, glued_vector<typename VectorOperations::vector_type, n>,
              std::vector<glued_vector<typename VectorOperations::vector_type, n>>,
              typename detail::glued_ordinal<VectorOperations>::type, typename VectorOperations::norm_type>>
{
public:
    using scalar_type      = typename VectorOperations::scalar_type;
    using norm_type        = typename VectorOperations::norm_type;
    using ordinal_type     = typename detail::glued_ordinal<VectorOperations>::type;
    using Ord              = ordinal_type;
    using vector_type      = glued_vector<typename VectorOperations::vector_type, n>;
    using multivector_type = std::vector<vector_type>;
    using operations_base_type =
        vector_operations_base<scalar_type, vector_type, multivector_type, ordinal_type, norm_type>;
    using parent_t = default_multivector_operations_base<
        glued_vector_operations, scalar_type, vector_type, ordinal_type, multivector_type, operations_base_type>;
    using component_operations_type = VectorOperations;
    using component_array_type      = std::array<std::shared_ptr<VectorOperations>, n>;
    using parent_t::add_lin_comb;
    using parent_t::assign;
    using parent_t::scalar_prod;
    using parent_t::scalar_prod_l2;

    explicit glued_vector_operations( component_array_type operations ) : internal_ops_( std::move( operations ) )
    {
        for ( const auto &component : internal_ops_ )
            if ( !component )
                throw std::invalid_argument( "Glued operations require nonnull components" );
    }
    explicit glued_vector_operations( std::shared_ptr<VectorOperations> operations )
    {
        if ( !operations )
            throw std::invalid_argument( "Glued operations require a nonnull component" );
        internal_ops_.fill( std::move( operations ) );
    }
    template <
        class... Args,
        std::enable_if_t<
            sizeof...( Args ) == n && ( std::is_same_v<Args, std::shared_ptr<VectorOperations>> && ... ), int> = 0>
    explicit glued_vector_operations( Args... operations )
        : glued_vector_operations( component_array_type{ std::move( operations )... } )
    {
    }

    void set_high_precision() override
    {
        parent_t::set_high_precision();
        for ( const auto &component : internal_ops_ )
            detail::glued_precision<VectorOperations>::high( *component );
    }
    void set_regular_precision() override
    {
        parent_t::set_regular_precision();
        for ( const auto &component : internal_ops_ )
            detail::glued_precision<VectorOperations>::regular( *component );
    }

    bool check_is_valid_number( const vector_type &x ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            if ( !internal_ops_[i]->check_is_valid_number( x.comp( i ) ) )
                return false;
        return true;
    }
    bool is_valid_number( const vector_type &x ) const override
    {
        return check_is_valid_number( x );
    }
    scalar_type scalar_prod( const vector_type &x, const vector_type &y ) const override
    {
        scalar_type result{};
        for ( std::size_t i = 0; i < n; ++i )
            result += internal_ops_[i]->scalar_prod( x.comp( i ), y.comp( i ) );
        return result;
    }
    scalar_type scalar_prod_l2( const vector_type &x, const vector_type &y ) const override
    {
        return scalar_prod( x, y );
    }
    scalar_type sum( const vector_type &x ) const override
    {
        scalar_type result{};
        for ( std::size_t i = 0; i < n; ++i )
            result += internal_ops_[i]->sum( x.comp( i ) );
        return result;
    }
    norm_type asum( const vector_type &x ) const override
    {
        norm_type result{};
        for ( std::size_t i = 0; i < n; ++i )
            result += internal_ops_[i]->asum( x.comp( i ) );
        return result;
    }
    norm_type norm_sq( const vector_type &x ) const override
    {
        norm_type result{};
        for ( std::size_t i = 0; i < n; ++i )
            result += internal_ops_[i]->norm_sq( x.comp( i ) );
        return result;
    }
    norm_type norm( const vector_type &x ) const override
    {
        return std::sqrt( norm_sq( x ) );
    }
    // The historical glued metric is the sum of component-normalized squared norms.
    norm_type norm2_sq( const vector_type &x ) const override
    {
        norm_type result{};
        for ( std::size_t i = 0; i < n; ++i )
            result += internal_ops_[i]->norm2_sq( x.comp( i ) );
        return result;
    }
    norm_type norm_l2( const vector_type &x ) const override
    {
        return std::sqrt( norm2_sq( x ) );
    }
    norm_type norm_inf( const vector_type &x ) const override
    {
        norm_type result{};
        for ( std::size_t i = 0; i < n; ++i )
        {
            const auto value = internal_ops_[i]->norm_inf( x.comp( i ) );
            if ( value != value )
                return value;
            result = std::max( result, value );
        }
        return result;
    }
    void assign_scalar( scalar_type value, vector_type &x ) const override
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->assign_scalar( value, x.comp( i ) );
    }
    void add_mul_scalar( scalar_type value, scalar_type scale, vector_type &x ) const override
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->add_mul_scalar( value, scale, x.comp( i ) );
    }
    void scale( scalar_type value, vector_type &x ) const override
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->scale( value, x.comp( i ) );
    }
    void assign( const vector_type &x, vector_type &y ) const override
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->assign( x.comp( i ), y.comp( i ) );
    }
    void assign_mul( scalar_type a, const vector_type &x, vector_type &y ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->assign_mul( a, x.comp( i ), y.comp( i ) );
    }
    void assign_mul( scalar_type a, const vector_type &x, scalar_type b, const vector_type &y, vector_type &z ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->assign_mul( a, x.comp( i ), b, y.comp( i ), z.comp( i ) );
    }
    void assign_lin_comb( scalar_type a, const vector_type &x, vector_type &y ) const override
    {
        assign_mul( a, x, y );
    }
    void assign_lin_comb(
        scalar_type a, const vector_type &x, scalar_type b, const vector_type &y, vector_type &z
    ) const override
    {
        assign_mul( a, x, b, y, z );
    }
    void add_mul( scalar_type a, const vector_type &x, scalar_type b, vector_type &y ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->add_mul( a, x.comp( i ), b, y.comp( i ) );
    }
    void add_mul( scalar_type a, const vector_type &x, vector_type &y ) const
    {
        add_mul( a, x, scalar_type( 1 ), y );
    }
    void add_mul(
        scalar_type a, const vector_type &x, scalar_type b, const vector_type &y, scalar_type c, vector_type &z
    ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->add_mul( a, x.comp( i ), b, y.comp( i ), c, z.comp( i ) );
    }
    void add_lin_comb( scalar_type a, const vector_type &x, scalar_type b, vector_type &y ) const override
    {
        add_mul( a, x, b, y );
    }
    void add_lin_comb( scalar_type a, const vector_type &x, vector_type &y ) const
    {
        add_mul( a, x, y );
    }
    void div_pointwise( scalar_type a, const vector_type &x, scalar_type b, const vector_type &y, vector_type &z ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->div_pointwise( a, x.comp( i ), b, y.comp( i ), z.comp( i ) );
    }
    template <class RealVector>
    void make_abs_copy( const vector_type &x, glued_vector<RealVector, n> &y ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->make_abs_copy( x.comp( i ), y.comp( i ) );
    }
    template <class RealVector>
    void max_pointwise( norm_type value, const glued_vector<RealVector, n> &x, glued_vector<RealVector, n> &y ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->max_pointwise( value, x.comp( i ), y.comp( i ) );
    }
    template <class RealVector>
    void max_pointwise( norm_type value, glued_vector<RealVector, n> &y ) const
    {
        for ( std::size_t i = 0; i < n; ++i )
            internal_ops_[i]->max_pointwise( value, y.comp( i ) );
    }

protected:
    component_array_type internal_ops_;
};

}
}

#endif
