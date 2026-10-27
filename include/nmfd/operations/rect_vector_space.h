#ifndef __NMFD_RECT_VECTOR_SPACE_H__
#define __NMFD_RECT_VECTOR_SPACE_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <nmfd/operations/vector_operations_base.h>
#include <nmfd/operations/vector_space_base.h>
#include <nmfd/operations/default_multivector_space_base.h>
#include <scfd/arrays/array_nd.h>
#include <scfd/static_vec/vec.h>
#include <scfd/arrays/tensorN_array_nd.h>
#include <scfd/communication/trivial_comm.h>
#include <scfd/functional/basic_ops.h>

#include "kernels/rect_vector_space.h"

namespace nmfd
{

template <
    class Type, int Dim, int TensorDim, class Backend,
    class CommOrOrdinal = scfd::communication::trivial_comm<typename Backend::memory_type>,
    class Ordinal =
        std::conditional_t<std::is_integral<CommOrOrdinal>::value, CommOrOrdinal, typename Backend::ordinal_type>,
    /***********************************************************/
    class VectorType = scfd::arrays::tensor1_array_nd<Type, Dim, typename Backend::memory_type, TensorDim>,
    class IdxType    = scfd::static_vec::vec<Ordinal, Dim>>
class rect_vector_space
    : public nmfd::operations::default_multivector_space_base<
          rect_vector_space<Type, Dim, TensorDim, Backend, CommOrOrdinal, Ordinal, VectorType, IdxType>, Type,
          VectorType, Ordinal>
{
    using parent_t = nmfd::operations::default_multivector_space_base<
        rect_vector_space<Type, Dim, TensorDim, Backend, CommOrOrdinal, Ordinal, VectorType, IdxType>, Type, VectorType,
        Ordinal>;

public:
    static const int dim        = Dim;
    static const int tensor_dim = TensorDim;
    using value_type            = Type;
    using scalar_type           = Type;
    using ordinal_type          = Ordinal;
    // The integral fifth argument remains accepted for local-only callers.
    using comm_type = std::conditional_t<
        std::is_integral<CommOrOrdinal>::value, scfd::communication::trivial_comm<typename Backend::memory_type>,
        CommOrOrdinal>;

    using backend_type = Backend;

    using for_each_nd_type = typename Backend::template for_each_nd_type<Dim>;
    using reduce_type      = typename Backend::reduce_type;
    using memory_type      = typename Backend::memory_type;

    using multivector_type = typename parent_t::multivector_type;
    using vector_type      = VectorType;
    using array_nd_type    = scfd::arrays::array_nd<Type, Dim, typename Backend::memory_type>;
    using idx_nd_type      = IdxType;

public: // Especially for SYCL
    using shur_prod_kernel      = kernels::shur_prod<idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
    using sum_kernel            = kernels::sum<idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
    using assign_scalar_kernel  = kernels::assign_scalar<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using add_mul_scalar_kernel = kernels::add_mul_scalar<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using scale_kernel          = kernels::scale<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_kernel         = kernels::assign<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_lin_comb_1_kernel = kernels::assign_lin_1_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using assign_lin_comb_2_kernel = kernels::assign_lin_2_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;
    using add_lin_comb_kernel      = kernels::add_lin_comb<idx_nd_type, scalar_type, vector_type, TensorDim>;

private:
    idx_nd_type   range;
    ordinal_type  points_      = 0;
    std::uint64_t global_size_ = 0;
    comm_type     comm_;
    ordinal_type  stencil_;
    ordinal_type  max_stencil_order_;
    array_nd_type mutable helper;

    for_each_nd_type for_each_nd_inst;
    reduce_type      reduce_inst;

public:
    rect_vector_space(
        idx_nd_type r, comm_type comm = comm_type{}, bool use_high_precision = false, ordinal_type stencil = 0,
        ordinal_type max_stencil_order = 0
    )
        : parent_t( use_high_precision ), range( r ), comm_( comm ), stencil_( stencil ),
          max_stencil_order_( max_stencil_order )
    {
        static_assert( Dim > 0 && TensorDim > 0, "Rectangular spaces need positive dimensions" );
        static_assert(
            std::is_integral<ordinal_type>::value && std::is_signed<ordinal_type>::value,
            "Rectangular spaces require a signed integral ordinal"
        );
        std::uint64_t points = 1, storage = TensorDim;
        bool          invalid = stencil < 0 || max_stencil_order < 0 || max_stencil_order > Dim;
        const auto    limit   = std::min(
            { static_cast<std::uint64_t>( std::numeric_limits<typename Backend::ordinal_type>::max() ),
                   static_cast<std::uint64_t>( std::numeric_limits<ordinal_type>::max() ),
                   static_cast<std::uint64_t>( std::numeric_limits<typename vector_type::ordinal_type>::max() ) }
        );
        for ( int axis = 0; axis < Dim; ++axis )
        {
            const auto extent = static_cast<std::uint64_t>( r[axis] );
            invalid =
                invalid || r[axis] < 0 || extent > limit ||
                ( stencil >= 0 && static_cast<std::uint64_t>( stencil ) > ( limit - std::min( extent, limit ) ) / 2 );
            if ( extent && points > limit / extent )
                invalid = true;
            else
                points *= extent;
            if ( !invalid )
            {
                const auto padded = extent + std::uint64_t( stencil ) * 2;
                if ( padded && storage > limit / padded )
                    invalid = true;
                else
                    storage *= padded;
            }
        }
#ifndef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        invalid = invalid || stencil != 0;
#endif
        if ( comm_.all_reduce_sum( int( invalid ) ) != 0 )
            throw std::invalid_argument( "Invalid rectangular range/halo, or index shifts are disabled" );
        points_      = static_cast<ordinal_type>( points );
        global_size_ = comm_.all_reduce_sum( points * TensorDim );
        if ( points_ != 0 )
            helper.init( range );
    }

    rect_vector_space( idx_nd_type r, bool use_high_precision )
        : rect_vector_space( r, comm_type{}, use_high_precision )
    {
    }

    ordinal_type get_stencil() const noexcept
    {
        return stencil_;
    }
    ordinal_type get_max_stencil_order() const noexcept
    {
        return max_stencil_order_;
    }
    idx_nd_type get_size() const noexcept
    {
        return range;
    }
    idx_nd_type size() const noexcept
    {
        return range;
    }

    // Scalar counts exclude halos; size()/get_size() retain their grid-shape interface.
    std::size_t get_default_size() const noexcept
    {
        return static_cast<std::size_t>( points_ ) * TensorDim;
    }
    std::size_t get_vector_size() const noexcept
    {
        return get_default_size();
    }
    std::size_t get_size( const vector_type & ) const noexcept
    {
        return get_default_size();
    }

public:
    void init_vector( vector_type &x ) const override
    {
#ifdef SCFD_ARRAYS_ENABLE_INDEX_SHIFT
        x.init(
            range + idx_nd_type::make_ones() * ( ordinal_type( 2 ) * stencil_ ),
            idx_nd_type::make_ones() * ( -stencil_ )
        );
#else
        x.init( range );
#endif
    }
    void free_vector( vector_type &x ) const override
    {
        x.free();
    }

    void start_use_vector( vector_type &x ) const override
    {
    }
    void stop_use_vector( vector_type &x ) const override
    {
    }

public: // Implementing Vector_Operations interface
    [[nodiscard]] bool is_valid_number( const vector_type &x ) const override
    {
        using kernel = kernels::rect_transform<
            kernels::rect_reduction::invalid, idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
        const auto invalid = local_reduce( kernel{ x, helper } );
        return comm_.all_reduce_sum( int( invalid != scalar_type( 0 ) ) ) == 0;
    }

    // reduction operations:
    [[nodiscard]] scalar_type scalar_prod( const vector_type &x, const vector_type &y ) const override
    {
        return comm_.all_reduce_sum( local_reduce( shur_prod_kernel{ x, y, helper } ) );
    }

    [[nodiscard]] scalar_type scalar_prod_l2( const vector_type &x, const vector_type &y ) const override
    {
        return scalar_prod( x, y );
    }

    // multivector interface implementations from parent restore because of overshadowing
    using parent_t::add_lin_comb;
    using parent_t::assign;
    using parent_t::scalar_prod;
    using parent_t::scalar_prod_l2;

    [[nodiscard]] scalar_type sum( const vector_type &x ) const override
    {
        return comm_.all_reduce_sum( local_reduce( sum_kernel{ x, helper } ) );
    }

    [[nodiscard]] scalar_type asum( const vector_type &x ) const override
    {
        using kernel = kernels::rect_transform<
            kernels::rect_reduction::absolute_sum, idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
        return comm_.all_reduce_sum( local_reduce( kernel{ x, helper } ) );
    }

    [[nodiscard]] scalar_type norm_inf( const vector_type &x ) const override
    {
        using kernel = kernels::rect_transform<
            kernels::rect_reduction::maximum, idx_nd_type, scalar_type, vector_type, array_nd_type, TensorDim>;
        return comm_.all_reduce_max( local_reduce( kernel{ x, helper }, scfd::functional::maximum<scalar_type>{} ) );
    }

    //standard vector norm:=sqrt(sum(x^2))
    [[nodiscard]] scalar_type norm( const vector_type &x ) const override
    {
        return std::sqrt( norm_sq( x ) );
    }

    //L2 emulation for the vector norm2:=sqrt(sum(x^2)/sz_)
    [[nodiscard]] scalar_type norm_l2( const vector_type &x ) const override
    {
        return std::sqrt( norm2_sq( x ) );
    }

    //standard vector norm_sq:=sum(x^2)
    [[nodiscard]] scalar_type norm_sq( const vector_type &x ) const override
    {
        return scalar_prod( x, x );
    }

    //L2 emulation for the vector norm2_sq:=sum(x^2)/sz_
    [[nodiscard]] scalar_type norm2_sq( const vector_type &x ) const override
    {
        const auto squared = norm_sq( x );
        return global_size_ == 0 ? scalar_type( 0 ) : squared / scalar_type( global_size_ );
    }

public:
    //calc: x := <vector_type with all elements equal to given scalar value>
    void assign_scalar( const scalar_type scalar, vector_type &x ) const override
    {
        local_apply( assign_scalar_kernel{ scalar, x } );
    }

    //calc: x := mul_x*x + <vector_type of all scalar value>
    void add_mul_scalar( const scalar_type scalar, const scalar_type mul_x, vector_type &x ) const override
    {
        local_apply( add_mul_scalar_kernel{ scalar, mul_x, x } );
    }

    //calc: x := scale*x
    void scale( const scalar_type scale, vector_type &x ) const override
    {
        local_apply( scale_kernel{ scale, x } );
    }

    //copy: y := x
    void assign( const vector_type &x, vector_type &y ) const override
    {
        local_apply( assign_kernel{ x, y } );
    }

    //calc: y := mul_x*x
    void assign_lin_comb( const scalar_type mul_x, const vector_type &x, vector_type &y ) const override
    {
        local_apply( assign_lin_comb_1_kernel{ mul_x, x, y } );
    }

    //calc: z := mul_x*x + mul_y*y
    void assign_lin_comb(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, const vector_type &y, vector_type &z
    ) const override
    {
        local_apply( assign_lin_comb_2_kernel{ mul_x, mul_y, x, y, z } );
    }

    //calc: y := mul_x*x + mul_y*y
    void add_lin_comb(
        const scalar_type mul_x, const vector_type &x, const scalar_type mul_y, vector_type &y
    ) const override
    {
        local_apply( add_lin_comb_kernel{ mul_x, mul_y, x, y } );
    }

private:
    template <class Kernel>
    void local_apply( const Kernel &kernel ) const
    {
        if ( points_ != 0 )
            for_each_nd_inst( kernel, range );
    }

    template <class Kernel, class Reduction = scfd::functional::plus<scalar_type>>
    scalar_type local_reduce( const Kernel &kernel, Reduction reduction = Reduction{} ) const
    {
        if ( points_ == 0 )
            return scalar_type( 0 );
        local_apply( kernel );
        for_each_nd_inst.wait();
        return reduce_inst( points_, helper.raw_ptr(), scalar_type( 0 ), reduction );
    }
};

} // namespace nmfd

#endif
