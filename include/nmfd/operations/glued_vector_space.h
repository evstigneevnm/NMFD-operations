#ifndef __NMFD_GLUED_VECTOR_SPACE_H__
#define __NMFD_GLUED_VECTOR_SPACE_H__

#include <nmfd/operations/default_multivector_factory_base.h>
#include <nmfd/operations/glued_vector_operations.h>

namespace nmfd
{
namespace operations
{

template <class VectorSpace, std::size_t n>
class glued_vector_space
    : public glued_vector_operations<VectorSpace, n>,
      public default_multivector_factory_base<
          glued_vector_space<VectorSpace, n>, typename VectorSpace::scalar_type,
          glued_vector<typename VectorSpace::vector_type, n>, typename detail::glued_ordinal<VectorSpace>::type>
{
public:
    using parent_t          = glued_vector_operations<VectorSpace, n>;
    using scalar_type       = typename parent_t::scalar_type;
    using norm_type         = typename parent_t::norm_type;
    using vector_type       = typename parent_t::vector_type;
    using multivector_type  = typename parent_t::multivector_type;
    using ordinal_type      = typename parent_t::ordinal_type;
    using Ord               = ordinal_type;
    using space_comp_type   = VectorSpace;
    using factory_base_type = vector_factory_base<scalar_type, vector_type, multivector_type, ordinal_type>;
    using parent_t::parent_t;

    void init_vector( vector_type &x ) const override
    {
        std::size_t initialized = 0;
        try
        {
            for ( ; initialized < n; ++initialized )
                this->internal_ops_[initialized]->init_vector( x.comp( initialized ) );
        }
        catch ( ... )
        {
            while ( initialized > 0 )
            {
                --initialized;
                this->internal_ops_[initialized]->free_vector( x.comp( initialized ) );
            }
            throw;
        }
    }
    void start_use_vector( vector_type &x ) const override
    {
        std::size_t started = 0;
        try
        {
            for ( ; started < n; ++started )
                this->internal_ops_[started]->start_use_vector( x.comp( started ) );
        }
        catch ( ... )
        {
            while ( started > 0 )
            {
                --started;
                this->internal_ops_[started]->stop_use_vector( x.comp( started ) );
            }
            throw;
        }
    }
    void stop_use_vector( vector_type &x ) const override
    {
        for ( std::size_t i = n; i > 0; --i )
            this->internal_ops_[i - 1]->stop_use_vector( x.comp( i - 1 ) );
    }
    void free_vector( vector_type &x ) const override
    {
        for ( std::size_t i = n; i > 0; --i )
            this->internal_ops_[i - 1]->free_vector( x.comp( i - 1 ) );
    }
    template <class... Vectors>
    void init_vectors( Vectors &...vectors ) const
    {
        ( init_vector( vectors ), ... );
    }
    template <class... Vectors>
    void start_use_vectors( Vectors &...vectors ) const
    {
        ( start_use_vector( vectors ), ... );
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
    VectorSpace &space_comp( std::size_t i )
    {
        return *this->internal_ops_[i];
    }
    const VectorSpace &space_comp( std::size_t i ) const
    {
        return *this->internal_ops_[i];
    }
    std::size_t size() const
    {
        std::size_t result = 0;
        for ( const auto &component : this->internal_ops_ )
            result += component->size();
        return result;
    }
    std::size_t get_default_size() const
    {
        return size();
    }
    std::size_t get_vector_size() const
    {
        return size();
    }
    std::size_t get_size( const vector_type &x ) const
    {
        std::size_t result = 0;
        for ( std::size_t i = 0; i < n; ++i )
            result += this->internal_ops_[i]->get_size( x.comp( i ) );
        return result;
    }
};

}
}

#endif
