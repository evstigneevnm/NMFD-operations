#ifndef __NMFD_PAIR_VECTOR_SPACE_H__
#define __NMFD_PAIR_VECTOR_SPACE_H__

#include <nmfd/operations/product_vector_space.h>

namespace nmfd
{
namespace operations
{

// Preserve the historical owning pair interface without duplicating product algebra.
template <class VectorSpace1, class VectorSpace2>
struct pair_vector_space : public product_vector_space<VectorSpace1, VectorSpace2>
{
    using parent_t = product_vector_space<VectorSpace1, VectorSpace2>;
    using typename parent_t::scalar_type;
    using typename parent_t::vector_type;

    pair_vector_space( std::shared_ptr<VectorSpace1> first, std::shared_ptr<VectorSpace2> second )
        : parent_t( std::move( first ), std::move( second ) )
    {
    }

    std::shared_ptr<VectorSpace1> first()
    {
        return this->first_owner_;
    }
    std::shared_ptr<VectorSpace2> second()
    {
        return this->second_owner_;
    }
    const std::shared_ptr<VectorSpace1> &first() const
    {
        return this->first_owner_;
    }
    const std::shared_ptr<VectorSpace2> &second() const
    {
        return this->second_owner_;
    }

    void assign_slices(
        const vector_type &x, const std::vector<std::pair<std::size_t, std::size_t>> slices, vector_type &y
    ) const
    {
        std::size_t index_y = 0;
        for ( const auto &slice : slices )
        {
            if ( slice.second > this->size() )
                throw std::logic_error(
                    "pair_vector_space::assign_slice: provided slice size is greater than input vector size."
                );
            for ( std::size_t j = slice.first; j < slice.second; ++j )
                this->set_value_at_point( this->get_value_at_point( j, x ), index_y++, y );
        }
    }

    void assign_skip_slices(
        const vector_type &x, const std::vector<std::pair<std::size_t, std::size_t>> skip_slices, vector_type &y
    ) const
    {
        std::size_t index_y = 0;
        for ( std::size_t j = 0; j < this->size(); ++j )
            for ( const auto &slice : skip_slices )
                if ( j <= slice.first || j > slice.second )
                    this->set_value_at_point( this->get_value_at_point( j, x ), index_y++, y );
    }
};

} // namespace operations
} // namespace nmfd

#endif
