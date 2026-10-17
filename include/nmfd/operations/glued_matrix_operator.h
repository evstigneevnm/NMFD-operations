// Copyright © 2020-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_GLUED_MATRIX_OPERATOR_H__
#define __NMFD_GLUED_MATRIX_OPERATOR_H__

#include <nmfd/operations/glued_matrix.h>
#include <nmfd/operations/glued_vector_space.h>

namespace nmfd
{
namespace operations
{

template <class Operations, std::size_t n>
class glued_matrix_operator : public glued_matrix<typename Operations::matrix_type, n>
{
public:
    using parent_t                   = glued_matrix<typename Operations::matrix_type, n>;
    using scalar_type                = typename Operations::scalar_type;
    using internal_vector_space_type = typename Operations::vector_space_type;
    using vector_type                = glued_vector<typename Operations::vector_type, n>;
    using vector_space_type          = glued_vector_space<internal_vector_space_type, n>;
    using block_array_type           = typename parent_t::block_array_type;
    using matrix_type                = typename Operations::matrix_type;

    explicit glued_matrix_operator( std::shared_ptr<Operations> operations ) : internal_ops_( std::move( operations ) )
    {
        if ( !internal_ops_ )
            throw std::invalid_argument( "Glued matrix operator requires operations" );
    }
    glued_matrix_operator( std::shared_ptr<Operations> operations, block_array_type matrices )
        : parent_t( std::move( matrices ) ), internal_ops_( std::move( operations ) )
    {
        prepare();
    }
    glued_matrix_operator( std::shared_ptr<Operations> operations, const parent_t &matrix )
        : parent_t( matrix ), internal_ops_( std::move( operations ) )
    {
        prepare();
    }

    using parent_t::comp_ptr;
    matrix_type &comp( std::size_t i, std::size_t j )
    {
        prepared_ = false;
        return parent_t::comp( i, j );
    }
    const matrix_type &comp( std::size_t i, std::size_t j ) const
    {
        return parent_t::comp( i, j );
    }

    // Rebuild geometry after block layout changes. Coefficient-only updates reuse it.
    void prepare() const
    {
        prepared_ = false;
        if ( !internal_ops_ )
            throw std::invalid_argument( "Glued matrix operator requires operations" );
        typename vector_space_type::component_array_type rows, columns;
        for ( std::size_t i = 0; i < n; ++i )
        {
            for ( std::size_t j = 0; j < n; ++j )
            {
                auto image  = internal_ops_->get_matrix_im_space( comp( i, j ) );
                auto domain = internal_ops_->get_matrix_dom_space( comp( i, j ) );
                if ( !image || !domain )
                    throw std::invalid_argument( "Glued block requires domain and image spaces" );
                if ( j == 0 )
                    rows[i] = std::move( image );
                else if ( rows[i]->size() != image->size() )
                    throw std::invalid_argument( "Glued blocks in a row have different image dimensions" );
                if ( i == 0 )
                    columns[j] = std::move( domain );
                else if ( columns[j]->size() != domain->size() )
                    throw std::invalid_argument( "Glued blocks in a column have different domain dimensions" );
            }
        }
        auto image  = std::make_shared<vector_space_type>( std::move( rows ) );
        auto domain = std::make_shared<vector_space_type>( std::move( columns ) );
        image_      = std::move( image );
        domain_     = std::move( domain );
        prepared_   = true;
    }

    void apply( const vector_type &x, vector_type &y ) const
    {
        if ( &x == &y )
            throw std::invalid_argument( "Glued matrix apply requires nonoverlapping input and output" );
        ensure_prepared();
        for ( std::size_t i = 0; i < n; ++i )
        {
            if ( domain_->space_comp( i ).get_size( x.comp( i ) ) != domain_->space_comp( i ).size() ||
                 image_->space_comp( i ).get_size( y.comp( i ) ) != image_->space_comp( i ).size() )
                throw std::invalid_argument( "Glued matrix apply: vector component dimension mismatch" );
        }
        for ( std::size_t i = 0; i < n; ++i )
        {
            internal_ops_->assign_scalar( scalar_type( 0 ), y.comp( i ) );
            for ( std::size_t j = 0; j < n; ++j )
                internal_ops_->add_matrix_vector_prod(
                    scalar_type( 1 ), comp( i, j ), x.comp( j ), scalar_type( 1 ), y.comp( i )
                );
        }
    }
    const std::shared_ptr<vector_space_type> &get_im_space() const
    {
        ensure_prepared();
        return image_;
    }
    const std::shared_ptr<vector_space_type> &get_dom_space() const
    {
        ensure_prepared();
        return domain_;
    }

private:
    void ensure_prepared() const
    {
        if ( !prepared_ )
            prepare();
    }
    std::shared_ptr<Operations>                internal_ops_;
    mutable std::shared_ptr<vector_space_type> image_, domain_;
    mutable bool                               prepared_ = false;
};

}
}

#endif
