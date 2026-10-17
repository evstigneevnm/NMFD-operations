#ifndef __NMFD_GLUED_VECTOR_H__
#define __NMFD_GLUED_VECTOR_H__

#include <array>
#include <cstddef>

namespace nmfd
{
namespace operations
{

template <class Vector, std::size_t n>
class glued_vector
{
    static_assert( n > 0, "A glued vector requires at least one component" );

public:
    using component_type                         = Vector;
    static constexpr std::size_t component_count = n;

    Vector &comp( std::size_t index )
    {
        return components_[index];
    }
    const Vector &comp( std::size_t index ) const
    {
        return components_[index];
    }

private:
    std::array<Vector, n> components_{};
};

}
}

#endif
