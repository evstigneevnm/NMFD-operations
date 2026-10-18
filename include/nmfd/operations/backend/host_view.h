#ifndef __NMFD_OPERATIONS_BACKEND_HOST_VIEW_H__
#define __NMFD_OPERATIONS_BACKEND_HOST_VIEW_H__

#include <nmfd/operations/detail/scfd_backend_ext/host_view.h>

namespace nmfd
{
namespace operations
{
namespace backend
{

using detail::scfd_backend_ext::ensure_host_array;
using detail::scfd_backend_ext::host_view_traits;
using detail::scfd_backend_ext::scoped_host_view;

}
}
}

#endif
