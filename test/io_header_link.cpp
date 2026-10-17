#include <nmfd/operations/io/file_operations.h>

std::size_t linked_square_rows( const std::string &path )
{
    return file_operations::read_matrix_size_square( path );
}
