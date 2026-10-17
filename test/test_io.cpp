#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <nmfd/operations/io/file_operations.h>

std::size_t linked_square_rows( const std::string &path );

int main()
{
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory =
        std::filesystem::temp_directory_path() / ( "nmfd_operations_io_test_" + std::to_string( stamp ) );
    if ( !std::filesystem::create_directory( directory ) )
        throw std::runtime_error( "failed to create temporary test directory" );
    try
    {
        const std::vector<double> original{ 1.25, -2.5, 3.75 };
        std::vector<double>       recovered( original.size() );
        const auto                path = ( directory / "vector.dat" ).string();
        file_operations::write_vector<double>( path, original.size(), original );
        file_operations::read_vector<double>( path, recovered.size(), recovered );
        if ( original != recovered )
            throw std::runtime_error( "vector round trip failed" );
        const auto matrix_path = ( directory / "matrix.dat" ).string();
        file_operations::write_matrix( matrix_path, 2, 2, std::vector<double>{ 1, 2, 3, 4 } );
        if ( linked_square_rows( matrix_path ) != 2 )
            throw std::runtime_error( "matrix indexing or cross-translation-unit I/O failed" );
        std::filesystem::remove_all( directory );
        std::cout << "I/O round trip passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::filesystem::remove_all( directory );
        std::cerr << error.what() << '\n';
        return 1;
    }
}
