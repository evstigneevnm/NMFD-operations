#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <scfd/backend/serial_cpu.h>
#include <scfd/backend/omp.h>
#ifdef TEST_GLUED_CUDA
#    include <scfd/backend/cuda.h>
#    include <scfd/external_libraries/cublas_wrap_singleton_impl.h>
#    include <nmfd/operations/dense_operations_cuda_hip.h>
#endif

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/dense_operations_base.h>
#include <nmfd/operations/glued_matrix_operator.h>
#include <nmfd/operations/scfd_vector_space.h>

namespace
{

unsigned int checks = 0;

void require( bool passed, const char *message )
{
    ++checks;
    if ( !passed )
        throw std::runtime_error( message );
}

template <class Function>
void require_throw( Function function, const char *message )
{
    bool threw = false;
    try
    {
        function();
    }
    catch ( const std::exception & )
    {
        threw = true;
    }
    require( threw, message );
}

template <class Backend, class T>
class tracked_space : public nmfd::operations::dense_vector_space<
                          nmfd::operations::detail::scfd_array_traits<T, typename Backend::memory_type>, Backend>
{
    using parent_t = nmfd::operations::dense_vector_space<
        nmfd::operations::detail::scfd_array_traits<T, typename Backend::memory_type>, Backend>;

public:
    using vector_type = typename parent_t::vector_type;
    using parent_t::parent_t;
    mutable std::size_t live = 0, active = 0;
    bool                fail_init = false, fail_start = false;

    void init_vector( vector_type &x ) const override
    {
        if ( fail_init )
            throw std::runtime_error( "injected initialization failure" );
        parent_t::init_vector( x );
        ++live;
    }
    void start_use_vector( vector_type &x ) const override
    {
        if ( fail_start )
            throw std::runtime_error( "injected activation failure" );
        parent_t::start_use_vector( x );
        ++active;
    }
    void stop_use_vector( vector_type &x ) const override
    {
        parent_t::stop_use_vector( x );
        --active;
    }
    void free_vector( vector_type &x ) const override
    {
        if ( !x.is_free() )
            --live;
        parent_t::free_vector( x );
    }
};

template <class Backend, class T>
void test_vectors()
{
    using component_type  = nmfd::operations::scfd_vector_space<Backend, T>;
    using space_type      = nmfd::operations::glued_vector_space<component_type, 2>;
    using norm_type       = typename space_type::norm_type;
    using traits_type     = typename component_type::scalar_traits;
    using operations_type = nmfd::operations::glued_vector_operations<component_type, 2>;
    static_assert( std::is_floating_point_v<norm_type> );
    static_assert( std::is_base_of_v<typename space_type::factory_base_type, space_type> );
    static_assert( std::is_base_of_v<typename space_type::operations_base_type, space_type> );
    const norm_type               tolerance   = norm_type( 128 ) * std::numeric_limits<norm_type>::epsilon();
    auto                          first       = std::make_shared<component_type>( 2 );
    auto                          second      = std::make_shared<component_type>( 3 );
    std::weak_ptr<component_type> first_owner = first, second_owner = second;
    {
        space_type                                          space( first, second );
        operations_type                                     operations( first, second );
        nmfd::detail::vector_wrap<space_type, false, false> x( space ), y( space ), z( space );
        x.init();
        y.init();
        z.init();
        x.start_use();
        y.start_use();
        z.start_use();
        const T a = traits_type::make( 2, 1 ), b = traits_type::make( 3, -0.5 );
        space.assign_scalar( a, *x );
        space.assign_scalar( b, *y );
        require( space.size() == 5 && space.get_size( *x ) == 5, "Unequal component dimensions" );
        const T expected_dot = T( 5 ) * traits_type::conj_mul( a, b );
        require(
            traits_type::abs_sq( operations.scalar_prod( *x, *y ) - expected_dot ) <= tolerance,
            "Conjugating glued scalar product"
        );
        require(
            std::abs( space.norm_sq( *x ) - norm_type( 5 ) * traits_type::abs_sq( a ) ) <= tolerance,
            "Glued unscaled norm"
        );
        require(
            std::abs( space.norm2_sq( *x ) - norm_type( 2 ) * traits_type::abs_sq( a ) ) <= tolerance,
            "Historical component-normalized metric"
        );
        require(
            std::abs( space.norm_inf( *x ) - traits_type::asum_term( a ) ) <= tolerance,
            "Real-valued glued infinity norm"
        );
        operations.assign_lin_comb( T( 2 ), *x, T( -1 ), *y, *z );
        require(
            traits_type::abs_sq( first->get_value_at_point( 0, ( *z ).comp( 0 ) ) - ( T( 2 ) * a - b ) ) <= tolerance,
            "Glued affine mapping"
        );
        typename space_type::multivector_type basis;
        space.init_multivector( basis, 2 );
        space.start_use_multivector( basis, 2 );
        space.assign( *x, basis, 2, 1 );
        require(
            traits_type::abs_sq( space.scalar_prod( basis, 2, 1, *y ) - expected_dot ) <= tolerance,
            "Glued multivector algebra"
        );
        space.stop_use_multivector( basis, 2 );
        space.free_multivector( basis, 2 );
        require( basis.empty(), "Glued multivector cleanup" );
        typename space_type::operations_base_type &base = space;
        base.set_high_precision();
        (void)base.scalar_prod( *x, *y );
        require(
            first->last_reduction_used_high_precision() && second->last_reduction_used_high_precision(),
            "Base precision selection reaches every component"
        );
        base.set_regular_precision();
        (void)base.scalar_prod( *x, *y );
        require(
            !first->last_reduction_used_high_precision() && !second->last_reduction_used_high_precision(),
            "Ordinary precision restored on every component"
        );
        first->set_value_at_point(
            traits_type::make( std::numeric_limits<norm_type>::infinity(), 0 ), 0, ( *x ).comp( 0 )
        );
        require( !base.is_valid_number( *x ), "Glued finite-number validation" );
        first.reset();
        second.reset();
        require( !first_owner.expired() && !second_owner.expired(), "Components retained by glued owners" );
    }
    require( first_owner.expired() && second_owner.expired(), "No ownership cycle in glued components" );
    require_throw( [] { operations_type invalid( std::shared_ptr<component_type>{} ); }, "Null operations rejected" );

    auto                                                    repeated = std::make_shared<component_type>( 2 );
    nmfd::operations::glued_vector_space<component_type, 1> single( repeated );
    nmfd::operations::glued_vector_space<component_type, 3> triple( repeated );
    require( single.size() == 2 && triple.size() == 6, "One- and three-component construction" );
}

template <class Backend>
void test_failure_cleanup()
{
    using component_type                    = tracked_space<Backend, double>;
    using space_type                        = nmfd::operations::glued_vector_space<component_type, 2>;
    auto                             first  = std::make_shared<component_type>( 2 );
    auto                             second = std::make_shared<component_type>( 3 );
    space_type                       space( first, second );
    typename space_type::vector_type vector;
    second->fail_init = true;
    require_throw( [&] { space.init_vector( vector ); }, "Injected block initialization failure" );
    require( first->live == 0 && second->live == 0, "Completed block initialization rolled back" );
    second->fail_init  = false;
    second->fail_start = true;
    {
        nmfd::detail::vector_wrap<space_type, false, false> owner( space );
        owner.init();
        require_throw( [&] { owner.start_use(); }, "Injected block activation failure" );
        require( first->active == 0 && second->active == 0, "Completed block activations rolled back" );
    }
    require( first->live == 0 && second->live == 0, "All initialized storage released after activation failure" );
    second->fail_start = false;
    space.init_vector( vector );
    space.start_use_vector( vector );
    space.stop_use_vector( vector );
    space.free_vector( vector );
    require(
        first->live == 0 && second->live == 0 && first->active == 0 && second->active == 0,
        "Factory remains reusable after failure"
    );
}

template <class Operations>
class counted_operations : public Operations
{
public:
    using typename Operations::matrix_type;
    using typename Operations::vector_space_type;
    mutable std::size_t space_requests = 0;
    auto                get_matrix_im_space( const matrix_type &matrix ) const
    {
        ++space_requests;
        return Operations::get_matrix_im_space( matrix );
    }
    auto get_matrix_dom_space( const matrix_type &matrix ) const
    {
        ++space_requests;
        return Operations::get_matrix_dom_space( matrix );
    }
};

template <class Operations>
void test_matrices()
{
    using operations_type                              = counted_operations<Operations>;
    using operator_type                                = nmfd::operations::glued_matrix_operator<operations_type, 2>;
    using T                                            = typename operations_type::scalar_type;
    using norm_type                                    = typename operations_type::norm_type;
    using traits_type                                  = typename operations_type::scalar_traits;
    using space_type                                   = typename operator_type::vector_space_type;
    const norm_type                          tolerance = norm_type( 256 ) * std::numeric_limits<norm_type>::epsilon();
    const std::array<std::size_t, 2>         rows{ 2, 1 }, columns{ 1, 3 };
    auto                                     operations = std::make_shared<operations_type>();
    typename operator_type::block_array_type blocks;
    std::array<std::array<std::vector<T>, 2>, 2> reference;
    for ( std::size_t i = 0; i < 2; ++i )
    {
        for ( std::size_t j = 0; j < 2; ++j )
        {
            blocks[i][j] = std::make_shared<typename operator_type::matrix_type>();
            operations->init_matrix( rows[i], columns[j], *blocks[i][j] );
            auto view = blocks[i][j]->create_view( false );
            for ( std::size_t column = 0; column < columns[j]; ++column )
                for ( std::size_t row = 0; row < rows[i]; ++row )
                {
                    const T value       = traits_type::make( 1 + row + 2 * column + 3 * i + j, 0.25 * ( i + j ) );
                    view( row, column ) = value;
                    reference[i][j].push_back( value );
                }
            view.release( true );
        }
    }
    operator_type op( operations, blocks );
    auto          domain = op.get_dom_space();
    auto          image  = op.get_im_space();
    require( domain->size() == 4 && image->size() == 3, "Rectangular glued domain and image" );
    require(
        domain->space_comp( 0 ).size() == 1 && domain->space_comp( 1 ).size() == 3 &&
            image->space_comp( 0 ).size() == 2 && image->space_comp( 1 ).size() == 1,
        "Block geometry"
    );
    nmfd::detail::vector_wrap<space_type, false, false> x( *domain ), y( *image );
    x.init();
    y.init();
    x.start_use();
    y.start_use();
    std::array<std::vector<T>, 2> host_x;
    for ( std::size_t j = 0; j < 2; ++j )
        for ( std::size_t column = 0; column < columns[j]; ++column )
        {
            const T value = traits_type::make( 1 + j + column, -0.25 * column );
            host_x[j].push_back( value );
            domain->space_comp( j ).set_value_at_point( value, column, ( *x ).comp( j ) );
        }
    image->assign_scalar( traits_type::make( std::numeric_limits<norm_type>::quiet_NaN(), 0 ), *y );
    const auto requests = operations->space_requests;
    const auto owners   = operations.use_count();
    for ( int repeat = 0; repeat < 3; ++repeat )
        op.apply( *x, *y );
    require( requests == operations->space_requests, "Repeated apply creates no spaces" );
    require( owners == operations.use_count(), "Repeated apply does not retain new owners" );
    for ( std::size_t i = 0; i < 2; ++i )
        for ( std::size_t row = 0; row < rows[i]; ++row )
        {
            T expected{};
            for ( std::size_t j = 0; j < 2; ++j )
                for ( std::size_t column = 0; column < columns[j]; ++column )
                    expected += reference[i][j][row + column * rows[i]] * host_x[j][column];
            const T actual = image->space_comp( i ).get_value_at_point( row, ( *y ).comp( i ) );
            require(
                std::sqrt( traits_type::abs_sq( actual - expected ) ) <=
                    tolerance * std::max( norm_type( 1 ), std::sqrt( traits_type::abs_sq( expected ) ) ),
                "Glued block multiplication matches assembled reference"
            );
        }
    require_throw( [&] { op.apply( *x, *x ); }, "In-place block multiplication rejected" );
    require(
        domain->space_comp( 0 ).get_value_at_point( 0, ( *x ).comp( 0 ) ) == host_x[0][0],
        "Rejected apply preserves input"
    );
    require_throw( [&] { op.apply( *y, *y ); }, "Invalid output alias rejected" );
    typename space_type::component_array_type wrong_components{
        std::make_shared<typename operations_type::vector_space_type>( 1 ),
        std::make_shared<typename operations_type::vector_space_type>( 2 )
    };
    space_type                                          wrong_space( wrong_components );
    nmfd::detail::vector_wrap<space_type, false, false> wrong( wrong_space );
    wrong.init();
    wrong.start_use();
    image->assign_scalar( T( 9 ), *y );
    require_throw( [&] { op.apply( *wrong, *y ); }, "Wrong input component size rejected" );
    require(
        image->space_comp( 0 ).get_value_at_point( 0, ( *y ).comp( 0 ) ) == T( 9 ), "Dimension failure preserves output"
    );
    wrong_space.assign_scalar( T( 7 ), *wrong );
    require_throw( [&] { op.apply( *x, *wrong ); }, "Wrong output component size rejected" );
    require(
        wrong_space.space_comp( 0 ).get_value_at_point( 0, ( *wrong ).comp( 0 ) ) == T( 7 ),
        "Output shape checked before any writes"
    );

    op.apply( *x, *y );
    const T original = image->space_comp( 0 ).get_value_at_point( 0, ( *y ).comp( 0 ) );
    {
        auto view = blocks[0][0]->create_view( true );
        view( 0, 0 ) += T( 1 );
        view.release( true );
    }
    op.apply( *x, *y );
    require(
        requests == operations->space_requests && op.get_dom_space().get() == domain.get(),
        "Coefficient updates reuse cached geometry"
    );
    require(
        std::sqrt(
            traits_type::abs_sq(
                image->space_comp( 0 ).get_value_at_point( 0, ( *y ).comp( 0 ) ) - original - host_x[0][0]
            )
        ) <= tolerance,
        "Coefficient updates affect multiplication"
    );

    nmfd::operations::glued_matrix<typename operator_type::matrix_type, 2> matrix( blocks );
    operator_type                                                          copied( operations, matrix );
    require(
        copied.get_im_space()->size() == 3 && copied.comp_ptr( 0, 0 ) == blocks[0][0],
        "Matrix constructor shares blocks and reconstructs spaces"
    );
    operator_type lazy( operations );
    for ( std::size_t i = 0; i < 2; ++i )
        for ( std::size_t j = 0; j < 2; ++j )
        {
            operations->init_matrix( rows[i], columns[j], lazy.comp( i, j ) );
            operations->assign_zero_matrix( lazy.comp( i, j ) );
        }
    lazy.apply( *x, *y );
    require( image->norm( *y ) == norm_type( 0 ), "Default operator lazily prepares geometry" );
    require( lazy.get_dom_space()->size() == 4, "Default operator derives domain from filled blocks" );

    auto saved_image = lazy.get_im_space();
    lazy.comp( 0, 0 ).free();
    operations->init_matrix( 3, 1, lazy.comp( 0, 0 ) );
    require_throw( [&] { lazy.prepare(); }, "Changed incompatible layout invalidates cache" );
    require_throw( [&] { lazy.apply( *x, *y ); }, "Failed preparation cannot reuse stale geometry" );
    lazy.comp( 0, 1 ).free();
    operations->init_matrix( 3, 3, lazy.comp( 0, 1 ) );
    lazy.prepare();
    require(
        lazy.get_im_space()->size() == 4 && saved_image->size() == 3,
        "New layouts rebuild spaces without changing previously returned spaces"
    );
    lazy.comp_ptr( 0, 0 )->free();
    operations->init_matrix( 4, 1, *lazy.comp_ptr( 0, 0 ) );
    lazy.comp_ptr( 0, 1 )->free();
    operations->init_matrix( 4, 3, *lazy.comp_ptr( 0, 1 ) );
    lazy.prepare();
    require( lazy.get_im_space()->size() == 5, "Shared-alias layout changes use explicit preparation" );

    auto invalid  = blocks;
    invalid[0][1] = std::make_shared<typename operator_type::matrix_type>();
    operations->init_matrix( 1, 3, *invalid[0][1] );
    require_throw( [&] { operator_type bad( operations, invalid ); }, "Inconsistent block-row dimensions rejected" );
    invalid[0][1]->free();
    operations->init_matrix( 2, 2, *invalid[0][1] );
    require_throw( [&] { operator_type bad( operations, invalid ); }, "Inconsistent block-column dimensions rejected" );
    invalid[0][1].reset();
    require_throw( [&] { operator_type bad( operations, invalid ); }, "Null matrix block rejected" );
    require_throw( [&] { operator_type bad( nullptr, blocks ); }, "Null matrix operations rejected" );

    std::weak_ptr<operations_type>                     weak_operations;
    std::weak_ptr<typename operator_type::matrix_type> weak_block;
    std::shared_ptr<space_type>                        surviving_space;
    {
        auto owned_operations = std::make_shared<operations_type>();
        weak_operations       = owned_operations;
        typename operator_type::block_array_type owned_blocks;
        for ( std::size_t i = 0; i < 2; ++i )
            for ( std::size_t j = 0; j < 2; ++j )
            {
                owned_blocks[i][j] = std::make_shared<typename operator_type::matrix_type>();
                owned_operations->init_matrix( rows[i], columns[j], *owned_blocks[i][j] );
            }
        weak_block = owned_blocks[0][0];
        operator_type owned( owned_operations, owned_blocks );
        owned_operations.reset();
        owned_blocks    = {};
        surviving_space = owned.get_dom_space();
        require( !weak_operations.expired() && !weak_block.expired(), "Operator retains its dependencies" );
    }
    require( weak_operations.expired() && weak_block.expired(), "Operator and block ownership have no cycles" );
    require( surviving_space->size() == 4, "Returned domain survives operator destruction" );
}

template <class Backend>
void test_host_backend()
{
    test_vectors<Backend, float>();
    test_vectors<Backend, double>();
    test_vectors<Backend, std::complex<double>>();
    test_failure_cleanup<Backend>();
    test_matrices<nmfd::operations::dense_operations<float, Backend>>();
    test_matrices<nmfd::operations::dense_operations<double, Backend>>();
    test_matrices<nmfd::operations::dense_operations<std::complex<double>, Backend>>();
}

}

int main()
{
    try
    {
#ifdef TEST_GLUED_CUDA
        scfd::backend::cuda<std::ptrdiff_t>::init_device();
        scfd::cublas_wrap blas( false, true );
        using backend_type = scfd::backend::cuda<std::ptrdiff_t>;
        using complex_type = nmfd::operations::detail::scfd_backend_ext::complex_t<backend_type, double>;
        test_vectors<backend_type, float>();
        test_vectors<backend_type, double>();
        test_vectors<backend_type, complex_type>();
        test_failure_cleanup<backend_type>();
        test_matrices<nmfd::operations::dense_operations_cuda_hip<float, backend_type>>();
        test_matrices<nmfd::operations::dense_operations_cuda_hip<double, backend_type>>();
        test_matrices<nmfd::operations::dense_operations_cuda_hip<complex_type, backend_type>>();
        scfd::cublas_wrap::set_inst( nullptr );
#else
        test_host_backend<scfd::backend::serial_cpu<std::ptrdiff_t>>();
        test_host_backend<scfd::backend::omp<std::ptrdiff_t>>();
#endif
        std::cout << "Glued operations: " << checks << " checks, 0 failures\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
