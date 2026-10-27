#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include <nmfd/operations/pair_vector_space.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <scfd/backend/serial_cpu.h>

namespace
{

unsigned checks = 0;

void require( bool condition, const char *message )
{
    ++checks;
    if ( !condition )
        throw std::runtime_error( message );
}

struct tracked_space : nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<>, double>
{
    using parent_t          = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<>, double>;
    mutable int initialized = 0, active = 0, init_calls = 0, start_calls = 0;
    int         fail_init = 0, fail_start = 0;

    tracked_space() : parent_t( 2 )
    {
    }
    void init_vector( vector_type &x ) const override
    {
        if ( ++init_calls == fail_init )
            throw std::runtime_error( "Injected initialization failure" );
        parent_t::init_vector( x );
        ++initialized;
    }
    void start_use_vector( vector_type &x ) const override
    {
        if ( ++start_calls == fail_start )
            throw std::runtime_error( "Injected activation failure" );
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
        parent_t::free_vector( x );
        --initialized;
    }
};

using product_type = nmfd::operations::product_vector_space<tracked_space, tracked_space>;

void test_ownership()
{
    std::weak_ptr<tracked_space> first_weak, second_weak;
    {
        auto first = std::make_shared<tracked_space>(), second = std::make_shared<tracked_space>();
        first_weak  = first;
        second_weak = second;
        product_type space( first, second );
        first.reset();
        second.reset();
        product_type copy = space;
        require( !first_weak.expired() && !second_weak.expired(), "Product retains both spaces" );
        require( copy.size() == 4, "Copy retains live component spaces" );
    }
    require( first_weak.expired() && second_weak.expired(), "Product releases both spaces without a cycle" );

    auto                                                              first = std::make_shared<tracked_space>();
    nmfd::operations::pair_vector_space<tracked_space, tracked_space> pair( first, first );
    require( pair.first() == first && pair.second() == first, "Pair may share one component space" );
    try
    {
        product_type invalid( first, nullptr );
        require( false, "Null shared component must be rejected" );
    }
    catch ( const std::invalid_argument & )
    {
        require( first->initialized == 0, "Null rejection precedes vector initialization" );
    }
}

void test_multivector_contract()
{
    auto         first = std::make_shared<tracked_space>(), second = std::make_shared<tracked_space>();
    product_type space( first, second );
    product_type::multivector_type vectors( 2 );
    space.init_multivector( vectors, 2 );
    require( first->initialized == 2 && second->initialized == 2, "Pre-sized uninitialized storage remains supported" );
    try
    {
        space.init_multivector( vectors, -1 );
        require( false, "Negative counts must be rejected" );
    }
    catch ( const std::invalid_argument & )
    {
        require( vectors.size() == 2 && first->initialized == 2, "Invalid count preserves existing storage" );
    }
    product_type::vector_type out;
    try
    {
        space.assign( vectors, 2, 2, out );
        require( false, "Out-of-range columns must be rejected before accessing vectors" );
    }
    catch ( const std::out_of_range & )
    {
        require( true, "Column bounds are checked" );
    }
    space.free_multivector( vectors, 2 );
    require( first->initialized == 0 && second->initialized == 0, "Pre-sized basis releases all resources" );
    space.init_multivector( vectors, 0 );
    space.start_use_multivector( vectors, 0 );
    space.stop_use_multivector( vectors, 0 );
    space.free_multivector( vectors, 0 );
    require( vectors.empty(), "Empty multivectors are supported" );
}

// Exercise failure at each component and each position for single, batch, and basis storage.
void test_failure( int kind, int position, bool second_component, bool activation )
{
    auto         first = std::make_shared<tracked_space>(), second = std::make_shared<tracked_space>();
    product_type space( first, second );
    auto        &failing = second_component ? *second : *first;
    const int    count   = kind == 0 ? 1 : 3;
    std::vector<product_type::vector_type> vectors;
    if ( kind != 2 )
        vectors.resize( count );
    if ( activation )
    {
        if ( kind == 2 )
            space.init_multivector( vectors, count );
        else if ( kind == 1 )
            space.init_vectors( vectors[0], vectors[1], vectors[2] );
        else
            space.init_vector( vectors[0] );
        failing.fail_start = position;
    }
    else
        failing.fail_init = position;
    bool threw = false;
    try
    {
        if ( kind == 2 )
        {
            if ( activation )
                space.start_use_multivector( vectors, count );
            else
                space.init_multivector( vectors, count );
        }
        else if ( kind == 1 )
        {
            if ( activation )
                space.start_use_vectors( vectors[0], vectors[1], vectors[2] );
            else
                space.init_vectors( vectors[0], vectors[1], vectors[2] );
        }
        else if ( activation )
            space.start_use_vector( vectors[0] );
        else
            space.init_vector( vectors[0] );
    }
    catch ( const std::runtime_error & )
    {
        threw = true;
    }
    require( threw, "Injected component failure must propagate" );
    require( first->active == 0 && second->active == 0, "Failure rolls back every completed activation" );
    if ( activation )
        for ( auto &vector : vectors )
            space.free_vector( vector );
    else if ( kind == 2 )
        require( vectors.empty(), "Failed basis initialization leaves empty storage" );
    require( first->initialized == 0 && second->initialized == 0, "Failure leaves no vector resources" );
}

}

int main()
{
    try
    {
        test_ownership();
        test_multivector_contract();
        for ( int kind : { 0, 1, 2 } )
            for ( int position = 1; position <= ( kind == 0 ? 1 : 3 ); ++position )
                for ( bool second : { false, true } )
                    for ( bool activation : { false, true } )
                        test_failure( kind, position, second, activation );
        std::cout << "Product lifetime: " << checks << " checks passed\n";
        return 0;
    }
    catch ( const std::exception &error )
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
