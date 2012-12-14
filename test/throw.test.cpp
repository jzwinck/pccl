#define BOOST_TEST_MODULE test_throw
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <sys/socket.h>

#include <boost/exception/get_error_info.hpp>

#include "../throw.hpp"

BOOST_AUTO_TEST_SUITE(test_throw)

BOOST_AUTO_TEST_CASE(test_SOURCE_LOCATION)
{
    BOOST_CHECK_EQUAL(PCCL_SHORT_SOURCE_LOCATION, "throw.test.cpp:15"); // if this line number changes, the string will need to change too
}

BOOST_AUTO_TEST_CASE(test_throw)
{
    BOOST_CHECK_THROW(pccl_throw("foo"), std::runtime_error);

    try
    {
        pccl_throw("bar");
    }
    catch (const std::runtime_error &ex)
    {
        BOOST_CHECK_EQUAL(ex.what(), "bar");
    }
}

BOOST_AUTO_TEST_CASE(test_Errno)
{
    try
    {
        int rc = shutdown(STDIN_FILENO, SHUT_WR);
        BOOST_REQUIRE(rc == -1); // errno should now be set
        BOOST_REQUIRE_EQUAL(errno, ENOTSOCK);
        pccl_throw_errno("baz");
    }
    catch (const std::runtime_error &ex)
    {
        BOOST_CHECK_EQUAL(ex.what(), "baz");
        const int *val = boost::get_error_info<boost::errinfo_errno>(ex);
        BOOST_REQUIRE(val != NULL);
        BOOST_CHECK_EQUAL(*val, ENOTSOCK);
        return;
    }

    BOOST_ERROR("no exception was thrown where one was expected");
}

BOOST_AUTO_TEST_SUITE_END()
