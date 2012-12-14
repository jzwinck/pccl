#define BOOST_TEST_MODULE InterpreterLockGuard
#define BOOST_TEST_DYN_LINK

#include <Python.h>
#include <boost/test/unit_test.hpp>
#include "../InterpreterLockGuard.hpp"

BOOST_AUTO_TEST_SUITE(test_InterpreterLockGuard)

BOOST_AUTO_TEST_CASE(test1)
{
    {
        // test building a guard without Python threads initialized
        // this isn't a common thing to do, but we do support it "just in case"
        pccl::python::InterpreterLockGuard guard1;
    }

    {
        Py_InitializeEx(false);
        pccl::python::InterpreterLockGuard guard2;
    }
}

BOOST_AUTO_TEST_SUITE_END()
