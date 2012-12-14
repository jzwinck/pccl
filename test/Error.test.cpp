#include <Python.h>

#define BOOST_TEST_MODULE Error
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/foreach.hpp>
#include <boost/python.hpp>
#include <boost/version.hpp>
#include "../Error.hpp"

BOOST_AUTO_TEST_SUITE(test_Error)

BOOST_AUTO_TEST_CASE(test_raise_with_value)
{
    using namespace pccl::python;
    using namespace boost::python;
    Py_InitializeEx(false);
    try
    {
        object main_module(handle<>(borrowed(PyImport_AddModule("__main__"))));
        object main_namespace = main_module.attr("__dict__");
        exec_statement("raise NameError", main_namespace, main_namespace);
        BOOST_FAIL("Expected a Python exception to be thrown");
    }
    catch (error_already_set& error)
    {
        Error description;
        description << boost::throw_function(BOOST_CURRENT_FUNCTION);
        description << boost::throw_line(__LINE__);
        BOOST_CHECK_EQUAL("NameError", description.what());
        BOOST_CHECK_EQUAL(
                "Throw in function void test_Error::test_raise_with_value::test_method()\n"
#if BOOST_VERSION < 104300 // older Boost versions don't demangle C++ type names
                "Dynamic exception type: N4pccl6python5ErrorE\n"
                "std::exception::what: NameError\n"
                "[PN4pccl6python16traceback_list_tE] = \n"
#else
                "Dynamic exception type: pccl::python::Error\n"
                "std::exception::what: NameError\n"
                "[pccl::python::traceback_list_t*] = \n"
#endif
                "Traceback (most recent call last):\n"
                "  File \"<string>\", line 1, in <module>\n"
                "NameError\n",
                boost::diagnostic_information(description));
    }
}

BOOST_AUTO_TEST_SUITE_END()
