#define BOOST_TEST_MODULE NumPyArray
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <boost/python.hpp>

#include "../NumPy.hpp"
#include "../NumPyArray.hpp"
#include "../NumPyDataType.hpp"
#include "../Error.hpp"

struct TemperatureRecord
{
    uint32_t epochTime;
    uint32_t sensorId;
    double temperature;
};

BOOST_AUTO_TEST_SUITE(test_NumPyArray)

BOOST_AUTO_TEST_CASE(test_raise_with_value)
{
    using namespace pccl::python;

    Py_InitializeEx(false/*signals*/);

    initializeNumPy();

    NumPyDataType dtype;
    dtype.append("epochTime", "u4");
    dtype.append<uint32_t>("sensorId");
    dtype.append("temperature", &TemperatureRecord::temperature);

    try
    {
        boost::python::object array = makeNumPyArrayZeros(dtype.getList(), 10);
        BOOST_REQUIRE(PyArray_CheckExact(array.ptr()));
        BOOST_CHECK_EQUAL(boost::python::extract<double>(array["temperature"][0]), 0);
    }
    catch (boost::python::error_already_set const&)
    {
        throw Error();
    }
}

BOOST_AUTO_TEST_SUITE_END()
