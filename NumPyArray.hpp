#pragma once

// create NumPy "ndarrays" from C++
// See http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html
// See NumPyDataType.hpp for a way to create the "dtype" lists used below
// If you have a type Foo which matches the array's dtype, you can access elements of the created arrays like this:
//    Foo* record = static_cast<Foo*>(PyArray_GETPTR1((PyArrayObject*)array.ptr(), index));

#include <boost/python/list.hpp>

namespace jz {
namespace python {

// builds a NumPy array with standard C alignment rules for the fields
// count is the number of elements
// the contents will be uninitialized
boost::python::object makeNumPyArrayEmpty(boost::python::list const& dtype, unsigned count);

// as above but initializes the array to all zeros
boost::python::object makeNumPyArrayZeros(boost::python::list const& dtype, unsigned count);

// as above, but use already-allocated storage for the array (it will not be initialized, and its format must match the given dtype)
boost::python::object makeNumPyArrayWithData(boost::python::list const& dtype, unsigned count, void* data);

// as above, but builds a "packed" array a la GCC's __attribute__((__packed__))
boost::python::object makeNumPyArrayEmptyPacked(boost::python::list const& dtype, unsigned count);
boost::python::object makeNumPyArrayZerosPacked(boost::python::list const& dtype, unsigned count);
boost::python::object makeNumPyArrayWithDataPacked(boost::python::list const& dtype, unsigned count, void* data);

} // namespace python
} // namespace jz
