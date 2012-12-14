#include "NumPyArray.hpp"

#include "NumPy.hpp"
#include "Error.hpp"
#include "throw.hpp"

namespace pccl {
namespace python {
namespace {

typedef int (*DescrConverter)(PyObject*, PyArray_Descr**);
typedef PyObject* (*ArrayCreator)(int, npy_intp*, PyArray_Descr*, int);

boost::python::object makeNumPyArray(boost::python::list const& dtype, unsigned count, DescrConverter converter, ArrayCreator creator)
{
    // first turn a list of (name, dtype) string pairs into a NumPy data type object
    PyArray_Descr* descr;
    int rc = converter(dtype.ptr(), &descr);
    pccl_unless(rc == 1, throw Error());

    npy_intp dimension = count;
    boost::python::handle<> arrayHandle(creator(1, &dimension, descr, 0/*fortran*/));
    pccl_unless(arrayHandle, throw Error());
    return boost::python::object(arrayHandle);
}

boost::python::object makeNumPyArrayWithData(boost::python::list const& dtype, unsigned count, void* data, DescrConverter converter)
{
    PyArray_Descr* descr;
    int rc = converter(dtype.ptr(), &descr); // aligned or not, depending on the function passed in
    pccl_unless(rc == 1, throw Error());

    npy_intp dimension = count;
    boost::python::handle<> arrayHandle(PyArray_NewFromDescr(&PyArray_Type, descr, 1, &dimension, NULL, data, 0/*flags*/, NULL/*init*/));
    return boost::python::object(arrayHandle);
}

} // namespace

boost::python::object makeNumPyArrayEmpty(boost::python::list const& dtype, unsigned count)
{
    initializeNumPy();
    return makeNumPyArray(dtype, count, PyArray_DescrAlignConverter2, PyArray_Empty);
}

boost::python::object makeNumPyArrayZeros(boost::python::list const& dtype, unsigned count)
{
    initializeNumPy();
    return makeNumPyArray(dtype, count, PyArray_DescrAlignConverter2, PyArray_Zeros);
}

boost::python::object makeNumPyArrayWithData(boost::python::list const& dtype, unsigned count, void* data)
{
    initializeNumPy();
    return makeNumPyArrayWithData(dtype, count, data, PyArray_DescrAlignConverter2);
}

boost::python::object makeNumPyArrayEmptyPacked(boost::python::list const& dtype, unsigned count)
{
    initializeNumPy();
    return makeNumPyArray(dtype, count, PyArray_DescrConverter2, PyArray_Empty);
}

boost::python::object makeNumPyArrayZerosPacked(boost::python::list const& dtype, unsigned count)
{
    initializeNumPy();
    return makeNumPyArray(dtype, count, PyArray_DescrConverter2, PyArray_Zeros);
}

boost::python::object makeNumPyArrayWithDataPacked(boost::python::list const& dtype, unsigned count, void* data)
{
    initializeNumPy();
    return makeNumPyArrayWithData(dtype, count, data, PyArray_DescrConverter2);
}

} // namespace python
} // namespace pccl
