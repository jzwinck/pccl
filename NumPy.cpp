// This file intentionally does not include its corresponding header at the top - see comments in the header
// If this isn't done properly, you'll see either that import_array1(), used below, is not found, or a linker error.

#include <Python.h>

#define PY_ARRAY_UNIQUE_SYMBOL JZ_PYTHON_NUMPY_ARRAY_UNIQUE_SYMBOL
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "NumPy.hpp"
#include "Error.hpp"

namespace jz {
namespace python {
namespace {

inline bool importNumPy()
{
    // wacky NumPy API: import_array1() is a macro which actually calls return with the given value
    import_array1(false);
    return true;
}

} // namespace

void initializeNumPy()
{
    if (!importNumPy())
        throw Error();
}

} // namespace python
} // namespace jz
