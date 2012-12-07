#pragma once

#include <Python.h>

// The NumPy C API is crazy, and requires special magic to #include its header file in multiple translation units.
// see http://stackoverflow.com/a/12259667 and http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL JZ_PYTHON_NUMPY_ARRAY_UNIQUE_SYMBOL
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define NO_IMPORT_ARRAY // this is here for the benefit of modules that #include this file, *not* for NumPy.cpp
#include <numpy/arrayobject.h>

namespace jz {
namespace python {

// Call NumPy's import_array() to allow the running process to use the NumPy C API.
// This function exists because the native API is crazy in terms of error handling and the reasons above.
void initializeNumPy();

} // namespace python
} // namespace jz
