#pragma once

#include <boost/python/object.hpp>

namespace pccl {
namespace python {

// given the name of a file containing Python text, run it and return the resulting top-level namespace
boost::python::object runFile(char const* filename);

// inline implementations

namespace detail {

boost::python::object makeNamespace();
boost::python::object const& runFile(boost::python::object const& mainNamespace, char const* filename);

} // namespace detail
} // namespace python
} // namespace pccl
