#include "run.hpp"

#include <boost/python.hpp>

#include "throw.hpp"
#include "Error.hpp"

namespace jz {
namespace python {
namespace {

boost::python::object makeNamespace()
{
    Py_InitializeEx(false/*signals*/);

    using namespace boost::python;
    object mainModule(handle<>(borrowed(PyImport_AddModule("__main__"))));
    object mainNamespace = mainModule.attr("__dict__");

    return mainNamespace;
}

boost::python::object const& runFile(boost::python::object const& mainNamespace, char const* filename)
try
{
    FILE *file = fopen(filename, "r");
    jz_unless(file, jz_throw_errno("failed to open file: ", filename));

    // capture the result of PyRun_* so it is cleaned up later
    boost::python::handle<>(PyRun_FileEx(file, filename, Py_file_input,
                                         mainNamespace.ptr(), mainNamespace.ptr(),
                                         true/*close*/));

    return mainNamespace;
}
catch (boost::python::error_already_set const &)
{
    BOOST_THROW_EXCEPTION(python::Error());
}

} // namespace

boost::python::object runFile(char const* filename)
{
    boost::python::object mainNamespace = makeNamespace();

    return runFile(mainNamespace, filename);
}

} // namespace python
} // namespace jz
