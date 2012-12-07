#include <Python.h> // must be included first per Python C API rules
#include "Error.hpp"

#include <boost/python.hpp>

namespace jz {
namespace python {

Error::Error()
{
    // Taken from wiki.python.org/moin/boost.python/EmbeddingPython
    using namespace boost::python;
    PyObject *typeRaw, *valueRaw, *tracebackRaw;
    PyErr_Fetch(&typeRaw, &valueRaw, &tracebackRaw);
    PyErr_NormalizeException(&typeRaw, &valueRaw, &tracebackRaw);
    handle<> htype(typeRaw), hvalue(allow_null(valueRaw)), htraceback(allow_null(tracebackRaw));
    std::vector<std::string> traceback;
    // Push back an empty string so that the traceback gets printed on a new line
    // when boost::diagnostic_information is called so that all of the python lines
    // are lined up as they would in a regular interpreter
    traceback.push_back("");
    // Prevent throwing an exception
    if (!hvalue)
    {
        str typeError(htype);
        extract<std::string> errorValue(typeError);
        // Check if extraction worked
        if (errorValue.check())
        {
            traceback.push_back(errorValue());
        }
    }
    else
    {
        object tracebackModule(import("traceback"));
        object format_exception(tracebackModule.attr("format_exception"));
        list formatted_list(format_exception(htype,hvalue,htraceback));
        boost::python::ssize_t items = len(formatted_list);
        for (boost::python::ssize_t ii = 0; ii < items; ii++)
        {
            str line(formatted_list[ii]);
            extract<std::string> value(line);
            if (!value.check())
                continue;
            std::string parsed(value());
            // Formatted lines comes with the newline in them. Remove them
            // so that 'what' will work correctly and to_string(python_traceback)
            // will just add them back in
            std::string::size_type newlinePosition = parsed.find('\n');
            if (newlinePosition != std::string::npos)
            {
                parsed.erase(newlinePosition);
            }
            traceback.push_back(parsed);
        }
    }
    *this << traceback_list(traceback);
}

} // namespace python
} // namespace jz
