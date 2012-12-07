#pragma once

/*
  python::Error - a C++ exception to throw when Python reports an error.

  Use with Boost Python like this:

  try {
      // do things using Boost Python
  } catch (const boost::python::error_already_set &) {
      throw python::Error();
  }

  This class fetches error information stored by Python, stores it,
  and provides an idiomatic C++ exception to throw instead.
*/

#include <exception>
#include <string>
#include <vector>
#include <boost/exception/all.hpp>

namespace jz {
namespace python {

class Error : public std::exception, public boost::exception
{
public:
    Error();

    virtual ~Error() throw() {}

    typedef boost::error_info<struct traceback_list_t, std::vector<std::string> > traceback_list;

    virtual const char *what() const throw()
    {
        // Just return the last line of the traceback
        std::vector<std::string> const* value =
                boost::get_error_info<traceback_list>(*this);
        if (value != 0)
            return value->back().c_str();
        else
            return "Unknown";
    }

    virtual const char *traceback() const throw()
    {
#if BOOST_VERSION >= 104100
        return boost::diagnostic_information_what(*this);
#else
        // Default to what with older versions
        return what();
#endif
    }
};

} // namespace python
} // namespace jz

/*
 *  Create an error_info object for the traceback so that it can be printed
 *  out
 *
 */
namespace boost
{
    inline
    std::string
    to_string(jz::python::Error::traceback_list const& traceback)
    {
        std::string result;
        if (traceback.value().empty())
            return result;
        result += traceback.value()[0];
        for (std::vector<std::string>::size_type ii = 1; ii < traceback.value().size(); ii++)
        {
            result += "\n" + traceback.value()[ii];
        }
        return result;
    }
}
