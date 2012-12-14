#include "terminate.hpp"

#include <cxxabi.h>

#include <boost/exception/diagnostic_information.hpp>

namespace pccl {
namespace error {
namespace detail {
namespace {

void handleTerminate()
{
    // std::uncaught_exception doesn't work in this context, so we use a GCC detail
    if (abi::__cxa_current_exception_type() != NULL) // we're here because of a C++ exception
    {
        try
        {
            throw;
        }
        catch (boost::exception const& ex)
        {
            // a boost::exception's what() may not tell the whole story, but here we do
            fputs("terminate called after throwing an instance of 'boost::exception'\n", stderr);
            fputs(diagnostic_information(ex).c_str(), stderr);
            abort();
        }
        catch (...)
        {
        }
    }

    // fall back to the default
    __gnu_cxx::__verbose_terminate_handler();
}

} // namespace

TerminateGuard::TerminateGuard()
{
    std::set_terminate(handleTerminate);
}

} // namespace detail
} // namespace error
} // namespace pccl
