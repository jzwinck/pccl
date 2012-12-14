#pragma once

// exception throwing macros with stream formatting support
// example usages:
// pccl_throw("something went wrong: ", reason);
// pccl_unless(rc == 0, pccl_throw_errno("error in recv()"));

#include <stdexcept>
#include <cerrno>
#include <boost/preprocessor/stringize.hpp>
#include <cstring>

#define PCCL_FULL_SOURCE_LOCATION __FILE__ ":" BOOST_PP_STRINGIZE(__LINE__)
#define PCCL_SHORT_SOURCE_LOCATION ((::strrchr(PCCL_FULL_SOURCE_LOCATION, '/') ? : PCCL_FULL_SOURCE_LOCATION- 1) + 1)
#define PCCL_SOURCE_LOCATION PCCL_SHORT_SOURCE_LOCATION
#define PCCL_PATH PCCL_FULL_SOURCE_LOCATION
#define PCCL_FILELINE PCCL_SHORT_SOURCE_LOCATION

#include <boost/exception/errinfo_errno.hpp>
#include <boost/throw_exception.hpp>
#include "likely.h"
#include "terminate.hpp" // sets up std::terminate() handler

#define PCCL_THROW_IMPL BOOST_THROW_EXCEPTION

// throw the given type of exception
#define pccl_throw_type(_type_, ...) \
    PCCL_THROW_IMPL(pccl::error::detail::makeException<_type_>(__VA_ARGS__))

// throw a std::runtime_error
#define pccl_throw(...) \
    pccl_throw_type(std::runtime_error, __VA_ARGS__)

// throw a std::logic_error
// this is appropriate when something "impossible" has happened,
// implying that the program is structurally deficient
#define pccl_throw_logic(...) \
    pccl_throw_type(std::logic_error, __VA_ARGS__)

// throw a std::logic_error for unimplemented functionality
// this is appropriate for cases where it was not intended that a portion of an
// interface would be called, the fact that it was called means that the
// program is structurally deficient
#define pccl_throw_unimplemented() \
    pccl_throw_type(std::logic_error, __PRETTY_FUNCTION__, " is unimplemented.")

// throw a std::runtime_error with the global errno attached as boost::errinfo_errno
#define pccl_throw_errno(...) do {                                        \
        BOOST_ASSERT(errno != 0);                                       \
        boost::errinfo_errno _info_(errno);                             \
        std::runtime_error _ex_ =                                       \
            pccl::error::detail::makeException<std::runtime_error>(__VA_ARGS__); \
        PCCL_THROW_IMPL(boost::enable_error_info(_ex_) << _info_);        \
    } while (false)

// deprecated exception-throwing macros
// these are like the above, but use left-shift (<<) rather than comma between arguments

#define PCCL_THROW_TYPE(_type_, _stream_) do { \
        std::ostringstream _what_;           \
        _what_ << _stream_;                  \
        PCCL_THROW_IMPL(_type_(_what_.str())); \
    } while (false)

#define PCCL_THROW(_stream_) do {                      \
        PCCL_THROW_TYPE(std::runtime_error, _stream_); \
    } while (false)

#define PCCL_THROW_LOGIC(_stream_) do {              \
        PCCL_THROW_TYPE(std::logic_error, _stream_); \
    } while (false)

#define PCCL_THROW_ERRNO(_stream_) do {                            \
        BOOST_ASSERT(errno != 0);                                \
        boost::errinfo_errno _info_(errno);                      \
        std::ostringstream _what_;                               \
        _what_ << _stream_;                                      \
        std::runtime_error _ex_(_what_.str());                   \
        PCCL_THROW_IMPL(boost::enable_error_info(_ex_) << _info_); \
    } while (false)

// if an unlikely condition is not met, do something, like throw
#define PCCL_UNLESS(_assertion_, _action_) do { \
        if (unlikely(!(_assertion_))) {       \
            _action_;                         \
        }                                     \
    } while (false)

// if an unlikely condition is met, do something, like throw
#define PCCL_IF(_assertion_, _action_) do { \
        if (unlikely(_assertion_)) {      \
            _action_;                     \
        }                                 \
    } while (false)

// support lowercase pccl_unless() to go along with pccl_throw()
#define pccl_unless PCCL_UNLESS
#define pccl_if PCCL_IF

// inline implementations

namespace pccl {
namespace error {
namespace detail {

// terminating case for variadic template recursion
inline void format(std::ostream&)
{
    // no arguments remain, nothing to do here
}

// formats 1 or more arguments to the given stream
template <typename First, typename... Rest>
void format(std::ostream& out, First const& first, Rest const&... rest)
{
    format(out << first, rest...);
}

// construct an exception of the given type with Args printed in what()
template <typename Exception, typename... Args>
Exception makeException(Args const&... args)
{
    std::ostringstream what;
    format(what, args...);
    return Exception(what.str());
}

} // namespace detail
} // namespace error
} // namespace pccl
