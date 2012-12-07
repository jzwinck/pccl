#pragma once

// exception throwing macros with stream formatting support
// example usages:
// jz_throw("something went wrong: ", reason);
// jz_unless(rc == 0, jz_throw_errno("error in recv()"));

#include <stdexcept>
#include <cerrno>
#include <boost/preprocessor/stringize.hpp>
#include <cstring>

#define JZ_FULL_SOURCE_LOCATION __FILE__ ":" BOOST_PP_STRINGIZE(__LINE__)
#define JZ_SHORT_SOURCE_LOCATION ((::strrchr(JZ_FULL_SOURCE_LOCATION, '/') ? : JZ_FULL_SOURCE_LOCATION- 1) + 1)
#define JZ_SOURCE_LOCATION JZ_SHORT_SOURCE_LOCATION
#define JZ_PATH JZ_FULL_SOURCE_LOCATION
#define JZ_FILELINE JZ_SHORT_SOURCE_LOCATION

#include <boost/exception/errinfo_errno.hpp>
#include <boost/throw_exception.hpp>
#include "likely.h"
#include "terminate.hpp" // sets up std::terminate() handler

#define JZ_THROW_IMPL BOOST_THROW_EXCEPTION

// throw the given type of exception
#define jz_throw_type(_type_, ...) \
    JZ_THROW_IMPL(jz::error::detail::makeException<_type_>(__VA_ARGS__))

// throw a std::runtime_error
#define jz_throw(...) \
    jz_throw_type(std::runtime_error, __VA_ARGS__)

// throw a std::logic_error
// this is appropriate when something "impossible" has happened,
// implying that the program is structurally deficient
#define jz_throw_logic(...) \
    jz_throw_type(std::logic_error, __VA_ARGS__)

// throw a std::logic_error for unimplemented functionality
// this is appropriate for cases where it was not intended that a portion of an
// interface would be called, the fact that it was called means that the
// program is structurally deficient
#define jz_throw_unimplemented() \
    jz_throw_type(std::logic_error, __PRETTY_FUNCTION__, " is unimplemented.")

// throw a std::runtime_error with the global errno attached as boost::errinfo_errno
#define jz_throw_errno(...) do {                                        \
        BOOST_ASSERT(errno != 0);                                       \
        boost::errinfo_errno _info_(errno);                             \
        std::runtime_error _ex_ =                                       \
            jz::error::detail::makeException<std::runtime_error>(__VA_ARGS__); \
        JZ_THROW_IMPL(boost::enable_error_info(_ex_) << _info_);        \
    } while (false)

// deprecated exception-throwing macros
// these are like the above, but use left-shift (<<) rather than comma between arguments

#define JZ_THROW_TYPE(_type_, _stream_) do { \
        std::ostringstream _what_;           \
        _what_ << _stream_;                  \
        JZ_THROW_IMPL(_type_(_what_.str())); \
    } while (false)

#define JZ_THROW(_stream_) do {                      \
        JZ_THROW_TYPE(std::runtime_error, _stream_); \
    } while (false)

#define JZ_THROW_LOGIC(_stream_) do {              \
        JZ_THROW_TYPE(std::logic_error, _stream_); \
    } while (false)

#define JZ_THROW_ERRNO(_stream_) do {                            \
        BOOST_ASSERT(errno != 0);                                \
        boost::errinfo_errno _info_(errno);                      \
        std::ostringstream _what_;                               \
        _what_ << _stream_;                                      \
        std::runtime_error _ex_(_what_.str());                   \
        JZ_THROW_IMPL(boost::enable_error_info(_ex_) << _info_); \
    } while (false)

// if an unlikely condition is not met, do something, like throw
#define JZ_UNLESS(_assertion_, _action_) do { \
        if (unlikely(!(_assertion_))) {       \
            _action_;                         \
        }                                     \
    } while (false)

// if an unlikely condition is met, do something, like throw
#define JZ_IF(_assertion_, _action_) do { \
        if (unlikely(_assertion_)) {      \
            _action_;                     \
        }                                 \
    } while (false)

// support lowercase jz_unless() to go along with jz_throw()
#define jz_unless JZ_UNLESS
#define jz_if JZ_IF

// inline implementations

namespace jz {
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
} // namespace jz
