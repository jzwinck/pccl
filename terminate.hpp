#pragma once

// including this file causes full diagnostic information to be printed for boost::exceptions
// this is meant to be included by throw.hpp and probably should not be used elsewhere

namespace jz {
namespace error {
namespace detail {

class TerminateGuard
{
public:
    TerminateGuard();
};

// this function will be called before main()
// what's important is just that every program calls std::set_terminate early on
void ensureTerminateGuard() __attribute__((constructor)); // called before main()

inline void ensureTerminateGuard()
{
    static TerminateGuard const guard;
}

} // namespace detail
} // namespace error
} // namespace jz
