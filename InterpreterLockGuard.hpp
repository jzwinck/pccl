#pragma once

#include <Python.h>
#include <boost/noncopyable.hpp>

namespace jz {
namespace python {

// When a C++ function is called by Python and it will be long-running,
// it should release the Global Interpreter Lock (GIL) by constructing one of these.
// This does not mean the C++ function needs to do this when invoked directly from C++:
// it should instead have a wrapper that sits in front of its Python binding, like this:
//
// void MyClass::longRunningFunction(); // native C++ code
//
// void myWrapper(MyClass& my) // local function in Python bindings module
// {
//     InterpreterLockGuard guard;
//     my.longRunningFunction();
// }
//
// // within BOOST_PYTHON_MODULE():
//     class_<MyClass>("MyClass")
//         .def("longRunningFunction", myWrapper);
//
// See http://wiki.python.org/moin/boost.python/HowTo#Multithreading_Support_for_my_function
class InterpreterLockGuard
    : boost::noncopyable
{
public:
    InterpreterLockGuard();
    ~InterpreterLockGuard();
private:
    bool m_doRelease;
};

class InterpreterLockAcquirer
    : boost::noncopyable
{
public:
    InterpreterLockAcquirer();
    ~InterpreterLockAcquirer();

private:
    PyThreadState* m_threadState;
};

} // namespace python
} // namespace jz
