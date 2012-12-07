#include "InterpreterLockGuard.hpp"

#include <boost/assert.hpp>
#include <boost/thread/tss.hpp>

namespace jz {
namespace python {

namespace {

void cleanup(PyThreadState*)
{
}

boost::thread_specific_ptr<PyThreadState> g_threadState(cleanup);
}

InterpreterLockGuard::InterpreterLockGuard()
    : m_doRelease(false)
{
    if (!g_threadState.get() && PyEval_ThreadsInitialized())
    {
        g_threadState.reset(PyEval_SaveThread());
        m_doRelease = true;
    }
}

InterpreterLockGuard::~InterpreterLockGuard()
{
    if (m_doRelease)
    {
        if (g_threadState.get())
        {
            PyEval_RestoreThread(g_threadState.get());
            g_threadState.reset();
        }
        else
        {
            // This should never happen.
            BOOST_ASSERT(false);
        }
    }
}

InterpreterLockAcquirer::InterpreterLockAcquirer()
    : m_threadState(g_threadState.get())
{
    if (m_threadState)
    {
        PyEval_RestoreThread(m_threadState);
        g_threadState.reset();
    }
}

InterpreterLockAcquirer::~InterpreterLockAcquirer()
{
    if (m_threadState)
    {
        BOOST_ASSERT(!g_threadState.get());
        g_threadState.reset(PyEval_SaveThread());
    }
}

} // namespace python
} // namespace jz
