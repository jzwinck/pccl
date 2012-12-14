#include "NumPyDataType.hpp"

#include <boost/python/tuple.hpp>

namespace pccl {
namespace python {

void NumPyDataType::append(char const* fieldName, char const* typeCode)
{
    m_dtype.append(boost::python::make_tuple(fieldName, typeCode));
}

} // namespace python
} // namespace pccl
