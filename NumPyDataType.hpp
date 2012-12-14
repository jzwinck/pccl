#pragma once

// NumPy Data Type (dtype) object creation in C++
// See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

#include <boost/python/list.hpp>

namespace pccl {
namespace python {

// builds a NumPy "dtype" data description to be used when constructing NumPy arrays
class NumPyDataType
{
public:
    // add a field by name with a scalar data type string
    // for valid codes, see http://docs.scipy.org/doc/numpy/reference/arrays.scalars.html#arrays-scalars-built-in
    void append(char const* fieldName, char const* typeCode);

    // as above but using a C++ type instead of a NumPy dtype
    template <typename T>
    void append(char const* fieldName);

    // as above but using the type of a pointer-to-member, like &MyClass::myField
    template <typename Class, typename Field>
    void append(char const* fieldName, Field Class::* field);

    // return the data type built so far as a Python list
    boost::python::list const& getList() const { return m_dtype; }

private:
    boost::python::list m_dtype;
};

// inline implementations

namespace detail {
template<typename T> struct Mapper {};
template<> struct Mapper<bool>          { static char const* getTypeCode() { return "?1"; } };
template<> struct Mapper<char>          { static char const* getTypeCode() { return "b1"; } };
template<> struct Mapper<unsigned char> { static char const* getTypeCode() { return "B1"; } };
template<> struct Mapper<int16_t>       { static char const* getTypeCode() { return "i2"; } };
template<> struct Mapper<uint16_t>      { static char const* getTypeCode() { return "u2"; } };
template<> struct Mapper<int32_t>       { static char const* getTypeCode() { return "i4"; } };
template<> struct Mapper<uint32_t>      { static char const* getTypeCode() { return "u4"; } };
template<> struct Mapper<int64_t>       { static char const* getTypeCode() { return "i8"; } };
template<> struct Mapper<uint64_t>      { static char const* getTypeCode() { return "u8"; } };
template<> struct Mapper<float>         { static char const* getTypeCode() { return "f4"; } };
template<> struct Mapper<double>        { static char const* getTypeCode() { return "f8"; } };
} // namespace detail

template <typename T>
void NumPyDataType::append(char const* fieldName)
{
    append(fieldName, detail::Mapper<T>::getTypeCode());
}

template <typename Class, typename Field>
void NumPyDataType::append(char const* fieldName, Field Class::* field)
{
    append<Field>(fieldName);
}

} // namespace python
} // namespace pccl
