#pragma once

// NumPy Data Type (dtype) object creation in C++
// See http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html

#include <boost/python/list.hpp>

namespace jz {
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
template<> struct Mapper<bool>          { static constexpr char const* typeCode = "?1"; };
template<> struct Mapper<char>          { static constexpr char const* typeCode = "b1"; };
template<> struct Mapper<unsigned char> { static constexpr char const* typeCode = "B1"; };
template<> struct Mapper<int16_t>       { static constexpr char const* typeCode = "i2"; };
template<> struct Mapper<uint16_t>      { static constexpr char const* typeCode = "u2"; };
template<> struct Mapper<int32_t>       { static constexpr char const* typeCode = "i4"; };
template<> struct Mapper<uint32_t>      { static constexpr char const* typeCode = "u4"; };
template<> struct Mapper<int64_t>       { static constexpr char const* typeCode = "i8"; };
template<> struct Mapper<uint64_t>      { static constexpr char const* typeCode = "u8"; };
template<> struct Mapper<float>         { static constexpr char const* typeCode = "f4"; };
template<> struct Mapper<double>        { static constexpr char const* typeCode = "f8"; };
} // namespace detail

template <typename T>
void NumPyDataType::append(char const* fieldName)
{
    append(fieldName, detail::Mapper<T>::typeCode);
}

template <typename Class, typename Field>
void NumPyDataType::append(char const* fieldName, Field Class::* field)
{
    append<Field>(fieldName);
}

} // namespace python
} // namespace jz
