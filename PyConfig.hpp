#pragma once

/*
PyConfig lets you write configurations for C++ programs as Python dictionaries.
You write C++ structs containing your configuration items, implement a single
static method to express the names that should be mapped in Python.  PyConfig
then lets you load such structs from a text file (or boost::python::object).

PyConfig already supports some standard types like int, string, map, and list.
Support for more types can be added by implementing more conversion routines.

PyConfig-derived classes support composition automatically, so you can include
them as members of other PyConfig classes.

Make sure you have boost-devel version 1.40 or later installed (earlier versions
are not known to work).
*/

#include <tr1/array>
#include <list>
#include <map>
#include <set>
#include <vector>
#include <tr1/unordered_map>
#include <unordered_map>
#include <utility> // std::pair

#include <boost/make_shared.hpp>
#include <boost/python.hpp>
#include <boost/assert.hpp>
#include <boost/foreach.hpp>
#include <boost/mpl/if.hpp>
#include <boost/optional.hpp>
#include <boost/type_traits.hpp>
#include <boost/unordered_map.hpp>

#include "throw.hpp"
#include "Error.hpp"
#include "run.hpp"

#include <boost/python/stl_iterator.hpp>
#include <Python.h>

namespace pccl {

struct PyConfigBase {};

namespace PyConfig_detail {

template <typename T>
T convert(const boost::python::object &);

template <typename T>
struct Converter
{
    T operator()(const boost::python::object &value)
    {
        return boost::python::extract<T>(value);
    }
};

template <typename First, typename Second>
struct Converter<std::pair<First, Second> >
{
    std::pair<First, Second> operator()(const boost::python::object &value)
    {
        using namespace boost::python;
        boost::python::tuple pair = extract<boost::python::tuple>(value);
        BOOST_ASSERT(len(pair) == 2);
        return std::pair<First, Second>(
            convert<First>(pair[0]), convert<Second>(pair[1]));
    }
};

template <typename T>
struct Converter<boost::optional<T> >
{
    boost::optional<T> operator()(const boost::python::object &value)
    {
        if (!value)
        {
            return boost::none;
        }

        return convert<T>(value);
    }
};


template<class ArrayType>
void convertArray(ArrayType&result, boost::python::object const&value)
{
    ::ssize_t ii = 0;
    ::ssize_t const maxLen = result.size();
    for (boost::python::stl_input_iterator<boost::python::object > b(value), e; b != e; ++ii, ++b)
    {
        if (ii == maxLen)
        {
            PyErr_SetString(PyExc_IndexError, "Too many elements for array");
            boost::python::throw_error_already_set();
        }
        result[ii]= convert<typename ArrayType::value_type>( *b);
    }
    if (ii != maxLen)
    {
        PyErr_SetString(PyExc_IndexError, "Not enough elements for array");
        boost::python::throw_error_already_set();
    }
}

template <typename T,std::size_t _Nm>
struct Converter<std::tr1::array<T,_Nm> >
{
    std::tr1::array<T,_Nm> operator()(const boost::python::object &value)
    {
        std::tr1::array<T,_Nm> result;
        convertArray(result,value);
        return result;
    }
};

template <typename T>
struct Converter<std::list<T> >
{
    std::list<T> operator()(const boost::python::object &value)
    {
        using namespace boost::python;
        boost::python::list tempList = extract<boost::python::list>(value);
        ::ssize_t tempLen = len(tempList);
        std::list<T> result;
        for (::ssize_t ii = 0; ii < tempLen; ++ii)
            result.push_back(convert<T>(tempList[ii]));
        return result;
    }
};

template <typename T>
struct Converter<std::vector<T> >
{
    std::vector<T> operator()(const boost::python::object &value)
    {
        using namespace boost::python;
        boost::python::list tempList = extract<boost::python::list>(value);
        ::ssize_t tempLen = len(tempList);
        std::vector<T> result;
        result.reserve(tempLen);
        for (::ssize_t ii = 0; ii < tempLen; ++ii)
            result.push_back(convert<T>(tempList[ii]));
        return result;
    }
};

template <typename T> struct MapConverter
{
    T operator()(boost::python::object const & value)
    {
        using namespace boost::python;
        dict tempDict = extract<dict>(value);
        boost::python::list tempList = extract<boost::python::list>(tempDict.items());
        ::ssize_t tempLen = len(tempList);
        T result;
        for (::ssize_t ii = 0; ii < tempLen; ++ii)
        {
            boost::python::tuple pair = extract<boost::python::tuple>(tempList[ii]);
            BOOST_ASSERT(len(pair) == 2);
            result.insert(typename T::value_type(
                    convert<typename T::key_type>(pair[0]), convert<typename T::mapped_type>(pair[1])));
        }
        return result;
    }
};

template <typename Key, typename Value>
struct Converter<std::map<Key, Value> >
{
    std::map<Key, Value> operator()(boost::python::object const & value)
    {
        return MapConverter<std::map<Key, Value> >()(value);
    }
};

template <typename Key, typename Value>
struct Converter<std::tr1::unordered_map<Key, Value> >
{
    std::tr1::unordered_map<Key, Value> operator()(const boost::python::object &value)
    {
        return MapConverter<std::tr1::unordered_map<Key, Value> >()(value);
    }
};

template <typename Key, typename Value>
struct Converter<boost::unordered_map<Key, Value> >
{
    boost::unordered_map<Key, Value> operator()(const boost::python::object &value)
    {
        return MapConverter<boost::unordered_map<Key, Value> >()(value);
    }
};

template <typename Key, typename Value>
struct Converter<std::unordered_map<Key, Value> >
{
    std::unordered_map<Key, Value> operator()(const boost::python::object &value)
    {
        return MapConverter<std::unordered_map<Key, Value> >()(value);
    }
};

struct FunctionBase
{
protected:
    FunctionBase(const boost::python::object &value) : func(value)
    {
        BOOST_ASSERT(PyCallable_Check(value.ptr()));
    }

    boost::python::object func;
};

template <typename Result>
struct Function0 : FunctionBase
{
    Function0(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()()
    {
        try
        {
            return boost::python::extract<Result>(func());
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }

};

template <typename Result, typename Arg1>
struct Function1 : FunctionBase
{
    Function1(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()(const Arg1 &arg1)
    {
        try
        {
            return boost::python::extract<Result>(func(arg1));
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }
};

template <typename Result, typename Arg1, typename Arg2>
struct Function2 : FunctionBase
{
    Function2(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()(const Arg1 &arg1, const Arg2 &arg2)
    {
        try
        {
            return boost::python::extract<Result>(func(arg1, arg2));
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3>
struct Function3 : FunctionBase
{
    Function3(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3)
    {
        try
        {
            return boost::python::extract<Result>(func(arg1, arg2, arg3));
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
struct Function4 : FunctionBase
{
    Function4(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4)
    {
        try
        {
            return boost::python::extract<Result>(func(arg1, arg2, arg3, arg4));
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
struct Function5 : FunctionBase
{
    Function5(const boost::python::object &value) : FunctionBase(value) {}

    Result operator()(const Arg1 &arg1, const Arg2 &arg2, const Arg3 &arg3, const Arg4 &arg4, const Arg5 &arg5)
    {
        try
        {
            return boost::python::extract<Result>(func(arg1, arg2, arg3, arg4, arg5));
        }
        catch (const boost::python::error_already_set &)
        {
            throw python::Error();
        }
        return *static_cast<Result*>(0);//quiet Eclipse complaints about no return in non-void function
    }
};

template <typename Result>
struct Converter<boost::function<Result()> >
{
    boost::function<Result()> operator()(const boost::python::object &value)
    {
        return Function0<Result>(value);
    }
};

template <typename Result, typename Arg1>
struct Converter<boost::function<Result(Arg1)> >
{
    boost::function<Result(Arg1)> operator()(const boost::python::object &value)
    {
        return Function1<Result, Arg1>(value);
    }
};

template <typename Result, typename Arg1, typename Arg2>
struct Converter<boost::function<Result(Arg1, Arg2)> >
{
    boost::function<Result(Arg1, Arg2)> operator()(const boost::python::object &value)
    {
        return Function2<Result, Arg1, Arg2>(value);
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3>
struct Converter<boost::function<Result(Arg1, Arg2, Arg3)> >
{
    boost::function<Result(Arg1, Arg2, Arg3)> operator()(const boost::python::object &value)
    {
        return Function3<Result, Arg1, Arg2, Arg3>(value);
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
struct Converter<boost::function<Result(Arg1, Arg2, Arg3, Arg4)> >
{
    boost::function<Result(Arg1, Arg2, Arg3, Arg4)> operator()(const boost::python::object &value)
    {
        return Function4<Result, Arg1, Arg2, Arg3, Arg4>(value);
    }
};

template <typename Result, typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename Arg5>
struct Converter<boost::function<Result(Arg1, Arg2, Arg3, Arg4, Arg5)> >
{
    boost::function<Result(Arg1, Arg2, Arg3, Arg4, Arg5)> operator()(const boost::python::object &value)
    {
        return Function5<Result, Arg1, Arg2, Arg3, Arg4, Arg5>(value);
    }
};

template <typename T, bool b>
T convert(const boost::python::object &value, const boost::integral_constant<bool, b> &)
{
    return Converter<T>()(value);
}

template <typename Derived>
Derived convert(const boost::python::object &value, const boost::true_type &)
{
    return Derived::load(value); // Derived is a subclass of PyConfig
}

template <typename T>
T convert(const boost::python::object &value)
{
    return convert<T>(value, boost::is_base_of<PyConfigBase, T>());
}

////

template <typename Derived>
struct FieldBase
{
    virtual void assign(Derived &, const boost::python::object &) = 0;

    // return true if this field was assignable from a string
    virtual bool assign(Derived &, const std::string &) { return false; }

    virtual void print(std::ostream&, Derived const&) const = 0;
};

template <typename Derived, typename T>
struct Field : FieldBase<Derived>
{
    Field(T Derived::*field) : field_(field) {}

    virtual void assign(Derived &instance, const boost::python::object &value)
    {
        instance.*field_ = convert<T>(value);
    }

    virtual void print(std::ostream& out, Derived const& instance) const
    {
        out << "<non-printable>";
    }

protected:
    T Derived::*field_;
};

template <typename Derived, typename T>
struct StringAssignableField : Field<Derived, T>
{
    StringAssignableField(T Derived::*field) : Field<Derived, T>(field) {}

    virtual bool assign(Derived &instance, const std::string &value)
    {
        instance.*Field<Derived, T>::field_ = boost::lexical_cast<T>(value);
        return true;
    }

    virtual void print(std::ostream& out, Derived const& instance) const
    {
        out << instance.*Field<Derived, T>::field_;
    }
};

} // namespace PyConfig_detail

template <class Derived>
class PyConfig : PyConfigBase
{
    // return this proxy from declare() to support setting additional attributes on fields
    friend struct AttributeProxy;
    struct AttributeProxy
    {
        AttributeProxy(const char* fieldName) : m_fieldName(fieldName) {}

        AttributeProxy& mandatory()
        {
            m_requiredFields.push_back(m_fieldName);
            return *this;
        }

        AttributeProxy& tuple(unsigned index)
        {
            bool ok = m_tupleMap.insert(typename TupleMap::value_type(index, m_fieldName)).second;
            BOOST_VERIFY(ok);
            return *this;
        }

    private:
        const char* const m_fieldName;
    };

public:
    static Derived load(const boost::python::object &);
    static Derived load(const char *filename, const char *objname);

    // set the given field in the Derived instance to the given value
    // if you call this after load(), it will override the loaded value, if any
    // you could also call this before load() to implement default values
    // the field's type must be convertible from std::string
    void set(std::string const& fieldName, std::string const&);

    // to be called by ostream << PyConfig
    std::ostream& print(std::ostream&) const;

protected:
    typedef boost::shared_ptr<PyConfig_detail::FieldBase<Derived> > FieldPtr;

    template <typename T>
    static AttributeProxy declare(const char* name, T Derived::*field)
    {
        // we need to decide if the field can be converted from a string in C++
        // we would like to use "boost::has_left_shift<std::ostream&, T const&>",
        // but that was only introduced in Boost 1.48, so we'll improvise for now
        typedef typename boost::mpl::if_<boost::mpl::or_<
            boost::is_arithmetic<T>, // support types like int and float
            boost::is_convertible<T const&, std::string> >,
                PyConfig_detail::StringAssignableField<Derived, T>,
                PyConfig_detail::Field<Derived, T> >::type FieldType;

        FieldPtr fieldPtr = boost::make_shared<FieldType>(field);
        bool ok = m_fields.insert(typename FieldMap::value_type(name, fieldPtr)).second;
        BOOST_VERIFY(ok);
        return AttributeProxy(name);
    }

private:
    typedef std::map<std::string, FieldPtr> FieldMap;
    static FieldMap m_fields;

    typedef std::map<unsigned, std::string> TupleMap;
    static TupleMap m_tupleMap;

    typedef std::vector<std::string> FieldVector;
    static FieldVector m_requiredFields;
};

template<class Derived>
typename PyConfig<Derived>::FieldMap PyConfig<Derived>::m_fields;

template<class Derived>
typename PyConfig<Derived>::TupleMap PyConfig<Derived>::m_tupleMap;

template<class Derived>
typename PyConfig<Derived>::FieldVector PyConfig<Derived>::m_requiredFields;

template <class Derived>
Derived PyConfig<Derived>::load(const boost::python::object &value)
{
    if (m_fields.empty())
        Derived::describe();

    BOOST_ASSERT(!m_fields.empty());

    Derived result;

    using namespace boost::python;

    std::set<FieldVector::value_type> requiredFields(m_requiredFields.begin(), m_requiredFields.end());

    if (PyDict_Check(value.ptr()))
    {
        dict confDict = extract<dict>(value);
        boost::python::list conf = confDict.items();
        ::size_t confSize = len(conf);
        for (::size_t ii = 0; ii < confSize; ++ii)
        {
            boost::python::tuple fieldAndValue = extract<boost::python::tuple>(conf[ii]);
            std::string fieldName = extract<std::string>(fieldAndValue[0]);
            typename FieldMap::iterator it = m_fields.find(fieldName);
            PCCL_UNLESS(it != m_fields.end(), PCCL_THROW("unknown key in dict: " << fieldName));
            it->second->assign(result, fieldAndValue[1]);
            requiredFields.erase(fieldName);
        }
    }
    else if (PyTuple_Check(value.ptr()))
    {
        boost::python::tuple conf = extract<boost::python::tuple>(value);
        ::size_t confSize = len(conf);
        for (::size_t ii = 0; ii < confSize; ++ii)
        {
            typename TupleMap::const_iterator itTuple = m_tupleMap.find(ii);
            PCCL_UNLESS(itTuple != m_tupleMap.end(), PCCL_THROW("invalid tuple index: " << ii));
            std::string const& fieldName = itTuple->second;
            typename FieldMap::iterator it = m_fields.find(fieldName);
            PCCL_UNLESS(it != m_fields.end(), PCCL_THROW("unknown key in dict: " << fieldName));
            it->second->assign(result, conf[ii]);
            requiredFields.erase(fieldName);
        }
    }
    else
    {
        PCCL_THROW("expected a dict or tuple, got: " << value.ptr()->ob_type->tp_name);
    }

    if (!requiredFields.empty())
    {
        std::ostringstream what;
        std::copy(requiredFields.begin(), requiredFields.end(), std::ostream_iterator<std::string>(what, ", "));
        PCCL_THROW("missing required fields: " << what.str());
    }
    return result;
}

template <class Derived>
Derived PyConfig<Derived>::load(const char *filename, const char *objname)
try
{
    boost::python::object mainNamespace = python::runFile(filename);
    return load(mainNamespace[objname]);
}
catch (const boost::python::error_already_set &)
{
    throw python::Error();
}

template <class Derived>
void PyConfig<Derived>::set(std::string const& fieldName, std::string const& value)
{
    typename FieldMap::const_iterator field = m_fields.find(fieldName);
    PCCL_UNLESS(field != m_fields.end(), PCCL_THROW("field not found: " << fieldName));

    FieldPtr const& fieldPtr = field->second;
    bool ok = fieldPtr->assign(static_cast<Derived&>(*this), value);
    PCCL_UNLESS(ok, PCCL_THROW("cannot assign field " << fieldName << " from string"));
}

template <class Derived>
std::ostream& PyConfig<Derived>::print(std::ostream& out) const
{
    Derived const& derived = static_cast<Derived const&>(*this);

    bool first = true;
    BOOST_FOREACH(typename FieldMap::value_type const& field, m_fields)
    {
        if (first)
            first = false;
        else
            out << ", \n";

        out << field.first << ": ";
        field.second->print(out, derived);
    }

    return out;
}

template <class Derived>
std::ostream& operator <<(std::ostream& out, PyConfig<Derived> const& config)
{
    return config.print(out);
}

} // namespace pccl
