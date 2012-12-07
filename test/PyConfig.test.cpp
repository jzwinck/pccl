
#include <list>
#include <map>
#include <vector>

#include <boost/function.hpp>
#include <boost/unordered_map.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/test/parameterized_test.hpp>

#include "../PyConfig.hpp"

struct SubConfig : jz::PyConfig<SubConfig>
{
    std::string whatever;

    static void describe()
    {
        declare("whatever", &SubConfig::whatever);
    }
};

struct MyTuple : jz::PyConfig<MyTuple>
{
    int a, b, c, d;

    MyTuple() : a(), b(), c(), d() {}

    static void describe()
    {
        declare("a", &MyTuple::a).tuple(0);
        declare("b", &MyTuple::b).tuple(1);
        declare("c", &MyTuple::c).tuple(2).mandatory();
        declare("d", &MyTuple::d).tuple(3);
    }
};

struct MyConfig : jz::PyConfig<MyConfig>
{
    std::string mystr;
    int myint;
    std::vector<int> myvec;
    std::map<std::string, int> mymap;
    boost::unordered_map<std::string, int> myunorderedmap;
    std::pair<int, int> mypair;
    std::list<std::pair<int, int> > mypairs;
    std::list<boost::unordered_map<std::string, int> > listOfMaps;
    SubConfig mysub;
    boost::function<std::string()> myfunc;
    boost::function<std::string(int)> myfunc1;
    boost::function<std::string(int x, int y)> myfunc2;
    boost::function<std::string(int x, int y, int z)> myfunc3;
    boost::function<std::string(int a, int b, int c, int d)> myfunc4;
    boost::function<std::string(int a, int b, int c, int d, int e)> myfunc5;
    MyTuple mytuple;

    static void describe()
    {
        declare("mystr",   &MyConfig::mystr);
        declare("myint",   &MyConfig::myint);
        declare("myvec",   &MyConfig::myvec);
        declare("mymap",   &MyConfig::mymap);
        declare("myunorderedmap", &MyConfig::myunorderedmap);
        declare("mypair",  &MyConfig::mypair);
        declare("mypairs", &MyConfig::mypairs);
        declare("listOfMaps", &MyConfig::listOfMaps);
        declare("mysub",   &MyConfig::mysub);
        declare("myfunc",  &MyConfig::myfunc);
        declare("myfunc1", &MyConfig::myfunc1);
        declare("myfunc2", &MyConfig::myfunc2);
        declare("myfunc3", &MyConfig::myfunc3);
        declare("myfunc4", &MyConfig::myfunc4);
        declare("myfunc5", &MyConfig::myfunc5);
        declare("mytuple", &MyConfig::mytuple);
    }
};

struct MyConfigMandatory : jz::PyConfig<MyConfigMandatory>
{
    int myint;

    static void describe()
    {
        declare("myint", &MyConfigMandatory::myint).mandatory();
    }
};


struct MyConfigOptional : jz::PyConfig<MyConfigOptional>
{
    boost::optional<int> myOptionalInt;
    boost::optional<int> myOptionalInt2;
    boost::optional<MyConfigMandatory> myOptionalPyConfig;

    static void describe()
    {
        declare("myOptionalInt",  &MyConfigOptional::myOptionalInt);
        declare("myOptionalInt2", &MyConfigOptional::myOptionalInt2);
        declare("myOptionalPyConfig", &MyConfigOptional::myOptionalPyConfig);
    }
};

void testConf(std::string const& filename)
{
    MyConfig conf = MyConfig::load(filename.c_str(), "conf");

    BOOST_CHECK_EQUAL(conf.myint, 42);

    // test use of overrides using set()
    BOOST_CHECK_EQUAL(conf.mystr, "");
    conf.set("mystr", "newstr");
    BOOST_CHECK_EQUAL(conf.mystr, "newstr");
    conf.set("myint", "51");
    BOOST_CHECK_EQUAL(conf.myint, 51);
    BOOST_CHECK_THROW(conf.set("myint", "foo"), std::bad_cast);

    BOOST_REQUIRE_EQUAL(conf.myvec.size(), 2);
    BOOST_CHECK_EQUAL(conf.myvec.at(0), 1);
    BOOST_CHECK_EQUAL(conf.myvec.at(1), 7);
    BOOST_CHECK_THROW(conf.set("myvec", "foo"), std::runtime_error);

    BOOST_REQUIRE_EQUAL(conf.mymap.size(), 2);
    BOOST_CHECK_EQUAL(conf.mymap.find("a")->second, 1);
    BOOST_CHECK_EQUAL(conf.mymap.find("b")->second, 2);

    BOOST_REQUIRE_EQUAL(conf.myunorderedmap.size(), 3);
    BOOST_CHECK_EQUAL(conf.myunorderedmap.find("c")->second, 3);
    BOOST_CHECK_EQUAL(conf.myunorderedmap.find("d")->second, 4);
    BOOST_CHECK_EQUAL(conf.myunorderedmap.find("e")->second, 5);

    BOOST_CHECK_EQUAL(conf.mypair.first, 6);
    BOOST_CHECK_EQUAL(conf.mypair.second, 7);

    BOOST_REQUIRE_EQUAL(conf.mypairs.size(), 2);
    BOOST_CHECK_EQUAL(conf.mypairs.front().first, 1);
    BOOST_CHECK_EQUAL(conf.mypairs.front().second, 2);
    BOOST_CHECK_EQUAL(conf.mypairs.back().first, 3);
    BOOST_CHECK_EQUAL(conf.mypairs.back().second, 4);

    BOOST_REQUIRE_EQUAL(conf.listOfMaps.size(), 2);
    BOOST_CHECK_EQUAL(conf.listOfMaps.front().find("a")->second, 1);
    BOOST_CHECK_EQUAL(conf.listOfMaps.front().find("b")->second, 2);
    BOOST_CHECK_EQUAL(conf.listOfMaps.back().find("c")->second, 3);
    BOOST_CHECK_EQUAL(conf.listOfMaps.back().find("d")->second, 4);

    BOOST_CHECK_EQUAL(conf.mysub.whatever, "ok");

    BOOST_REQUIRE(conf.myfunc);
    BOOST_CHECK_EQUAL(conf.myfunc(), "funky");

    BOOST_REQUIRE(conf.myfunc1);
    BOOST_CHECK_EQUAL(conf.myfunc1(42), "funky 42");

    BOOST_REQUIRE(conf.myfunc2);
    BOOST_CHECK_EQUAL(conf.myfunc2(42, 7), "mystandalonefun got args 42  7");

    BOOST_CHECK_EQUAL(conf.mytuple.a, 1);
    BOOST_CHECK_EQUAL(conf.mytuple.b, 2);
    BOOST_CHECK_EQUAL(conf.mytuple.c, 3);
    BOOST_CHECK_EQUAL(conf.mytuple.d, 0);

    BOOST_CHECK_THROW(MyConfigMandatory::load(filename.c_str(), "conf2"), std::runtime_error);

    MyConfigOptional conf3 = MyConfigOptional::load(filename.c_str(), "conf3");
    BOOST_CHECK_EQUAL(conf3.myOptionalInt.is_initialized(), false);
    BOOST_CHECK_EQUAL(conf3.myOptionalInt2.is_initialized(), true);
    BOOST_CHECK_EQUAL(conf3.myOptionalInt2.get(), 42);
    BOOST_CHECK_EQUAL(conf3.myOptionalPyConfig.is_initialized(), false);

    MyConfigOptional conf4 = MyConfigOptional::load(filename.c_str(), "conf4");
    BOOST_CHECK_EQUAL(conf4.myOptionalInt.is_initialized(), false);
    BOOST_CHECK_EQUAL(conf4.myOptionalInt2.is_initialized(), true);
    BOOST_CHECK_EQUAL(conf4.myOptionalInt2.get(), 42);
    BOOST_CHECK_EQUAL(conf4.myOptionalPyConfig.is_initialized(), true);
    BOOST_CHECK_EQUAL(conf4.myOptionalPyConfig->myint, 43);
}

struct MyArgConfig : jz::PyConfig<MyArgConfig>
{
    std::vector<std::string> args;

    static void describe()
    {
        declare("args", &MyArgConfig::args);
    }
};

void testStreamOutput(std::string const& filename)
{
    MyConfig conf = MyConfig::load(filename.c_str(), "conf");
    std::cout << conf << '\n';
}

boost::unit_test_framework::test_suite *init_unit_test_suite(int argc, char *argv[])
{
    JZ_UNLESS(argc > 1, JZ_THROW("config file(s) must be specified"));

    boost::unit_test::framework::master_test_suite().add(
        BOOST_PARAM_TEST_CASE(testConf, argv + 1, argv + argc));

    boost::unit_test::framework::master_test_suite().add(
        BOOST_PARAM_TEST_CASE(testStreamOutput, argv + 1, argv + argc));

    return NULL;
}
