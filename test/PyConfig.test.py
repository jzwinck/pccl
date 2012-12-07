#!/usr/bin/env python

import sys

def mystandalonefun(x, y):
    return "mystandalonefun got args " + str(x) + "  " + str(y)

def myfun3(x, y, z):
    return "myfun3 got args " + str(x) + "  " + str(y) + "  " + str(z)

def myfun4(a, b, c, d):
    return "myfun4 got args " + str(a) + "  " + str(b) + "  " + str(c) + "  " + str(d)

def myfun5(a, b, c, d, e):
    return "myfun4 got args " + str(a) + "  " + str(b) + "  " + str(c) + "  " + str(d) + "  " + str(e)


conf = {
    'myint' : 42,
    'myvec' : [ 1, 7 ],
    'mymap' : { 'a' : 1, 'b' : 2 },
    'myunorderedmap' : { 'c' : 3, 'd' : 4, 'e' : 5 },
    'mypair' : (6, 7),
    'mypairs' : [ (1, 2), (3, 4) ],
    'listOfMaps' : [ { 'a' : 1, 'b' : 2 }, { 'c' : 3, 'd' : 4 } ],
    'mysub'   : { 'whatever' : 'ok' },
    'myfunc' : lambda: 'funky',
    'myfunc1' : lambda x: 'funky ' + str(x),
    'myfunc2' : lambda x, y: mystandalonefun(x, y),
    'myfunc3' : lambda x, y, z: myfun3(x, y, z),
    'myfunc4' : lambda a, b, c, d: myfun4(a, b, c, d),
    'myfunc5' : lambda a, b, c, d, e: myfun5(a, b, c, d, e),
    'mytuple' : (1, 2, 3),
}

conf2 = {
}

conf3 = {
    'myOptionalInt2': 42
}

conf4 = {
    'myOptionalInt2': 42,
    'myOptionalPyConfig': {"myint": 43 }
}
