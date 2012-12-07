cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wformat=2 -Winit-self -Wno-unused-parameter -std=c++0x")

# if we're on Linux only:
# set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-no-undefined")

set(CMAKE_MODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}) # for FindNumPy.cmake

find_package(Boost REQUIRED python thread)
set(PYTHON_INCLUDE_DIRS "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7")
set(PYTHON_LIBRARIES "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
#find_package(PythonLibs REQUIRED)
find_package(NumPy)
#set(NUMPY_INCLUDE_DIRS "/System/Library/Frameworks/Python.framework/Versions/Current/Extras/lib/python/numpy/core/include") # or use FindNumPy.cmake

include_directories(${PYTHON_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIRS})

# for debugging
#message(${PYTHON_INCLUDE_DIRS})
#message(${PYTHON_LIBRARIES})
#message(${Boost_LIBRARIES})
#message(${NUMPY_INCLUDE_DIRS})

add_library(jzpy
    Error.cpp
    InterpreterLockGuard.cpp
    NumPy.cpp
    run.cpp
    NumPyArray.cpp
    NumPyDataType.cpp
    terminate.cpp
    )

target_link_libraries(jzpy ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_subdirectory(test)
