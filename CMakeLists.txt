cmake_minimum_required(VERSION 2.4)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wformat=2 -Winit-self -Wno-unused-parameter -std=c++0x")

find_package(Boost REQUIRED python thread)

if(APPLE)
    cmake_minimum_required(VERSION 2.8) # FindNumPy.cmake has errors in CMake 2.6
    set(PYTHON_INCLUDE_PATH "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7")
    set(PYTHON_LIBRARIES "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib")
    set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}) # for FindNumPy.cmake
    find_package(NumPy)
    # or, without FindNumPy.cmake: set(NUMPY_INCLUDE_DIRS "/System/Library/Frameworks/Python.framework/Versions/Current/Extras/lib/python/numpy/core/include")
    include_directories(${NUMPY_INCLUDE_DIRS})
else() # assume Linux or similar
    find_package(PythonLibs REQUIRED)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-no-undefined")
endif()

include_directories(${PYTHON_INCLUDE_PATH})

# for debugging
#message(${PYTHON_INCLUDE_PATH})
#message(${PYTHON_LIBRARIES})
#message(${Boost_LIBRARIES})
#message(${NUMPY_INCLUDE_DIRS})

add_library(pccl
    Error.cpp
    InterpreterLockGuard.cpp
    NumPy.cpp
    run.cpp
    NumPyArray.cpp
    NumPyDataType.cpp
    terminate.cpp
    )

target_link_libraries(pccl ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})

add_subdirectory(test)
