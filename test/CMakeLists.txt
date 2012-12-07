
find_package(Boost REQUIRED unit_test_framework)

add_executable(NumPyArray NumPyArray.test.cpp)
target_link_libraries(NumPyArray jzpy ${Boost_LIBRARIES})

add_executable(PyConfig PyConfig.test.cpp)
target_link_libraries(PyConfig jzpy ${Boost_LIBRARIES})

add_executable(InterpreterLockGuard InterpreterLockGuard.test.cpp)
target_link_libraries(InterpreterLockGuard jzpy ${Boost_LIBRARIES})

add_executable(Error Error.test.cpp)
target_link_libraries(Error jzpy ${Boost_LIBRARIES})

add_executable(throw throw.test.cpp)
target_link_libraries(throw jzpy ${Boost_LIBRARIES})
