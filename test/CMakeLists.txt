
find_package(Boost REQUIRED unit_test_framework)

add_executable(NumPyArray NumPyArray.test.cpp)
target_link_libraries(NumPyArray pccl ${Boost_LIBRARIES})

add_executable(PyConfig PyConfig.test.cpp)
target_link_libraries(PyConfig pccl ${Boost_LIBRARIES})

add_executable(InterpreterLockGuard InterpreterLockGuard.test.cpp)
target_link_libraries(InterpreterLockGuard pccl ${Boost_LIBRARIES})

add_executable(Error Error.test.cpp)
target_link_libraries(Error pccl ${Boost_LIBRARIES})

add_executable(throw throw.test.cpp)
target_link_libraries(throw pccl ${Boost_LIBRARIES})
