cmake_minimum_required(VERSION 3.10)
PROJECT (HelloSLAM)
# include head file
include_directories(include)
# used to build lib
add_library(libHello src/hello.cpp)
# build executable file
add_executable(sayHello src/useHello.cpp)
# link lib to exectable file
target_link_libraries(sayHello libHello)

# 6 8 10 可以合并为这一句
# add_executable(main src/useHello.cpp src/hello.cpp)