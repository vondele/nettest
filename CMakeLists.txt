cmake_minimum_required(VERSION 3.12)
project(hello)

message(STATUS "Test message")
add_executable(hello hello.cpp)
install(TARGETS hello)
