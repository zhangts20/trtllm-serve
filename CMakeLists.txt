cmake_minimum_required(VERSION 3.12)

project(my_test)

include("${CMAKE_SOURCE_DIR}/vendor/rapidjson.cmake")

include_directories(${RAPIDJSON_INCLUDE_DIR})

add_executable(demo main.cpp)
