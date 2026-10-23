cmake_minimum_required(VERSION 4.3)
project(SLIC)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 20)

add_executable(SLIC slic_sequential.cpp)
target_link_libraries(SLIC ${OpenCV_LIBS})
