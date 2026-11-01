cmake_minimum_required(VERSION 4.3)
project(SLIC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_library(slic_lib
        slic_sequential.cpp
        SLIC_parallel_AOS.cpp
        SLIC_parallel_SOA.cpp)

target_link_libraries(slic_lib PUBLIC ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
target_compile_definitions(slic_lib PUBLIC PROJECT_DIR="${CMAKE_SOURCE_DIR}")
target_compile_options(slic_lib PUBLIC -O3 -march=native)

add_executable(SLIC_scaling main_scaling.cpp)
target_link_libraries(SLIC_scaling PRIVATE slic_lib)

add_executable(SLIC_soa main_soa.cpp)
target_link_libraries(SLIC_soa PRIVATE slic_lib)

option(USE_ASAN "AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(USE_TSAN "ThreadSanitizer" OFF)

if (USE_ASAN AND USE_TSAN)
    message(FATAL_ERROR "ASan e TSan non si possono usare insieme")
endif()

if (USE_ASAN)
    target_compile_options(slic_lib PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(slic_lib PUBLIC -fsanitize=address,undefined)
endif()

if (USE_TSAN)
    target_compile_options(slic_lib PUBLIC -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(slic_lib PUBLIC -fsanitize=thread)
endif()