cmake_minimum_required(VERSION 4.3)
project(SLIC)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(SLIC slic_sequential.cpp
        main.cpp
        slic.h
        SLIC_parallel_AOS.cpp
)
target_compile_definitions(SLIC PRIVATE PROJECT_DIR="${CMAKE_SOURCE_DIR}")
target_link_libraries(SLIC PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

option(USE_ASAN "AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(USE_TSAN "ThreadSanitizer (data race)" OFF)

if (USE_ASAN AND USE_TSAN)
    message(FATAL_ERROR "ASan e TSan can't be used at the same time")
endif()

if (USE_ASAN)
    target_compile_options(SLIC PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
    target_link_options(SLIC PRIVATE -fsanitize=address,undefined)
endif()

if (USE_TSAN)
    target_compile_options(SLIC PRIVATE -fsanitize=thread -fno-omit-frame-pointer -g)
    target_link_options(SLIC PRIVATE -fsanitize=thread)
endif()