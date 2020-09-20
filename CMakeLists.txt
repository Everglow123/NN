cmake_minimum_required(VERSION 3.0.0)
project(NN VERSION 0.1.0)
# find_package(OpenMP REQUIRED) if(OPENMP_FOUND) message("openmp found")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -v  -o3")
#
# endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -v  -O3 -Wall")
set(CMAKE_CXX_STANDARD 17)
include(CTest)
include_directories(${PROJECT_SOURCE_DIR}/include/ "/usr/include/eigen3")
enable_testing()

add_executable(NN main.cpp utils.cpp neural_network.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
