cmake_minimum_required(VERSION 3.16)

project(custom_factor LANGUAGES CXX)

#Important CMAKE stuff
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "-march=native -O3 -fno-math-errno -funroll-loops -g1")

find_package(spacetime REQUIRED)

add_executable(estimator src/main.cpp)
target_link_libraries(estimator PRIVATE spacetime::spacetime)
target_sources(estimator PRIVATE src/customMeasurementFactor.hpp)

