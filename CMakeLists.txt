cmake_minimum_required(VERSION 3.16)

project(spacetime VERSION 1.0.0 LANGUAGES C CXX)

#Important CMAKE stuff
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 17)

# Add GNUInstallDirs for standard installation directories
include(GNUInstallDirs)

### Build Options
option(Profile "Profile" OFF)
option(Debug "Debug" OFF)
option(Estimator "Estimator" ON)
# option(BUILD_TESTS "BUILD_TESTS" ON)

message(STATUS "USE_AUTODIFF = ${USE_AUTODIFF}")

if(Debug)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Og -g3")
else()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "-march=native -O3 -g0")
endif()

if(Profile)
    string(APPEND CMAKE_CXX_FLAGS " -pg")
endif()



### Dependency Finds

#Qt
find_package(Qt6 REQUIRED COMPONENTS Core Gui Xml Network Widgets OpenGLWidgets)
qt_standard_project_setup()
#MPI
find_package(MPI COMPONENTS CXX REQUIRED)
set(MPI_C_COMPILER /usr/bin/mpicc)
set(MPI_CXX_COMPILER /usr/bin/mpicxx)
#Eigen
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Lgmath 
find_package(lgmath REQUIRED)

# VTK 
find_package(VTK 9.3 REQUIRED)

# Json
find_package(jsoncpp REQUIRED)

# tsl-robin-map
find_package(tsl-robin-map REQUIRED)


### Project Library build and export 
file(GLOB_RECURSE CPP_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

# Include header files so MOC can process Q_OBJECT macros
file(GLOB_RECURSE H_FILES
    ${PROJECT_SOURCE_DIR}/include/*.hpp
)

# Add Qt MOC processing for headers with Q_OBJECT
set(CMAKE_AUTOMOC ON)
qt_add_library(${PROJECT_NAME} STATIC ${CPP_FILES} ${H_FILES})


target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
    ${lgmath_INCLUDE_DIRS}/include
    ${VTK_INCLUDE_DIRS}
)

if(USE_AUTODIFF)
    message(STATUS "Using autodiff")
    find_package(autodiff)
    target_compile_definitions(${PROJECT_NAME} PUBLIC USE_AUTODIFF=1)
    target_link_libraries(${PROJECT_NAME} PUBLIC autodiff::autodiff)
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found, linking OpenMP")
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
    Qt6::Core
    Qt6::Gui
    Qt6::Xml
    Qt6::Network
    Qt6::Widgets
    Qt6::OpenGLWidgets
    Eigen3::Eigen
    JsonCpp::JsonCpp
    lgmath
    tsl::robin_map
    ${VTK_LIBRARIES}
)

# Add an alias target for better namespacing
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Add an alias target for better namespacing
message(STATUS "CMAKE_INSTALL_LIBDIR: " ${CMAKE_INSTALL_LIBDIR})
message(STATUS "CMAKE_INSTALL_BINDIR: " ${CMAKE_INSTALL_BINDIR})
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: " ${CMAKE_INSTALL_INCLUDEDIR})
message(STATUS "Installation Configuration for ${PROJECT_NAME}:")
message(STATUS "  Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Header destination: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "  Library destination: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
message(STATUS "  Binary destination: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}")
message(STATUS "  CMake config destination: ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

# Install header files
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Installation rules
install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the targets file with proper namespace
install(EXPORT ${PROJECT_NAME}Targets
    FILE ${PROJECT_NAME}Targets.cmake
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

set(PROJECT_LIBRARY ${PROJECT_NAME})
include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

# if(BUILD_TESTS)
#     message(STATUS "Building tests")
#     include(FetchContent)
#     FetchContent_Declare(
#         googletest
#         URL https://github.com/google/googletest/archive/e88cb95b92acbdce9b058dd894a68e1281b38495.zip
#     )
#     # For Windows: Prevent overriding the parent project's compiler/linker settings
#     set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#     FetchContent_MakeAvailable(googletest)

#     enable_testing()
#     include(GoogleTest)

#     add_executable(test_factors tests/FactorTests.cpp)
#     target_link_libraries(test_factors ${PROJECT_NAME} GTest::gtest_main)
#     gtest_discover_tests(test_factors)

#     add_executable(test_functions tests/FunctionTests.cpp)
#     target_link_libraries(test_functions ${PROJECT_NAME} GTest::gtest_main)
#     gtest_discover_tests(test_functions)

#     add_executable(test_jacobians tests/AutodiffJacTests.cpp)
#     target_link_libraries(test_jacobians ${PROJECT_NAME} GTest::gtest_main)
#     gtest_discover_tests(test_jacobians)
# endif()


if(Estimator)
    file(GLOB_RECURSE H_FILES_TOOLS
        ${PROJECT_SOURCE_DIR}/scripts/*.hpp
    )
    file(GLOB_RECURSE CPP_FILES_TOOLS
        ${PROJECT_SOURCE_DIR}/scripts/*.cpp
    )

    qt_add_executable(estimator scripts/main.cpp)
    target_link_libraries(estimator PRIVATE ${PROJECT_NAME})
    target_sources(estimator PRIVATE ${H_FILES_TOOLS} ${CPP_FILES_TOOLS})

    qt_add_executable(BilevelOptimizer src/BilevelOptimizer.cpp)
    target_link_libraries(BilevelOptimizer PRIVATE ${PROJECT_NAME})
endif()
