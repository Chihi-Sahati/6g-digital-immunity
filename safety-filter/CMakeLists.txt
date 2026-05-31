# ============================================================================
# CMakeLists.txt — Deterministic Digital Immunity for 6G Networks
# Phase 2: DSF Core — CBF Safety Filter CMake Build Configuration
# ============================================================================
#
# Modern CMake (3.21+) build system for the CBF mathematical core.
# Produces:
#   1. libcbf_core.a      — Static library with CBF/Lyapunov evaluation
#   2. cbf_python.so       — Python binding (optional, requires pybind11)
#
# Dependencies:
#   - Eigen3 (REQUIRED)    — Linear algebra for barrier function evaluation
#   - pybind11 (OPTIONAL) — Python bindings via pybind11
#
# Build instructions:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_BINDINGS=ON
#   cmake --build . --parallel
#   cmake --install . --prefix /usr/local
# ============================================================================

cmake_minimum_required(VERSION 3.21)

# ============================================================================
# Project Declaration
# ============================================================================

project(
    cbf_safety_filter
    VERSION 1.0.0
    DESCRIPTION "Control Barrier Function Safety Filter for 6G Digital Immunity"
    LANGUAGES CXX
)

# ============================================================================
# C++ Standard & Compiler Flags
# ============================================================================

# Enforce C++20 for concepts, ranges, coroutines, and std::format.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Strict warning policy — no warnings allowed in a safety-critical library.
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Werror=return-type          # All code paths must return a value
    -Werror=uninitialized        # No use of uninitialised variables
    -Werror=sign-compare         # No signed/unsigned comparison surprises
    -Werror=shadow               # No variable shadowing
    -Werror=format=2             # Strict printf-style format checks
    -Werror=missing-include-dirs # All include directories must exist
)

# Release optimisations with debug symbols for profiling.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-O2 -march=native)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-g3 -O0 -fno-omit-frame-pointer -fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# ============================================================================
# Dependencies
# ============================================================================

# --- Eigen3 (REQUIRED) ---
# Eigen3 is a header-only linear algebra library. It provides the
# Eigen::VectorXd, Eigen::RowVectorXd, and matrix operations used by
# the CBF engine for barrier function evaluation and Lie derivative
# computation.
#
# FindEigen3.cmake is included with recent CMake versions (>= 3.10).
# Fallback: set EIGEN3_INCLUDE_DIR if not found automatically.
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Log the found Eigen3 path for diagnostics.
message(STATUS "Eigen3 found: ${EIGEN3_INCLUDE_DIR}")

# --- pybind11 (OPTIONAL) ---
# pybind11 enables seamless exposure of the CBF engine to Python via
# a native C++ Python extension module.
#
# The binding is optional because the primary Python integration uses
# ctypes (see cbf_engine.py) for simpler deployment without pybind11.
#
# Lookup order:
#   1. find_package(pybind11 CONFIG) — system-installed pybind11
#   2. find_package(pybind11)         — pybind11's own FindPybind11.cmake
#   3. add_subdirectory(fetched)      — FetchContent fallback
option(BUILD_PYTHON_BINDINGS "Build Python bindings via pybind11" OFF)

if(BUILD_PYTHON_BINDINGS)
    include(FetchContent)

    # Try system pybind11 first, then fall back to FetchContent.
    find_package(pybind11 QUIET CONFIG)

    if(NOT pybind11_FOUND)
        find_package(pybind11 QUIET)

        if(NOT pybind11_FOUND)
            message(STATUS "pybind11 not found on system; fetching via FetchContent...")
            FetchContent_Declare(
                pybind11
                GIT_REPOSITORY https://github.com/pybind/pybind11.git
                GIT_TAG        v2.13.6
                GIT_SHALLOW    TRUE
            )
            FetchContent_MakeAvailable(pybind11)
        endif()
    endif()

    if(pybind11_FOUND OR TARGET pybind11::module)
        set(PYBIND11_AVAILABLE ON)
        message(STATUS "pybind11 available — Python bindings will be built")
    else()
        set(PYBIND11_AVAILABLE OFF)
        message(WARNING "pybind11 could not be found — skipping Python bindings")
        set(BUILD_PYTHON_BINDINGS OFF CACHE BOOL "Build Python bindings via pybind11" FORCE)
    endif()
endif()

# ============================================================================
# Library Target: cbf_core (STATIC)
# ============================================================================

# Core CBF safety filter library — pure C++, no external dependencies
# beyond Eigen3. This library is intended to be linked into the DSF
# daemon, the Python ctypes wrapper, and any test harnesses.

add_library(cbf_core STATIC
    core/cbf_math.cpp
)

# Target-specific include directories.
target_include_directories(cbf_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/core>
        $<INSTALL_INTERFACE:include/cbf>
)

# Link Eigen3 headers.
target_link_libraries(cbf_core
    PUBLIC
        Eigen3::Eigen
)

# Compiler definition for library version.
target_compile_definitions(cbf_core
    PUBLIC
        CBF_SAFETY_FILTER_VERSION="${PROJECT_VERSION}"
)

# ============================================================================
# Library Target: cbf_python (pybind11 MODULE, OPTIONAL)
# ============================================================================

if(BUILD_PYTHON_BINDINGS AND PYBIND11_AVAILABLE)
    add_library(cbf_python MODULE
        core/cbf_engine.cpp
    )

    # Link against the core library and pybind11.
    target_link_libraries(cbf_python
        PRIVATE
            cbf_core
            pybind11::module
    )

    # pybind11 sets the extension suffix automatically, but ensure
    # no "lib" prefix on all platforms (Python convention).
    set_target_properties(cbf_python PROPERTIES
        PREFIX ""
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

    # Position-independent code required for shared library.
    set_target_properties(cbf_python PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )

    message(STATUS "Python binding target 'cbf_python' configured")
endif()

# ============================================================================
# Installation Rules
# ============================================================================

include(GNUInstallDirs)

# Install the static library.
install(
    TARGETS cbf_core
    EXPORT cbf_safety_filterTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install the Python module (if built).
if(BUILD_PYTHON_BINDINGS AND PYBIND11_AVAILABLE)
    install(
        TARGETS cbf_python
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Install public headers.
install(
    FILES
        core/cbf_math.hpp
    DESTINATION
        ${CMAKE_INSTALL_INCLUDEDIR}/cbf
)

# Export CMake package for downstream consumers.
install(
    EXPORT cbf_safety_filterTargets
    NAMESPACE cbf::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cbf_safety_filter
)

# Generate and install CMake package config for find_package(cbf_safety_filter).
# include(CMakePackageConfigHelpers)
#
# configure_package_config_file(
#     ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cbf_safety_filterConfig.cmake.in
#     ${CMAKE_CURRENT_BINARY_DIR}/cbf_safety_filterConfig.cmake
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cbf_safety_filter
# )
#
# write_basic_package_version_file(
#     ${CMAKE_CURRENT_BINARY_DIR}/cbf_safety_filterConfigVersion.cmake
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )
#
# install(
#     FILES
#         ${CMAKE_CURRENT_BINARY_DIR}/cbf_safety_filterConfig.cmake
#         ${CMAKE_CURRENT_BINARY_DIR}/cbf_safety_filterConfigVersion.cmake
#     DESTINATION
#         ${CMAKE_INSTALL_LIBDIR}/cmake/cbf_safety_filter
# )

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "=== CBF Safety Filter Build Summary ===")
message(STATUS "  Version:          ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen3:           ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  Python Bindings:  ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Install Prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
