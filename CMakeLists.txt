################################################################################
# General Informations
################################################################################

cmake_minimum_required(VERSION 3.1)
project(ElasticRods)

# CMP0063: Honor visibility properties for all target types.
if(POLICY CMP0063)
    cmake_policy(SET CMP0063 NEW)
endif()

################################################################################

set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELWITHASSERT        "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_C_FLAGS_RELWITHASSERT          "-O3" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT    "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT "" CACHE STRING "Flags used during RelWithAssertions builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELWITHASSERT
    CMAKE_C_FLAGS_RELWITHASSERT
    CMAKE_EXE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_SHARED_LINKER_FLAGS_RELWITHASSERT
    CMAKE_MODULE_LINKER_FLAGS_RELWITHASSERT
    CMAKE_STATIC_LINKER_FLAGS_RELWITHASSERT
    )

set(CMAKE_CXX_FLAGS_RELEASENATIVE        "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_C_FLAGS_RELEASENATIVE          "-O3 -march=native -DNDEBUG" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE    "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE "" CACHE STRING "Flags used during native release builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_RELEASENATIVE
    CMAKE_C_FLAGS_RELEASENATIVE
    CMAKE_EXE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_SHARED_LINKER_FLAGS_RELEASENATIVE
    CMAKE_MODULE_LINKER_FLAGS_RELEASENATIVE
    CMAKE_STATIC_LINKER_FLAGS_RELEASENATIVE
    )

set(CMAKE_CXX_FLAGS_DEBUGNATIVE        "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_C_FLAGS_DEBUGNATIVE          "-g -march=native" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE    "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
set(CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE "" CACHE STRING "Flags used during native debug builds" FORCE)
mark_as_advanced(
    CMAKE_CXX_FLAGS_DEBUGNATIVE
    CMAKE_C_FLAGS_DEBUGNATIVE
    CMAKE_EXE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_SHARED_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_MODULE_LINKER_FLAGS_DEBUGNATIVE
    CMAKE_STATIC_LINKER_FLAGS_DEBUGNATIVE
    )


if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type selected, default to RelWithAssert")
    set(CMAKE_BUILD_TYPE "RelWithAssert")
endif()

set(CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "Choose the type of build. Options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert" FORCE)

# Color output
include(UseColors)

# Make sure warnings/errors are still colorized when using Ninja for building.
if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    add_definitions(-fdiagnostics-color=always)
endif()

# Export compile flags(used for autocompletion of the C++ code)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# CMake plugin for vscode
include(CMakeToolsHelpers OPTIONAL)

# Enable more warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -Wno-comment")

# We need -fPIC when compiling our libraries and our dependencies for
# the python bindings to link.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

################################################################################
# Common libraries
################################################################################
if(NOT TARGET MeshFEM)
    option(MESHFEM_ENABLE_BENCHMARKING "" ON)
    add_subdirectory(${THIRD_PARTY_DIR}/MeshFEM)
endif()

if(NOT TARGET rotation_optimization)
    add_subdirectory(${THIRD_PARTY_DIR}/RotationOptimization)
endif()

add_subdirectory(${THIRD_PARTY_DIR}/visvalingam_simplify)

find_package(Spectra REQUIRED)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use static library" ON)

find_package(LIBIGL QUIET)
################################################################################
# Elastic Rod Library
################################################################################
add_library(ElasticRods ElasticRod.cc ElasticRodOutput.cc RodMaterial.cc CrossSection.cc eigensolver.cc)
target_include_directories(ElasticRods SYSTEM PUBLIC ${SPECTRA_INCLUDE_DIR})
target_link_libraries(ElasticRods MeshFEM VisvalingamSimplify)
set_target_properties(ElasticRods PROPERTIES CXX_STANDARD 14)
set_target_properties(ElasticRods PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(ElasticRods SYSTEM PUBLIC .)

add_library(RodLinkages RodLinkage.cc linkage_deformation_analysis.cc LinkageOptimization.cc TargetSurfaceFitter.cc infer_target_surface.cc)
target_link_libraries(RodLinkages ElasticRods rotation_optimization igl::core)
set_target_properties(RodLinkages PROPERTIES CXX_STANDARD 14)
set_target_properties(RodLinkages PROPERTIES CXX_STANDARD_REQUIRED ON)
target_include_directories(RodLinkages SYSTEM PUBLIC .)

################################################################################
# Third Party Libraries
################################################################################
# Knitro
find_package(Knitro QUIET)
if(KNITRO_FOUND)
    target_include_directories(ElasticRods SYSTEM PUBLIC ${KNITRO_INCLUDE_DIRS})
    target_link_libraries(ElasticRods ${KNITRO_LIBRARIES})
    target_compile_definitions(ElasticRods PUBLIC -DHAS_KNITRO)

    target_include_directories(RodLinkages SYSTEM PUBLIC ${KNITRO_INCLUDE_DIRS})
    target_link_libraries(RodLinkages ${KNITRO_LIBRARIES})
    target_compile_definitions(RodLinkages PUBLIC -DHAS_KNITRO)
else()
    message(STATUS "Knitro not found; knitro solvers and dependent binaries disabled")
endif()

################################################################################
# Python Bindings
################################################################################
add_subdirectory(python_bindings)

################################################################################
# Grasshopper Bindings
################################################################################
add_subdirectory(grasshopper)

################################################################################
# More binaries
################################################################################
add_subdirectory(tests)
add_subdirectory(tools)
