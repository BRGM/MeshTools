cmake_minimum_required(VERSION 3.12)
cmake_policy(VERSION 3.12)

project(MeshTools CXX)

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif()
set(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS true)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(
    FATAL_ERROR
      "-------------------------------
In-source compilation forbidden.
Build a directory build: mkdir build.
Then: cd build; cmake ..
--------------------------------"
  )
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE
      "Release"
      CACHE
        STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE
  )
endif(NOT CMAKE_BUILD_TYPE)

# compiled python modules will be put in
# ${CMAKE_INSTALL_PREFIX}/${MESHTOOLS_PYTHON_PACKAGE_NAME}
set(MESHTOOLS_PYTHON_PACKAGE_NAME MeshTools)

# will not override a user-provided installation prefix (e.g. using cmake
# -DCMAKE_INSTALL_PREFIX=...)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  if(MESHTOOLS_PYTHON_PACKAGE_DIRECTORY)
    set(CMAKE_INSTALL_PREFIX
        ${MESHTOOLS_PYTHON_PACKAGE_DIRECTORY}
        CACHE
          PATH
          "Default installation prefix, will be through MESHTOOLS_PYTHON_PACKAGE_DIRECTORY variable only if no installation prefix is given."
          FORCE
    )
  else()
    # if no specific instalation prefix was provided and if the
    # MESHTOOLS_PYTHON_PACKAGE_DIRECTORY variable is not set default to the
    # source directory cmake export will be put there and python modules will be
    # put in ${CMAKE_INSTALL_PREFIX}/${MESHTOOLS_PYTHON_PACKAGE_NAME}
    set(CMAKE_INSTALL_PREFIX
        ${CMAKE_CURRENT_SOURCE_DIR}
        CACHE
          PATH
          "Default installation prefix, will be through MESHTOOLS_PYTHON_PACKAGE_DIRECTORY variable only if no installation prefix is given."
          FORCE
    )
  endif()
endif()

# Check if meshtools is being used directly or via add_subdirectory
set(MESHTOOLS_MASTER_PROJECT OFF)
set(MESHTOOLS_ROOT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(MESHTOOLS_MASTER_PROJECT ON)
endif()

set(PYBIND11_PYTHON_VERSION
    3
    CACHE STRING "Python version to use for compiling modules"
)

# -- Dependencies --

set(MESHTOOLS_THIRDPARTIES_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/thirdparties)

find_package(pybind11 QUIET) # try locally
if(NOT pybind11_FOUND) # else fetch it from github
  message(STATUS "Could NOT find pybind11 (fetching it from GitHub)")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.4.3
  )
  if(${CMAKE_VERSION} VERSION_LESS 3.14)
    FetchContent_GetProperties(pybind11)
    if(NOT pybind11_POPULATED)
      FetchContent_Populate(pybind11)
      add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
    endif()
  else()
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

# The following test is not mandatory as python interpreter comes as a
# by-product of requiring pybind11
if(NOT ${PYTHONINTERP_FOUND})
  message(FATAL_ERROR "python interpreter is mandatory, CMake will stop.")
endif(NOT ${PYTHONINTERP_FOUND})
# The following test is not mandatory as python interpreter comes as a
# by-product of requiring pybind11
if(NOT ${PYTHONLIBS_FOUND})
  message(FATAL_ERROR "python libraries are mandatory, CMake will stop.")
endif(NOT ${PYTHONLIBS_FOUND})

# CGAL and its components
option(MESHTOOLS_TRIES_TO_USE_CGAL
       "look for CGAL and tries to compile specific bindings" ON
)
if(${MESHTOOLS_TRIES_TO_USE_CGAL})
  find_package(CGAL QUIET COMPONENTS Core)
  if(CGAL_FOUND)
    if(DEFINED CGAL_VERSION)
      if(${CGAL_VERSION} VERSION_LESS 4.12)
        message(
          WARNING
            "WARNING CGAL version is too old ("
            ${CGAL_VERSION}
            ")"
            "MeshTools will be built without CGAL support. If you need CGAL, set CGAL_DIR to a directory containing CGAL 4.12 or greater."
        )
        set(MESHTOOLS_USES_CGAL FALSE)
      else(${CGAL_VERSION} VERSION_LESS 4.12)
        set(MESHTOOLS_USES_CGAL TRUE)
      endif(${CGAL_VERSION} VERSION_LESS 4.12)
    else(DEFINED CGAL_VERSION)
      message(
        WARNING
          "CGAL has been found but does not seem to define CGAL_VERSION. I'm trying to use it anyway but you might have to check that you have at least CGAL 4.12."
      )
      set(MESHTOOLS_USES_CGAL TRUE)
    endif(DEFINED CGAL_VERSION)
  else(CGAL_FOUND)
    set(MESHTOOLS_USES_CGAL FALSE)
  endif(CGAL_FOUND)
else(${MESHTOOLS_TRIES_TO_USE_CGAL})
  set(MESHTOOLS_USES_CGAL FALSE)
endif(${MESHTOOLS_TRIES_TO_USE_CGAL})
if(${MESHTOOLS_USES_CGAL})
  set(MESHTOOLS_CGAL_DIRECTORY ${MESHTOOLS_ROOT_DIRECTORY}/src/cgal)
endif(${MESHTOOLS_USES_CGAL})

option(MESHTOOLS_COMPILES_EXPERIMENTAL_FEATURES
       "Compiles experimental features." ON
)

# -- MeshTools source code --

add_subdirectory(src)
if(MESHTOOLS_MASTER_PROJECT)
  add_subdirectory(tests)
endif(MESHTOOLS_MASTER_PROJECT)

# -- targets and installation

# install(EXPORT MeshTools DESTINATION share/cmake/) the usual way is to pass
# the python package directory through MESHTOOLS_PYTHON_PACKAGE_DIRECTORY yet
# targets (python modules) are also exported directly from the build tree to use
# them you will need to reference the build tree cf.
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/Exporting-and-Importing-Targets
# export(EXPORT MeshTools FILE share/cmake/meshtools-exports.cmake)
