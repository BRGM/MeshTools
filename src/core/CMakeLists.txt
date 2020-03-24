# FIXME: to be removed -> export cmake targets
include_directories(${MESHTOOLS_THIRDPARTIES_DIRECTORY}/mapbox/include)

pybind11_add_module(_MeshTools meshtools-module.cpp meshtools-wrapper.cpp)

install(TARGETS _MeshTools DESTINATION .)
