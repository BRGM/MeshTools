# FIXME: to be removed -> export cmake targets
include_directories(${MESHTOOLS_THIRDPARTIES_DIRECTORY}/mapbox/include) 

pybind11_add_module(_MeshTools meshtools-module.cpp meshtools-wrapper.cpp)
set_target_properties(_MeshTools
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

install(TARGETS _MeshTools DESTINATION . EXPORT MeshTools)
