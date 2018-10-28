# FIXME: to be removed -> export cmake targets
include_directories(${MESHTOOLS_THIRDPARTIES_DIRECTORY}/mapbox/include) 

pybind11_add_module(_MeshTools meshtools-module.cpp meshtools-wrapper.cpp)
set_target_properties(_MeshTools
    PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

# A first test on compressed storage structure
add_executable(Chunks_chain Chunks_chain.cpp)

install(TARGETS _MeshTools DESTINATION ${MESHTOOLS_PYTHON_PACKAGE_NAME} EXPORT MeshTools)
install(TARGETS Chunks_chain DESTINATION tests/experimental EXPORT MeshTools)
