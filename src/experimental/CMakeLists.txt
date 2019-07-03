
pybind11_add_module(SplitManager split-manager.cpp)
set_target_properties(SplitManager
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

install(
    TARGETS SplitManager
    DESTINATION ${MESHTOOLS_PYTHON_PACKAGE_NAME}
    EXPORT MeshTools
)
