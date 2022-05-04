pybind11_add_module(SplitManager split-manager.cpp)

install(
  TARGETS SplitManager
  DESTINATION ${MESHTOOLS_PYTHON_PACKAGE}
  EXPORT MeshTools
)
