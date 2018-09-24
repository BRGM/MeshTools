# FIXME: this should be a dependence on the target MeshTools
include_directories(../core)

pybind11_add_module(URG urg.cpp)
set_target_properties(URG
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(URG PRIVATE CGAL::CGAL)

install(TARGETS URG DESTINATION ${MESHTOOLS_PYTHON_PACKAGE_NAME} EXPORT MeshTools)
