# FIXME: this should be a dependence on the target MeshTools
include_directories(../core)

pybind11_add_module(URG urg.cpp)
target_link_libraries(URG PRIVATE CGAL::CGAL)

install(
  TARGETS URG
  DESTINATION ${MESHTOOLS_PYTHON_PACKAGE}
  EXPORT MeshTools
)
