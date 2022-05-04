pybind11_add_module(
  CGALWrappers
  CGALWrappers.cpp
  EpickWrapper-module.cpp
  C3t3Wrapper-module.cpp
  DTMWrapper.cpp
  Polygon2Wrapper-module.cpp
  implicit_functions.cpp
  mesh_implicit_domains.cpp
)
target_compile_definitions(CGALWrappers PRIVATE BOOST_PARAMETER_MAX_ARITY=12)
target_link_libraries(CGALWrappers PRIVATE CGAL::CGAL)
if(ACTIVATE_CONCURRENCY AND TBB_FOUND)
  cgal_target_use_tbb(CGALWrappers)
endif()

pybind11_add_module(Corefinement corefinement.cpp)
target_link_libraries(Corefinement PRIVATE CGAL::CGAL)

install(
  TARGETS CGALWrappers Corefinement
  DESTINATION ${MESHTOOLS_PYTHON_PACKAGE}
  EXPORT MeshTools
)
