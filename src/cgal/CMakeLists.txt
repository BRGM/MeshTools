
pybind11_add_module(
    CGALWrappers CGALWrappers.cpp EpickWrapper-module.cpp C3t3Wrapper-module.cpp DTMWrapper.cpp
    Polygon2Wrapper-module.cpp
    implicit_functions.cpp mesh_implicit_domains.cpp
)
target_compile_definitions(CGALWrappers PRIVATE BOOST_PARAMETER_MAX_ARITY=12)
set_target_properties(CGALWrappers
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(CGALWrappers PRIVATE CGAL::CGAL)
if(ACTIVATE_CONCURRENCY AND TBB_FOUND)
    CGAL_target_use_TBB(CGALWrappers)
endif()

pybind11_add_module(Corefinement corefinement.cpp)
#target_compile_definitions(Corefinement PRIVATE BOOST_PARAMETER_MAX_ARITY=12)
set_target_properties(Corefinement
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(Corefinement PRIVATE CGAL::CGAL)

install(TARGETS CGALWrappers Corefinement DESTINATION ${MESHTOOLS_PYTHON_PACKAGE_NAME} EXPORT MeshTools)
