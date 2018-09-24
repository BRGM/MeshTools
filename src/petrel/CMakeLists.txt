
pybind11_add_module(PetrelMesh petrel-mesh.cpp)
#target_compile_definitions(PetrelMesh PRIVATE BOOST_PARAMETER_MAX_ARITY=12)
set_target_properties(PetrelMesh
  PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)
target_link_libraries(PetrelMesh PRIVATE CGAL::CGAL)

install(TARGETS PetrelMesh DESTINATION ${MESHTOOLS_PYTHON_PACKAGE_NAME} EXPORT MeshTools)
