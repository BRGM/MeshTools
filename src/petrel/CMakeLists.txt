pybind11_add_module(PetrelMesh petrel-mesh.cpp)
target_link_libraries(PetrelMesh PRIVATE CGAL::CGAL)

install(TARGETS PetrelMesh DESTINATION .)
