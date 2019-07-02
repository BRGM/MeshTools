add_executable(test-DTM test-DTM.cpp)
target_include_directories(test-DTM PRIVATE ${MESHTOOLS_CGAL_DIRECTORY})
target_link_libraries(test-DTM PRIVATE CGAL::CGAL)

add_executable(test-petrel test-petrel.cpp)
target_include_directories(test-petrel PRIVATE ${MESHTOOLS_CGAL_DIRECTORY})
target_include_directories(test-petrel PRIVATE ${CMAKE_SOURCE_DIR}/src/petrel)
target_link_libraries(test-petrel PRIVATE CGAL::CGAL)
