add_executable(test-DTM test-DTM.cpp)
target_include_directories(test-DTM PRIVATE ${MESHTOOLS_CGAL_DIRECTORY})
target_link_libraries(test-DTM PRIVATE CGAL::CGAL)
