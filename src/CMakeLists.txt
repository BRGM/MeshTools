# CGAL and its components
if(MESHTOOLS_USES_CGAL)
    # Boost and its components
    # FIXME: should fail silently without including CGAL
    find_package( Boost REQUIRED )
    if ( NOT Boost_FOUND )
      message(STATUS "This project requires the Boost library, and will not be compiled.")
      return()
    endif()
    option(ACTIVATE_CONCURRENCY "Enable concurrency" OFF)
    if( ACTIVATE_CONCURRENCY )
        find_package( TBB )
        if( NOT TBB_FOUND )
            message( STATUS "NOTICE: Intel TBB NOT found! No parallelism activated." )
        endif()
    endif()
    add_definitions(-DMESHTOOLS_USES_CGAL)
    add_subdirectory(cgal)
    add_subdirectory(petrel)
    add_subdirectory(URG)
endif(MESHTOOLS_USES_CGAL)

if(MESHTOOLS_COMPILES_EXPERIMENTAL_FEATURES)
    add_subdirectory(experimental)
endif(MESHTOOLS_COMPILES_EXPERIMENTAL_FEATURES)

add_subdirectory(core)
