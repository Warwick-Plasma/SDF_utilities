project(SDF_UTILITIES)
cmake_minimum_required(VERSION 2.8.11)

if(${CMAKE_BUILD_TYPE}x STREQUAL "Debugx")
   add_definitions(-DSDF_DEBUG_ALL)
endif()
add_definitions(-D_XOPEN_SOURCE=600)

include_directories(${SDFC_INCLUDE_DIR})

add_executable(sdf2ascii sdf2ascii.c)
target_link_libraries(sdf2ascii sdfc dl)

add_executable(sdffilter sdffilter.c)
target_link_libraries(sdffilter sdfc dl m)

find_program(PYTHON "python")
configure_file(setup.py.in setup.py)
add_custom_command(OUTPUT pybuild/timestamp
   COMMAND ${PYTHON} setup.py build -b pybuild
   COMMAND ${CMAKE_COMMAND} -E touch pybuild/timestamp
   DEPENDS setup.py sdf_python.c sdfc)
add_custom_target(sdfpy ALL DEPENDS pybuild/timestamp)
