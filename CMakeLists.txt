project(SDF_UTILITIES)
cmake_minimum_required(VERSION 2.8.11)

if(${CMAKE_BUILD_TYPE}x STREQUAL "Debugx")
   add_definitions(-DSDF_DEBUG_ALL)
   set(DEBUG_OPT --debug)
   set(OPT_LEVEL 0)
   set(EXTRA_COMPILE_ARGS -gdwarf-2)
else()
   set(OPT_LEVEL 3)
endif()
add_definitions(-D_XOPEN_SOURCE=600)

include_directories(${SDFC_INCLUDE_DIR})


execute_process(COMMAND sh gen_commit_string.sh
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sdf2ascii sdf2ascii.c)
target_link_libraries(sdf2ascii sdfc dl)

add_executable(sdffilter sdffilter.c)
target_link_libraries(sdffilter sdfc dl m)

add_custom_target(sdf-utilities DEPENDS sdf2ascii sdffilter)

find_package(PythonInterp)
configure_file(setup.py.in setup.py)
configure_file(setup.cfg.in setup.cfg)
add_custom_command(OUTPUT build/timestamp
   COMMAND ${PYTHON_EXECUTABLE} setup.py build ${DEBUG_OPT}
   COMMAND ${CMAKE_COMMAND} -E touch build/timestamp
   DEPENDS setup.py sdf_python.c sdfc)
add_custom_target(sdfpy ALL DEPENDS build/timestamp)

install(TARGETS sdf2ascii sdffilter DESTINATION bin COMPONENT sdf-utilities)

configure_file(sdfpy_install.cmake.in sdfpy_install.cmake)
install(SCRIPT ${PROJECT_BINARY_DIR}/sdfpy_install.cmake COMPONENT sdfpy)
