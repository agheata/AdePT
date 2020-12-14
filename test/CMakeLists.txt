# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# Helper Macros/Functions
macro(build_cpu_tests TESTS)
  foreach(TEST ${TESTS})
    #set_source_files_properties(${TEST} PROPERTIES LANGUAGE CXX)
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    set(TARGET_NAME_CPU ${TARGET_NAME}_cpu)
    get_filename_component(TARGET_EXT ${TEST} LAST_EXT)
    if ("${TARGET_EXT}" MATCHES ".cu")
      set(SRC_FILE_CU ${CMAKE_CURRENT_SOURCE_DIR}/${TEST})
      set(SRC_FILE_CPP ${CMAKE_CURRENT_BINARY_DIR}/cuda_src/${TARGET_NAME}.cpp)

      add_custom_command(
        OUTPUT ${SRC_FILE_CPP}
        COMMAND ${CMAKE_COMMAND} -E copy ${SRC_FILE_CU} ${SRC_FILE_CPP}
        DEPENDS ${SRC_FILE_CU})
      add_executable(${TARGET_NAME_CPU} ${SRC_FILE_CPP})
    else()
      add_executable(${TARGET_NAME_CPU} ${TEST})
    endif()
    target_include_directories(${TARGET_NAME_CPU} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/test>
      $<INSTALL_INTERFACE:base>
    )
    target_link_libraries(${TARGET_NAME_CPU} VecCore::VecCore CopCore::CopCore_CPU)
   endforeach()
endmacro()

macro(build_cuda_tests TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    set(TARGET_NAME ${TARGET_NAME}_cuda)
    add_executable(${TARGET_NAME} ${TEST})
    target_include_directories(${TARGET_NAME} PUBLIC
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
      $<INSTALL_INTERFACE:base>
    )
    target_link_libraries(${TARGET_NAME} VecCore::VecCore CopCore::CopCore_CUDA)
   endforeach()
endmacro()

macro(add_to_cpu_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    set(TARGET_NAME ${TARGET_NAME}_cpu)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

macro(add_to_cuda_test TESTS)
  foreach(TEST ${TESTS})
    get_filename_component(TARGET_NAME ${TEST} NAME_WE)
    set(TARGET_NAME ${TARGET_NAME}_cuda)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
   endforeach()
endmacro()

# - Unit tests
set(ADEPT_UNIT_TESTS_CUDA
  test_atomic.cu               # Unit test for atomic ops
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_ranluxpp.cu             # Unit test for RANLUX++
  test_track_block.cu          # Unit test for BlockData
)

set(ADEPT_UNIT_TESTS_CPU
  test_atomic.cu              # Unit test for atomic ops
  test_queue.cu                # Unit test for mpmc_bounded_queue
  test_ranluxpp.cu             # Unit test for RANLUX++
  test_track_block.cu          # Unit test for BlockData
)

add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>")
add_compile_options("$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")

#add_executable(test_launcher test_launcher.cu test_launcher.cpp)
#target_include_directories(test_launcher PUBLIC
#  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
#  $<INSTALL_INTERFACE:base>
#)
#target_link_libraries(test_launcher VecCore::VecCore CopCore::CopCore_CUDA)

build_cpu_tests("${ADEPT_UNIT_TESTS_CPU}")
add_to_cpu_test("${ADEPT_UNIT_TESTS_CPU}")

if(COPCORE_ENABLE_CUDA)
  build_cuda_tests("${ADEPT_UNIT_TESTS_CUDA}")
  add_to_cuda_test("${ADEPT_UNIT_TESTS_CUDA}")
endif()
