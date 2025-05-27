# Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)

project(sampleYOLOV11)

add_compile_options(-std=c++11)

add_definitions(-DENABLE_DVPP_INTERFACE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  "../out")
set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "set INC_PATH: ${INC_PATH}")
endif ()

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/7.0/runtime/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "set LIB_PATH: ${LIB_PATH}")
endif ()

set(THIRDPART $ENV{THIRDPART_PATH})
if (NOT DEFINED ENV{THIRDPART_PATH})
    set(THIRDPART "/usr/local/Ascend/thirdpart/aarch64")
    message(STATUS "set default THIRDPART: ${THIRDPART}")
else()
    message(STATUS "set THIRDPART: ${THIRDPART}")
endif()


find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(
   ${INC_PATH}/runtime/include/
   /usr/local/include/opencv4/
   ${THIRDPART}/include/acllite/
   ../common/include/
   /usr/include/
)

link_directories(
    ${THIRDPART}/lib/
    ${LIB_PATH}
    /usr/local/lib
)

add_executable(main
        sampleYOLOV11.cpp)

target_link_libraries(main 
    ${OpenCV_LIBS} 
    /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/lib64/libascendcl.so
)

if(target STREQUAL "Simulator_Function")
    target_link_libraries(main funcsim)
else()
    target_link_libraries(main ascendcl acl_dvpp stdc++ acllite opencv_core opencv_imgproc opencv_imgcodecs dl rt)
endif()

install(TARGETS main DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
