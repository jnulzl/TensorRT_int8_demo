cmake_minimum_required(VERSION 3.15)
project(Yolov5TensorRT CUDA CXX C)

set(CMAKE_SYSTEM_NAME "Linux") # Linux Android

set(BUILD_SHARED_LIBS ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_COLOR_MAKEFILE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

message(STATUS "Operate System : " ${CMAKE_SYSTEM_NAME})
message(STATUS "Compiler ID : " ${CMAKE_CXX_COMPILER_ID})
message(STATUS "Compiler Version : " ${CMAKE_CXX_COMPILER_VERSION})

# Select Device
if(NOT DEVICE_LOWER)
    set(DEVICE_LOWER cuda) # cpu cuda
endif()
string(TOUPPER ${DEVICE_LOWER} DEVICE_UPPER)
set(USE_${DEVICE_UPPER} ON)
message(STATUS "Device type is : " ${DEVICE_LOWER})

# Select Engine
if(NOT ENGINE_LOWER)
    set(ENGINE_LOWER tensorrt)
endif()
string(TOUPPER ${ENGINE_LOWER} ENGINE_UPPER)
set(USE_${ENGINE_UPPER} ON)
message(STATUS "Engine type is : " ${ENGINE_LOWER})


set(OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/${CMAKE_SYSTEM_NAME})
if("Linux" STREQUAL ${CMAKE_SYSTEM_NAME})
    set(CMAKE_CUDA_STANDARD 14)
    #    set(CMAKE_SKIP_BUILD_RPATH ON)

    if(NOT CUDA_HOME)
        set(CUDA_HOME /usr/local/cuda)
    endif()
    message(STATUS "CUDA_HOME : " ${CUDA_HOME})

    if(NOT CUDNN_HOME)
        set(CUDNN_HOME ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/cudnn)
    endif()
    message(STATUS "CUDNN_HOME : " ${CUDNN_HOME})

    if(NOT TRT_HOME)
        set(TRT_HOME ${PROJECT_SOURCE_DIR}/third_party/${CMAKE_SYSTEM_NAME}/TensorRT)
    endif()
    message(STATUS "TRT_HOME :  " ${TRT_HOME})

    set(TRT_SAMPLES_COMMON_HOME ${TRT_HOME}/samples/common)

    ##############TensorRT##############
    set(TENSORRT_INCLUDE
            ${TRT_SAMPLES_COMMON_HOME}
            ${TRT_HOME}/include
            ${CUDA_HOME}/include
            ${CUDNN_HOME}/include
            )
    set(TENSORRT_LIB_PATH
            ${CUDA_HOME}/lib64
            ${CUDA_HOME}/lib
            ${CUDNN_HOME}/lib64
            ${CUDNN_HOME}/lib
            ${TRT_HOME}/lib64
            ${TRT_HOME}/lib
            )
endif()

##############OpenCV##############
if(NOT OpenCV_DIR)
    set(OpenCV_DIR "/usr")
endif()
message(STATUS "OpenCV_DIR :  " ${OpenCV_DIR})
set(OpenCV_INCLUDE
        ${OpenCV_DIR}/include
        ${OpenCV_DIR}/include/opencv2
        ${OpenCV_DIR}/include/opencv4
        ${OpenCV_DIR}/include/opencv4/opencv2)
set(OpenCV_LIB_PATH
        ${OpenCV_DIR}/lib
        ${OpenCV_DIR}/lib64
        ${OpenCV_DIR}/lib/x86_64-linux-gnu
        )

file(MAKE_DIRECTORY ${OUTPUT_DIRECTORY})
set(EXECUTABLE_OUTPUT_PATH ${OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${EXECUTABLE_OUTPUT_PATH})


set(INCLUDE_PATH ${INCLUDE_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE}
        ${OpenCV_INCLUDE})
set(LIB_PATH ${LIB_PATH}
        ${TENSORRT_LIB_PATH}
        ${OpenCV_LIB_PATH})

include_directories(${INCLUDE_PATH})
link_directories(${LIB_PATH})

add_subdirectory(tensorrt_int8_demo)
add_subdirectory(npp_demo)

add_subdirectory(src)
add_subdirectory(examples)
