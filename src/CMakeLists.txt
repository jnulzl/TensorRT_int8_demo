
file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(src_all ${src_all} ${_tmp})

if(USE_${ENGINE_UPPER})
    file(GLOB _tmp ${CMAKE_CURRENT_SOURCE_DIR}/${ENGINE_LOWER}/*.cpp)
    set(src_all ${src_all} ${_tmp})
endif()

if(USE_TENSORRT)
    set(src_all ${src_all} ${TRT_SAMPLES_COMMON_HOME}/logger.cpp)
endif()

string(TOLOWER ${PROJECT_NAME} LIB_NAME)

if (BUILD_SHARED_LIBS)
    add_library(${LIB_NAME} SHARED ${src_all})
    set_target_properties(${LIB_NAME} PROPERTIES VERSION 1.0.0 SOVERSION 1)
else()
    set(src_all ${src_all})
    add_library(${LIB_NAME} STATIC ${src_all})
endif ()
target_compile_options(${LIB_NAME} PRIVATE
        -fvisibility=hidden -fvisibility-inlines-hidden
        )

target_include_directories(${LIB_NAME} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/ghc
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:${PROJECT_SOURCE_DIR}/include/${ENGINE_LOWER}>
        )

target_compile_definitions(${LIB_NAME} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_options(${LIB_NAME} PRIVATE
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
        )

if(USE_CUDA)
    target_link_options(
            ${LIB_NAME} PUBLIC
            "-Wl,-rpath-link,${CUDNN_HOME}/lib64"
            "-Wl,-rpath-link,${TRT_HOME}/lib64"
            "-Wl,-rpath-link,${CUDA_HOME}/lib64"
    )

    target_link_libraries(${LIB_NAME} PUBLIC
    #        cuda
            cudart
            nvinfer
            nvparsers
            nvonnxparser
            nvinfer_plugin
            pthread
            dl
            nppig
            nppidei
            nppial
            )
endif()
