set(exe_name onnx2trtInt8)

file(GLOB _tmp ${TRT_SAMPLES_COMMON_HOME}/*.cpp)
set(srcs_all ${srcs_all} ${_tmp})
set(srcs_all ${srcs_all}
        ${CMAKE_CURRENT_SOURCE_DIR}/onnx_to_trt_int8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/npp_image_preprocess.cpp
        )
add_executable(${exe_name} ${srcs_all})

target_include_directories(${exe_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TRT_SAMPLES_COMMON_HOME}
        )

target_link_options(
        ${exe_name} PRIVATE
        "-Wl,-rpath-link,${CUDNN_HOME}/lib64"
        "-Wl,-rpath-link,${TRT_HOME}/lib64"
        "-Wl,-rpath-link,${CUDA_HOME}/lib64"
        )

target_compile_definitions(${exe_name} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        )

target_link_libraries(${exe_name} PRIVATE
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
        ######################
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
        opencv_video
        m
        )
