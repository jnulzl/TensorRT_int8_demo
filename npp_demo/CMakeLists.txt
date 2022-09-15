set(exe_name npp_demo)

set(srcs_all ${srcs_all} ${CMAKE_CURRENT_SOURCE_DIR}/npp_demo.cpp)
add_executable(${exe_name} ${srcs_all})

target_include_directories(${exe_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

target_link_options(
        ${exe_name} PRIVATE
        "-Wl,-rpath-link,${CUDA_HOME}/lib64"
        )

target_link_libraries(${exe_name} PRIVATE
        cuda
        cudart
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
