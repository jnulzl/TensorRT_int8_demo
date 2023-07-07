set(demo_src demo.cpp)
get_filename_component(exe_name ${demo_src} NAME_WLE)
add_executable(${exe_name} ${demo_src})
target_include_directories(${exe_name} PRIVATE  ${PROJECT_SOURCE_DIR}/include)

string(TOLOWER ${PROJECT_NAME} LIB_NAME)
target_compile_definitions(${exe_name} PRIVATE
        $<$<BOOL:${USE_${ENGINE_UPPER}}>:USE_${ENGINE_UPPER}>
        $<$<BOOL:${USE_${DEVICE_UPPER}}>:USE_${DEVICE_UPPER}>
        -DPROJECT_ROOT="${PROJECT_SOURCE_DIR}"
        )

target_link_options(${exe_name} PRIVATE
        -Wl,--no-undefined
        -Wl,--no-allow-shlib-undefined
#        -Wl,--enable-new-dtags # using RUNPATH (default)
        -Wl,--disable-new-dtags # using RPATH (default)
        )

target_link_libraries(${exe_name} PRIVATE
        ${LIB_NAME}
        opencv_core
        opencv_highgui
        opencv_imgproc
        opencv_imgcodecs
        opencv_videoio
        m
        )
