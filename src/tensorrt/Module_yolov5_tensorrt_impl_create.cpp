#include "tensorrt/Module_yolov5_tensorrt_impl.h"
#include "Module_yolov5.h"

CModule_yolov5::CModule_yolov5(const std::string& engine_name)
{
    impl_ = new ALG_ENGINE_IMPL(yolov5, tensorrt);
}