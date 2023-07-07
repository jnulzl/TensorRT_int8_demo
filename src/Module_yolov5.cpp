#include "Module_yolov5_impl.h"
#include "Module_yolov5.h"

#include "debug.h"


CModule_yolov5::~CModule_yolov5()
{
    delete ANY_POINTER_CAST(impl_, CModule_yolov5_impl);
#if defined(ALG_DEBUG) || defined(AI_ALG_DEBUG)
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
}

void CModule_yolov5::init(const YoloConfig& config)
{
    ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->init(config);
}

void CModule_yolov5::process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType)
{
    ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->process(src, src_height, src_width, inputDataType);
}

const BoxInfos* CModule_yolov5::get_result()
{
    return ANY_POINTER_CAST(impl_, CModule_yolov5_impl)->get_result();
}
