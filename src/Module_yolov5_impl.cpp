#include <assert.h>
#include <string.h>

#include "Module_yolov5_impl.h"
#include "pre_process.h"
#include "post_process.h"
#include "debug.h"


CModule_yolov5_impl::CModule_yolov5_impl()
{

}
CModule_yolov5_impl::~CModule_yolov5_impl() 
{
#ifdef AI_ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
    for (int idx = 0; idx < config_.batch_size; ++idx)
    {
        delete[] boxs_batch_[idx].boxes;
    }
}

void CModule_yolov5_impl::init(const YoloConfig& config)
{
	config_ = config;
    engine_init();

    net_input_float_tensor_.batch = config_.batch_size;
    net_input_float_tensor_.channels= config_.net_inp_channels;
    net_input_float_tensor_.height = config_.net_inp_height;
    net_input_float_tensor_.width = config_.net_inp_width;
    net_input_float_tensor_.format = NetFloatTensor::DimensionType::NCHW;
    data_in_.resize(config_.net_inp_channels * config_.net_inp_width * config_.net_inp_height + 64);
    src_resize_.resize(data_in_.size());
    net_input_float_tensor_.data = data_in_.data();

    img_heights_.resize(config_.batch_size);
    img_widths_.resize(config_.batch_size);

    boxs_batch_.resize(config_.batch_size);
    for (int idx = 0; idx < config_.batch_size; ++idx)
    {
        boxs_batch_[idx].boxes = new BoxInfo[MAX_DET_NUM];
        boxs_batch_[idx].capacity = MAX_DET_NUM;
        boxs_batch_[idx].size = 0;
    }

    step_each_obj_ = 5 + config_.num_cls;
}

void CModule_yolov5_impl::pre_process(const uint8_t *src, int src_height, int src_width, InputDataType inputDataType)
{
    size_t num_channels;
    switch (inputDataType)
    {
        case InputDataType::IMG_BGR:
        case InputDataType::IMG_RGB:
            num_channels = 3;
            break;
        case InputDataType::IMG_GRAY:
            num_channels = 1;
            break;
        default:
            num_channels = 4;
    }
    size_t src_stride = num_channels * src_width;
    size_t des_stride = num_channels * config_.net_inp_width;

    memset(src_resize_.data(), 0, src_resize_.size());

    ai_utils_resize_with_affine(src, src_height, src_width, src_stride,
                                src_resize_.data(), config_.net_inp_height, config_.net_inp_width,
                                des_stride,
                                num_channels, 1);
    if(config_.model_include_preprocess)
    {
        int num_data = src_resize_.size();
        for(int idx = 0; idx < num_data; idx++)
        {
            data_in_[idx] = 1.0f * src_resize_[idx];
        }
    }
    else
    {
        color_normalize_scale_and_chw(src_resize_.data(), config_.net_inp_height, config_.net_inp_width,
                                      config_.net_inp_channels * config_.net_inp_width, config_.means, config_.scales,
                                      data_in_.data(), 1,
#if defined(USE_TFLITE) || defined(USE_TFLITEGPU)
                0);
#else
                                      1);
#endif
    }
}

void CModule_yolov5_impl::pre_batch_process(const ImageInfoUint8* imageInfos, int batch_size)
{
    //pre_batch_process cpu version
}

void CModule_yolov5_impl::process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType)
{
    img_heights_[0] = src_height;
    img_widths_[0] = src_width;

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

    pre_process(src, src_height, src_width, inputDataType);

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("Preprocess time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
#endif

    engine_run();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_run = std::chrono::system_clock::now();
    std::printf("Inference time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time_run - end_time).count());
#endif

    post_process();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_post = std::chrono::system_clock::now();
    std::printf("Postprocess time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time_post - end_time_run).count());
#endif
}

void CModule_yolov5_impl::process_batch(const ImageInfoUint8* imageInfos, int batch_size)
{
    //TODO : check if batch_size == config_.batch_size
    for (int bs = 0; bs < config_.batch_size; ++bs)
    {
        img_heights_[bs] = imageInfos[bs].img_height;
        img_widths_[bs] = imageInfos[bs].img_width;
    }

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif

    pre_batch_process(imageInfos, batch_size);

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("Preprocess time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count());
#endif

    engine_run();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_run = std::chrono::system_clock::now();
    std::printf("Inference time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time_run - end_time).count());
#endif

    post_process();

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time_post = std::chrono::system_clock::now();
    std::printf("Postprocess time %ld us\n", std::chrono::duration_cast<std::chrono::microseconds>(end_time_post - end_time_run).count());
#endif
}


void CModule_yolov5_impl::post_process()
{
    int num_obj = data_out_.size() / step_each_obj_ / config_.batch_size;
    if(keep_indexs_.size() < num_obj)
    {
        keep_indexs_.resize(num_obj);
        boxs_tmp_.resize(num_obj);
    }

    for (int bs = 0; bs < config_.batch_size; ++bs)
    {
        int det_obj = 0;
        non_max_suppression(data_out_.data() + bs * num_obj * step_each_obj_, num_obj, step_each_obj_, config_.conf_thres, config_.nms_thresh,
                            config_.net_inp_height, config_.net_inp_width, img_heights_[bs], img_widths_[bs],
                            boxs_tmp_.data(), keep_indexs_.data(), &det_obj);
        for (int idx = 0; idx < det_obj; ++idx)
        {
            if(boxs_batch_[bs].capacity < det_obj)
            {
                delete[] boxs_batch_[bs].boxes;
                boxs_batch_[bs].boxes = new BoxInfo[det_obj];
                boxs_batch_[bs].capacity = det_obj;
                boxs_batch_[bs].size = 0;
            }
            boxs_batch_[bs].boxes[idx] = boxs_tmp_[keep_indexs_[idx]];
        }
        boxs_batch_[bs].size = det_obj;
    }
}

const BoxInfos* CModule_yolov5_impl::get_result()
{
    return boxs_batch_.data();
}
