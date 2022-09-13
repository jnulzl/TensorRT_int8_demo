#ifndef MODULE_YOLOV5_IMPL_H
#define MODULE_YOLOV5_IMPL_H

#include <string>
#include <vector>

#include "data_type.h"

class CModule_yolov5_impl
{
public:
	CModule_yolov5_impl();
	virtual ~CModule_yolov5_impl() ;

    void init(const YoloConfig &config);

	void process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);

	const std::vector<BoxInfo>& get_result();

protected:
    virtual void pre_process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);
    virtual void post_process();

    virtual void engine_init() = 0;
    virtual void engine_run() = 0;

protected:
    YoloConfig config_;

    std::vector<float> data_in_;
    NetFloatTensor net_input_float_tensor_;

    std::vector<uint8_t> src_resize_;
    std::vector<float> data_out_;

    int step_each_obj_;
	std::vector<BoxInfo> boxs_;
    int img_height_;
    int img_width_;
#if !defined(USE_DETECT_LAYER)
    std::vector<std::vector<float>> output_tmp_;
#endif
};

#endif // MODULE_YOLOV5_IMPL_H

