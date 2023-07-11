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

    void process_batch(const ImageInfoUint8* imageInfos, int batch_size);

    const BoxInfos* get_result();

protected:
    virtual void pre_process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);
    virtual void pre_batch_process(const ImageInfoUint8* imageInfos, int batch_size);
    virtual void post_process();

    virtual void engine_init() = 0;
    virtual void engine_run() = 0;

protected:
    YoloConfig config_;

    std::vector<float> data_in_;
    NetFloatTensor net_input_float_tensor_;

    std::vector<uint8_t> src_resize_;
    std::vector<float> data_out_;

    std::vector<int> keep_indexs_;
    std::vector<int> img_heights_;
    std::vector<int> img_widths_;

    std::vector<BoxInfo> boxs_tmp_;
    std::vector<BoxInfos> boxs_batch_;

    int step_each_obj_;
};

#endif // MODULE_YOLOV5_IMPL_H

