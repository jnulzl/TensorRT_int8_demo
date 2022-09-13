//
// Created by jnulzl on 2022/9/8.
//

#ifndef YOLOV5TENSORRT_NPP_IMAGE_PREPROCESS_HPP
#define YOLOV5TENSORRT_NPP_IMAGE_PREPROCESS_HPP

#include "alg_define.h"
#include "argsParser.h"

//!
//! \brief The SampleINT8Params structure groups the additional parameters required by
//!         the INT8 sample.
//!
struct SampleINT8Params : public samplesCommon::OnnxSampleParams
{
    int nbCalBatches;                            //!< The number of batches for calibration
    int calBatchSize;                            //!< The calibration batch size
    std::string networkName;                     //!< The name of the network
    std::string onnxFilePath;                    //!< The onnx model path for calibration
    std::string engine_file_save_path;           //!< The TensorRT int8 model file path
    std::string preprocessed_binary_file_path;   //!< The calibration image(preprocessed float data)
    std::string img_list_file;                   //!< The image list txt file for calibration model
    int numImages;                               //!< The number for calibration image
    int imageChannels;                           //!< The channels for network input
    int imageHeight;                             //!< The height for network input
    int imageWidth;                              //!< The width for network input
    float means[3];                              //!< The mean for network
    float scales[3];                             //!< The scale for network
    bool isFixResize;                            //!< The equal ratio resize
    bool isSymmetryPad;                          //!< If Symmetry pad, only valid when isFixResize is true
    bool isBGR2RGB;                              //!< If convert bgr to rgb
    bool isHWC2CHW;                              //!< If convert hwc to chw
    bool is_save_jpg_after_bgr2rgb;               //!< If save image when convert bgr(rgb) to rgb(bgr)
};

bool is_support_int8();

bool set_quant_params(const std::string& config_json_path, SampleINT8Params& params);

void preprocess_func(const SampleINT8Params& params, std::vector<float>& net_input_preprocessed_data);

size_t get_image_num(const SampleINT8Params& params);

#endif //YOLOV5TENSORRT_NPP_IMAGE_PREPROCESS_HPP
