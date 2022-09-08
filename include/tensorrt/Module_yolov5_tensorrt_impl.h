//
// Created by lizhaoliang-os on 2020/6/9.
//

#ifndef MODULE_YOLOV5_TENSORRT_IMPL_H
#define MODULE_YOLOV5_TENSORRT_IMPL_H

#include <memory>

#include <NvInferRuntime.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <buffers.h>
#include <common.h> // for include samplesCommon::InferDeleter

#include <npp.h> // For gpu preprocess function(affineTransform, bgr2rgb, chw, (x - a) * b)

#include "Module_yolov5_impl.h"

class CModule_yolov5_tensorrt_impl : public CModule_yolov5_impl
{
public:
    CModule_yolov5_tensorrt_impl();
    virtual ~CModule_yolov5_tensorrt_impl();

private:
    virtual void engine_init() override;
    virtual void engine_run() override;

    virtual void pre_process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR) override;
    
    void pre_process_cpu(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);
    void pre_process_gpu(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);
private:

    template <typename T>
    using TRTUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

    //!
    //! \brief Parses an ONNX model and creates a TensorRT network
    //!
    bool constructNetwork(TRTUniquePtr<nvinfer1::IBuilder>& builder,
                          TRTUniquePtr<nvinfer1::INetworkDefinition>& network, TRTUniquePtr<nvinfer1::IBuilderConfig>& config,
                          TRTUniquePtr<nvonnxparser::IParser>& parser);

    bool isSupported(DataType dataType);

private:
    TRTUniquePtr<nvinfer1::INetworkDefinition> network_;
    TRTUniquePtr<nvinfer1::IBuilder> builder_;
    TRTUniquePtr<nvinfer1::IBuilderConfig> config_trt_;
    TRTUniquePtr<nvonnxparser::IParser> parser_;

    std::shared_ptr<nvinfer1::ICudaEngine> net_; //!< The TensorRT engine used to run the network

    // Create RAII buffer manager object
    std::unique_ptr<samplesCommon::BufferManager> buffers_;
    TRTUniquePtr<nvinfer1::IExecutionContext> context_;

    // For gpu preprocess
    size_t src_pixel_num_pre_;
    size_t dst_pixel_num_;
    Npp8u* src_ptr_d_;
    Npp8u* dst_ptr_d_;
    Npp32f* dst_float_ptr_d_;
    Npp32f * dst_chw_float_ptr_d_;
};

#endif //MODULE_YOLOV5_TENSORRT_IMPL_H
