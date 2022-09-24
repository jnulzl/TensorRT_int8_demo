//
// Created by lizhaoliang-os on 2020/6/9.
//
#include <cstring>
#include <npp.h>
#include "tensorrt/Module_yolov5_tensorrt_impl.h"
#include "logger.h"
#include "common.h"

#include "alg_define.h"
#include "post_process.h"
#include "debug.h"

 #if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || (defined(__cplusplus) && __cplusplus >= 201703L)) && defined(__has_include)
 #if __has_include(<filesystem>) && (!defined(__MAC_OS_X_VERSION_MIN_REQUIRED) || __MAC_OS_X_VERSION_MIN_REQUIRED >= 101500)
 #define GHC_USE_STD_FS
 #include <filesystem>
 namespace fs = std::filesystem;
 #endif
 #endif
 #ifndef GHC_USE_STD_FS
 #include <ghc/filesystem.hpp>
 namespace fs = ghc::filesystem;
 #endif


CModule_yolov5_tensorrt_impl::CModule_yolov5_tensorrt_impl()
{

}

CModule_yolov5_tensorrt_impl::~CModule_yolov5_tensorrt_impl()
{
#ifdef AI_ALG_DEBUG
    std::printf("%d,%s\n", __LINE__, __FUNCTION__);
#endif
    cudaFastFree(src_ptr_d_);
    cudaFastFree(dst_ptr_d_);
    cudaFastFree(dst_float_ptr_d_);
    cudaFastFree(dst_chw_float_ptr_d_);
}

void CModule_yolov5_tensorrt_impl::engine_init()
{
    CUDACHECK(cudaSetDevice(config_.device_id));

    // config_.weights_path is onnx model
    std::string engine_file_path = config_.weights_path.substr(0, config_.weights_path.size() - 4) + "trt";
    if("trt" == config_.weights_path.substr(config_.weights_path.size() - 3, 3))
    {
        engine_file_path = config_.weights_path;
    }
    if(!fs::exists(engine_file_path))
    {
        engine_file_path = config_.weights_path + ".trt";
    }
    if(fs::exists(engine_file_path))
    {
#ifdef AI_ALG_DEBUG
        std::cout << "Using TensorRT engine : " << engine_file_path << std::endl;
#endif
        std::ifstream engineFile(engine_file_path, std::ios::binary);
        if (!engineFile)
        {
            std::cerr << "Error opening engine file: " << engine_file_path << std::endl;
        }

        engineFile.seekg(0, engineFile.end);
        long int fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);

        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);
        if (!engineFile)
        {
            std::cerr << "Error loading engine file: " << engine_file_path << std::endl;
        }

        TRTUniquePtr<nvinfer1::IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        if (config_.dlaCore != -1)
        {
            runtime->setDLACore(config_.dlaCore);
        }

        net_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                runtime->deserializeCudaEngine(engineData.data(), engineData.size()), samplesCommon::InferDeleter());
    }
    else
    {
        builder_ = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder_)
        {
            AIWORKS_ASSERT(0);
        }

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        network_ = TRTUniquePtr<nvinfer1::INetworkDefinition>(builder_->createNetworkV2(explicitBatch));
        if (!network_)
        {
            AIWORKS_ASSERT(0);
        }

        config_trt_ = TRTUniquePtr<nvinfer1::IBuilderConfig>(builder_->createBuilderConfig());
        if (!config_trt_)
        {
            AIWORKS_ASSERT(0);
        }

        parser_ = TRTUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network_, sample::gLogger.getTRTLogger()));

        if (!parser_)
        {
            AIWORKS_ASSERT(0);
        }

        auto constructed = constructNetwork(builder_, network_, config_trt_, parser_);
        if (!constructed)
        {
            AIWORKS_ASSERT(0);
        }

        net_ = std::shared_ptr<nvinfer1::ICudaEngine>(
                builder_->buildEngineWithConfig(*network_, *config_trt_), samplesCommon::InferDeleter());

        if (!net_)
        {
            AIWORKS_ASSERT(0);
        }

        // serialize() to disk for fast loader later
        if(!fs::exists(engine_file_path))
        {
            std::ofstream engineFile(engine_file_path, std::ios::binary);
            if (!engineFile)
            {
                std::cerr << "Cannot open engine file: " << engine_file_path << std::endl;
            }

            TRTUniquePtr<nvinfer1::IHostMemory> serializedEngine{net_->serialize()};
            if (!serializedEngine)
            {
                std::cerr << "Engine serialization failed" << std::endl;
            }
#ifdef AI_ALG_DEBUG
            std::cout << "Writing TensorRT engine to " << engine_file_path << std::endl;
#endif

            engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
            engineFile.close();

#ifdef AI_ALG_DEBUG
            std::cout << "Writing TensorRT engine finished!" <<  std::endl;
#endif
        }
    }
    assert(network_->getNbInputs() == config_.input_names.size());
//    mInputDims = network->getInput(0)->getDimensions();
//    assert(mInputDims.nbDims == 4);

    assert(network_->getNbOutputs() == config_.output_names.size());
//    mOutputDims = network->getOutput(0)->getDimensions();
//    assert(mOutputDims.nbDims == 2);

    // Create RAII buffer manager object
    buffers_ = std::make_unique<samplesCommon::BufferManager>(net_, config_.batch_size);
    context_ = TRTUniquePtr<nvinfer1::IExecutionContext>(net_->createExecutionContext());
    if (!context_)
    {
        AIWORKS_ASSERT(0);
    }

    // For gpu preprocess
    /************Device memory allocator and initialization***********/
    src_ptr_d_ = nullptr;
    src_pixel_num_pre_ = 0;
    dst_pixel_num_ = config_.net_inp_height  * config_.net_inp_width * config_.net_inp_channels;
    dst_ptr_d_ = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * dst_pixel_num_));
    if(!dst_ptr_d_)
    {
        AIWORKS_ASSERT(0);
    }
    dst_float_ptr_d_ = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num_));
    if(!dst_float_ptr_d_)
    {
        AIWORKS_ASSERT(0);
    }
    dst_chw_float_ptr_d_  = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num_));
    if(!dst_chw_float_ptr_d_)
    {
        AIWORKS_ASSERT(0);
    }
}

void CModule_yolov5_tensorrt_impl::pre_process_cpu(const uint8_t *src, int src_height, int src_width,
                                               InputDataType inputDataType)
{
#ifdef AI_ALG_DEBUG
    std::cout << "Using pre_process_cpu" <<  std::endl;
#endif
    CModule_yolov5_impl::pre_process(src, src_height, src_width, inputDataType);

    // Read the input data into the managed buffers
    memcpy(buffers_->getHostBuffer(config_.input_names[0]),
           net_input_float_tensor_.data,
           sizeof(float) * config_.batch_size * config_.net_inp_channels * config_.net_inp_height * config_.net_inp_width);

    // Memcpy from host input buffers to device input buffers
    buffers_->copyInputToDevice();
}

void CModule_yolov5_tensorrt_impl::pre_process_gpu(const uint8_t *src, int src_height, int src_width,
                                                   InputDataType inputDataType)
{
#ifdef AI_ALG_DEBUG
    std::cout << "Using pre_process_gpu" <<  std::endl;
#endif
    /************Device memory allocator and initialization***********/
    size_t src_pixel_num = src_height * src_width * config_.net_inp_channels;
    if(src_pixel_num_pre_ < src_pixel_num)
    {
        cudaFastFree(src_ptr_d_);
        src_pixel_num_pre_ = src_pixel_num;
        src_ptr_d_ = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * src_pixel_num_pre_));
    }
    CUDACHECK(cudaMemcpy(src_ptr_d_, src, sizeof(Npp8u) * src_pixel_num, cudaMemcpyHostToDevice));

    /**********************getAffineTransform*************************/
    NppiRect oSrcROI = {.x=0, .y=0, .width=src_width, .height=src_height};
    double aQuad[4][2] = {{0.0, 0.0}, {1.0 * config_.net_inp_width, 0.0},
                          {1.0 * config_.net_inp_width, 1.0 * config_.net_inp_height},
                          {0, 1.0 * config_.net_inp_height}};
    bool is_resize_with_pad = true;
    if(is_resize_with_pad)
    {
        float roi_width = src_width;
        float roi_height = src_height;

        float scale_wh = 1.0 * std::fmax(1.0 * config_.net_inp_height, 1.0 * config_.net_inp_width) /
                         std::fmax(1.0 * src_height, 1.0 * src_width);
        float roi_new_width = roi_width * scale_wh;
        float roi_new_height = roi_height * scale_wh;

//        float x = (config_.net_inp_width - roi_new_width) / 2.0f;
//        float y = (config_.net_inp_height - roi_new_height) / 2.0f;
        float x = 0;
        float y = 0;
        aQuad[0][0] = x, aQuad[0][1] = y;
        aQuad[1][0] = x + roi_new_width, aQuad[1][1] = y;
        aQuad[2][0] = x + roi_new_width, aQuad[2][1] = y + roi_new_height;
        aQuad[3][0] = x, aQuad[3][1] = y + roi_new_height;
    }
    double aCoeffs[2][3];
    nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);

    /**********************warpAffine**********************/
    // nppiWarpAffine_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    //                            Npp8u * pDst, int nDstStep, NppiRect oDstROI,
    //                      const double aCoeffs[2][3], int eInterpolation)
    nppiWarpAffine_8u_C3R(src_ptr_d_, {src_width, src_height},
                          sizeof(Npp8u) * src_width * config_.net_inp_channels,
                          {0, 0, src_width, src_height},
                          dst_ptr_d_, sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                          {0, 0, config_.net_inp_width, config_.net_inp_height},
                          aCoeffs,
                          NPPI_INTER_LINEAR);
    /**********************bgr2rgb*************************/
    const int aDstOrder[3] = {2, 1, 0};
    nppiSwapChannels_8u_C3IR(dst_ptr_d_,
                             sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                             {config_.net_inp_width, config_.net_inp_height},
                             aDstOrder);

    /**********************uint8 -> float*****************/
    nppiConvert_8u32f_C3R(dst_ptr_d_, sizeof(Npp8u) * config_.net_inp_width * config_.net_inp_channels,
                          dst_float_ptr_d_,
                          sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                          {config_.net_inp_width, config_.net_inp_height});

    /**********************(x - a) / b*********************/
    /*1.-------- y = (x - a) --------*/
    // const Npp32f means[3] = {0.0f, 0.0f, 0.0f};
    nppiSubC_32f_C3IR(config_.means, dst_float_ptr_d_, sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels, {config_.net_inp_width, config_.net_inp_height});

    /*2.---------- y * s ----------*/
    // const Npp32f scales[3] = {0.00392157f, 0.00392157f, 0.00392157f};
    nppiMulC_32f_C3IR(config_.scales, dst_float_ptr_d_, sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels, {config_.net_inp_width, config_.net_inp_height});

    /**********************hwc2chw*************************/
    Npp32f * const aDst[3] = {dst_chw_float_ptr_d_,
                              dst_chw_float_ptr_d_ + config_.net_inp_width * config_.net_inp_height,
                              dst_chw_float_ptr_d_ + 2 * config_.net_inp_width * config_.net_inp_height};
    nppiCopy_32f_C3P3R(dst_float_ptr_d_,
                       sizeof(Npp32f) * config_.net_inp_width * config_.net_inp_channels,
                       aDst,
                       sizeof(Npp32f) * config_.net_inp_width,
                       {config_.net_inp_width, config_.net_inp_height});


    /*------copy preprocessed data to net input poiner------*/
    CUDACHECK(cudaMemcpy(buffers_->getDeviceBuffer(config_.input_names[0]), dst_chw_float_ptr_d_,
                     sizeof(Npp32f) * dst_pixel_num_, cudaMemcpyDeviceToDevice));
}

void CModule_yolov5_tensorrt_impl::pre_process(const uint8_t *src, int src_height, int src_width,
                                               InputDataType inputDataType)
{
    //pre_process_cpu(src, src_height, src_width, inputDataType);
    pre_process_gpu(src, src_height, src_width, inputDataType);
}

void CModule_yolov5_tensorrt_impl::engine_run()
{
#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
#endif
    bool status = context_->executeV2(buffers_->getDeviceBindings().data());
    if (!status)
    {
        AIWORKS_ERROR("Error %d line in file %s", __LINE__, __FILE__);
    }

#ifdef AI_ALG_DEBUG
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::printf("TensorRT inference time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif

#ifdef AI_ALG_DEBUG
    begin_time = std::chrono::system_clock::now();
#endif

    buffers_->copyOutputToHost();

    // get output
    int net_output_num = config_.output_names.size();
    // The numbers of output element;
    long out_data_num = 0;
    for (size_t idx = 0; idx < net_output_num; idx++)
    {
        auto mOutputDims = net_->getBindingDimensions(net_->getBindingIndex(config_.output_names[idx].c_str()));
//            auto mOutputDims = network_->getOutput(idx)->getDimensions();
        int num_of_elems = 1;
        for (int idy = 0; idy < mOutputDims.nbDims; ++idy)
        {
            num_of_elems *= mOutputDims.d[idy];
        }
        out_data_num += num_of_elems;
    }

    if (data_out_.size() < out_data_num)
    {
#ifdef AI_ALG_DEBUG
        std::cout << "Resize data_out_ : " << out_data_num << std::endl;
#endif
        data_out_.resize(out_data_num);
    }

    float* data_ = data_out_.data();
    int step_tmp = 0;
    for (size_t idx = 0; idx < net_output_num; idx++)
    {
        auto mOutputDims = net_->getBindingDimensions(net_->getBindingIndex(config_.output_names[idx].c_str()));
        int step = 1;
        for (int idy = 0; idy < mOutputDims.nbDims; ++idy)
        {
            step *= mOutputDims.d[idy];
        }
        float* output = static_cast<float*>(buffers_->getHostBuffer(config_.output_names[idx]));
        memcpy(data_ + idx * step_tmp, output, sizeof(float) * step);
        if(1 != net_output_num)
        {
            decode_net_output(data_ + idx * step_tmp, mOutputDims.d[0], mOutputDims.d[1], mOutputDims.d[2],
                              mOutputDims.d[3], config_.strides[idx], config_.anchor_grids[idx].data());
        }
        step_tmp += step;
    }
#ifdef AI_ALG_DEBUG
    end_time = std::chrono::system_clock::now();
    std::printf("postprocess time %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count());
#endif
}

bool CModule_yolov5_tensorrt_impl::constructNetwork(TRTUniquePtr<nvinfer1::IBuilder> &builder,
                                                    TRTUniquePtr<nvinfer1::INetworkDefinition> &network,
                                                    TRTUniquePtr<nvinfer1::IBuilderConfig> &config,
                                                    TRTUniquePtr<nvonnxparser::IParser> &parser) {

    auto parsed = parser->parseFromFile(config_.weights_path.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(config_.batch_size);
    config->setMaxWorkspaceSize(2_GiB);
    if (config_.fp16)
    {
        if(isSupported(DataType::kHALF))
        {
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
    }
    if (config_.int8)
    {
        if(isSupported(DataType::kINT8))
        {
            config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }
#if NV_TENSORRT_MAJOR >= 8
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
#else
        samplesCommon::setAllTensorScales(network.get(), 127.0f, 127.0f);
#endif
    }

    samplesCommon::enableDLA(builder.get(), config.get(), config_.dlaCore);

    return true;
}

bool CModule_yolov5_tensorrt_impl::isSupported(DataType dataType)
{
    auto builder = TRTUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8())
    || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }

    return true;
}

