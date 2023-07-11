//
// Created by jnulzl on 2021/5/25.
//

/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ONNX_BATCHSTREAM_H
#define ONNX_BATCHSTREAM_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <assert.h>
#include <stdio.h>
#include <vector>

#include <cuda_runtime_api.h>
#include "BatchStream.h"
#include "EntropyCalibrator.h"
#include "argsParser.h"
#include "buffers.h"
#include "logger.h"
#include "NvInfer.h"
#include "common.h"
#include "NvOnnxParser.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"


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
#include "npp_image_preprocess.h"

class ONNXBatchStream : public IBatchStream
{
public:
    ONNXBatchStream(const SampleINT8Params& params)
            : mBatchSize{params.calBatchSize}
            , mMaxBatches{params.nbCalBatches}
            , mDims{3, params.imageChannels, params.imageHeight, params.imageWidth} //!< We already know the dimensions of MNIST images.
            , mNumImages{params.numImages}
            , mImageC{params.imageChannels}
            , mImageH{params.imageHeight}
            , mImageW{params.imageWidth}
    {
        if(fs::exists(params.img_list_file))
        {
            sample::gLogInfo << "Reading and preprocess calibrator data from " << params.img_list_file << std::endl;
            preprocess_func(params, mData);
        }
        else if(fs::exists(params.preprocessed_binary_file_path))
        {
            sample::gLogInfo << "Reading preprocessed calibrator data from " << params.preprocessed_binary_file_path << std::endl;
            readDataFile(params.preprocessed_binary_file_path);
        }
        else
        {
            throw std::invalid_argument("img_list_file and preprocessed_binary_file_path can not is empty at the same time!");
        }
        sample::gLogInfo << "Reading calibrator data end and the total images is : " << params.numImages << std::endl;
        mLabels.resize(mNumImages, 1.0f);
    }

    void reset(int firstBatch) override
    {
        mBatchCount = firstBatch;
    }

    bool next() override
    {
        if (mBatchCount >= (mMaxBatches - 1))
        {
            return false;
        }
        ++mBatchCount;
        return true;
    }

    void skip(int skipCount) override
    {
        mBatchCount += skipCount;
    }

    float* getBatch() override
    {
        return mData.data() + (mBatchCount * mBatchSize * samplesCommon::volume(mDims));
    }

    float* getLabels() override
    {
        return mLabels.data() + (mBatchCount * mBatchSize);
    }

    int getBatchesRead() const override
    {
        return mBatchCount;
    }

    int getBatchSize() const override
    {
        return mBatchSize;
    }

    nvinfer1::Dims getDims() const override
    {
        return nvinfer1::Dims{4, {mBatchSize, mDims.d[0], mDims.d[1], mDims.d[2]}};
    }

private:
    void readDataFile(const std::string& preprocessed_binary_data_file)
    {
        std::ifstream file{preprocessed_binary_data_file.c_str(), std::ios::binary};
        int numElements = mNumImages * mImageC * mImageH * mImageW;
        mData.resize(numElements);
        file.read(reinterpret_cast<char*>(mData.data()), numElements * sizeof(float));
    }

    int mBatchSize{0};
    int mBatchCount{0}; //!< The batch that will be read on the next invocation of next()
    int mMaxBatches{0};
    nvinfer1::Dims mDims{};
    std::vector<float> mData{};
    std::vector<float> mLabels{};
    int mNumImages;
    int mImageC;
    int mImageH;
    int mImageW;
};

#if NV_TENSORRT_MAJOR >= 8
using samplesCommon::SampleUniquePtr;
#else
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
#endif

//! \brief  The SampleINT8 class implements the INT8 sample
//!
//! \details It creates the network using a onnx model
//!
class SampleINT8
{
public:
    SampleINT8(const SampleINT8Params& params)
            : mParams(params)
            , mEngine(nullptr)
    {
        initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build(nvinfer1::DataType dataType);

private:
    SampleINT8Params mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims; //!< The dimensions of the input to the network.

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses a onnx model and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                          SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                          SampleUniquePtr<nvonnxparser::IParser>& parser, nvinfer1::DataType dataType);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the network by parsing the onnx model and builds
//!          the engine that will be used to run the model (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
bool SampleINT8::build(nvinfer1::DataType dataType)
{

    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8())
        || (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16()))
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser, dataType);
    if (!constructed)
    {
        return false;
    }

    ASSERT(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    ASSERT(mInputDims.nbDims == 4); // input format is nchw

    return true;
}

//!
//! \brief Uses a onnx parser to create the network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the network
//!
//! \param builder Pointer to the engine builder
//!
bool SampleINT8::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
                                  SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
                                  SampleUniquePtr<nvonnxparser::IParser>& parser, nvinfer1::DataType dataType)
{
    mEngine = nullptr;
    auto parsed = parser->parseFromFile(mParams.onnxFilePath.c_str(),
                                        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

//    for (auto& s : mParams.outputTensorNames)
//    {
//        network->markOutput(*blobNameToTensor->find(s.c_str()));
//    }

    // Calibrator life time needs to last until after the engine is built.
    std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;

    config->setAvgTimingIterations(1);
    config->setMinTimingIterations(1);
//    config->setMaxWorkspaceSize(2_GiB);
//    config->setFlag(nvinfer1::BuilderFlag::kVERSION_COMPATIBLE);
    config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    if (dataType == nvinfer1::DataType::kHALF)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (dataType == nvinfer1::DataType::kINT8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
    }
    builder->setMaxBatchSize(mParams.batchSize);

    if (dataType == nvinfer1::DataType::kINT8)
    {
        ONNXBatchStream calibrationStream(mParams);
        calibrator.reset(new Int8EntropyCalibrator2<ONNXBatchStream>(
                calibrationStream, 0, mParams.networkName.c_str(), mParams.inputTensorNames[0].c_str()));
        config->setInt8Calibrator(calibrator.get());
    }

    if (mParams.dlaCore >= 0)
    {
        samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);
        if (mParams.batchSize > builder->getMaxDLABatchSize())
        {
            sample::gLogError << "Requested batch size " << mParams.batchSize
                              << " is greater than the max DLA batch size of " << builder->getMaxDLABatchSize()
                              << ". Reducing batch size accordingly." << std::endl;
            return false;
        }
    }

#if NV_TENSORRT_MAJOR >= 8
    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<nvinfer1::IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return false;
    }

    SampleUniquePtr<nvinfer1::IRuntime> runtime{nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }
#else
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        return false;
    }
#endif
    /************************save engine file to disk***********************/
    std::ofstream engineFile(mParams.engine_file_save_path, std::ios::binary);
    if (!engineFile)
    {
        sample::gLogError << "Cannot open engine file: " << mParams.engine_file_save_path << std::endl;
    }

    SampleUniquePtr<nvinfer1::IHostMemory> serializedEngine{mEngine->serialize()};
    if (!serializedEngine)
    {
        sample::gLogError << "Int8 Engine serialization failed" << std::endl;
    }

    sample::gLogInfo << "Writing TensorRT int8 engine to " << mParams.engine_file_save_path << std::endl;
    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    engineFile.close();
    sample::gLogInfo << "Writing TensorRT int8 engine finished!" <<  std::endl;

    return true;
}


#endif //ONNX_BATCHSTREAM_H
