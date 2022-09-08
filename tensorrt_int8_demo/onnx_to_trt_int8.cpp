//! Created by jnulzl on 2022/9/6.
//!
//! onnx_to_txt_int8.cpp
//! This file contains the implementation of the sample. It creates the network using
//! the onnx model.

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

#include "json.hpp"
#include "OnnxBatchStream.hpp"
#include "npp_image_preprocess.h"

int main(int argc, char** argv)
{
    if(2 != argc)
    {
        std::cout << "Usage:\n\t " <<  argv[0] << " quant_int8_conifg.json" << std::endl;
        return -1;
    }
    if(!is_support_int8())
    {
        return 0;
    }

    std::string quant_int8_conifg = argv[1];
    std::ifstream input_file(quant_int8_conifg);
    nlohmann::json json_file;
    input_file >> json_file;

    SampleINT8Params params;
    params.onnxFilePath = json_file["onnxFilePath"];
    std::string save_prefix = params.onnxFilePath.substr(0,params.onnxFilePath.size() - 5) + std::string("_int8.trt");
    params.engine_file_save_path = json_file.contains("engine_file_save_path") ?
                                   json_file["engine_file_save_path"] : save_prefix.c_str();
    params.preprocessed_binary_file_path = json_file.contains("preprocessed_binary_file_path") ? json_file["preprocessed_binary_file_path"]:"";
    params.img_list_file = json_file.contains("img_list_file") ? json_file["img_list_file"]:"";
    params.batchSize = json_file.contains("maxInferenceBatchSize") ? static_cast<int>(json_file["maxInferenceBatchSize"]) : 4;
    params.imageChannels = json_file.contains("netInputChannels") ? static_cast<int>(json_file["netInputChannels"]) : 3;
    params.imageHeight = static_cast<int>(json_file["netInputHeight"]);
    params.imageWidth = static_cast<int>(json_file["netInputWidth"]);
    params.calBatchSize = json_file.contains("calBatchSize") ? static_cast<int>(json_file["calBatchSize"]) : 32;
    params.numImages = get_image_num(params);
    params.nbCalBatches = params.numImages / params.calBatchSize;
    auto inputs = json_file["input"].get<std::vector<std::string>>();
    for (const auto& item:inputs)
    {
        params.inputTensorNames.emplace_back(item);
    }
    auto outputs = json_file["output"].get<std::vector<std::string>>();
    for (const auto& item:outputs)
    {
        params.outputTensorNames.emplace_back(item);
    }
    params.networkName = json_file.contains("networkName") ? json_file["networkName"]:"model_int8";
    std::vector<float> means_tmp = json_file["means"].get<std::vector<float>>();
    for (int idx = 0; idx < means_tmp.size(); ++idx)
    {
        params.means[idx] = means_tmp[idx];
    }
    std::vector<float> scales_tmp = json_file["scales"].get<std::vector<float>>();
    for (int idx = 0; idx < scales_tmp.size(); ++idx)
    {
        params.scales[idx] = scales_tmp[idx];
    }
    params.isFixResize   = 0 != static_cast<int>(json_file["isFixResize"]);
    params.isSymmetryPad = 0 != static_cast<int>(json_file["isSymmetryPad"]);
    params.isBGR2RGB     = 0 != static_cast<int>(json_file["isBGR2RGB"]);
    params.isHWC2CHW     = 0 != static_cast<int>(json_file["isHWC2CHW"]);
    params.dlaCore = json_file.contains("dlaCore") ? static_cast<int>(json_file["dlaCore"]) : -1;;
    params.is_save_jpg_after_bgr2rgb     = 0 != static_cast<int>(json_file["is_save_jpg_after_bgr2rgb"]);

    SampleINT8 sample(params);
    auto sampleTest = sample::gLogger.defineTest("SampleInt8Demo", argc, argv);
    sample::gLogger.reportTestStart(sampleTest);
    sample::gLogInfo << "Building and running a GPU inference engine for INT8 sample" << std::endl;

    std::vector<std::string> dataTypeNames = {"INT8"};
    std::vector<DataType> dataTypes = {DataType::kINT8};
    for (size_t i = 0; i < dataTypes.size(); i++)
    {
        if (!sample.build(dataTypes[i]))
        {
            if (!samplesCommon::isDataTypeSupported(dataTypes[i]))
            {
                sample::gLogWarning << "Skipping " << dataTypeNames[i]
                                    << " since the platform does not support this data type." << std::endl;
                continue;
            }
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    return sample::gLogger.reportPass(sampleTest);
}
