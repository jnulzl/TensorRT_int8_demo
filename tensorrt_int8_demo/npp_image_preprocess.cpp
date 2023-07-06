//
// Created by jnulzl on 2022/9/8.
//

#include <fstream>

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

#include "json.hpp"
#include "npp.h"
#include "npp_image_preprocess.h"

#include "logger.h"
#include "NvInfer.h"
#include "common.h"

#if NV_TENSORRT_MAJOR >= 8
using samplesCommon::SampleUniquePtr;
#else
template <typename T>
using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;
#endif

std::vector<std::string> split(const std::string& string, char separator, bool ignore_empty)
{
    std::vector<std::string> pieces;
    std::stringstream ss(string);
    std::string item;
    while (getline(ss, item, separator))
    {
        if (!ignore_empty || !item.empty())
        {
            pieces.push_back(std::move(item));
        }
    }
    return pieces;
}

std::string trim(const std::string& str) {
    size_t left = str.find_first_not_of(' ');
    if (left == std::string::npos) {
        return str;
    }
    size_t right = str.find_last_not_of(' ');
    return str.substr(left, (right - left + 1));
}

bool is_support_int8()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    if(builder->platformHasFastInt8())
    {
        sample::gLogInfo << "Current environment support int8!" << std::endl;
        return true;
    }
    else
    {
        sample::gLogError << "Current environment can not support int8!" << std::endl;
        return false;
    }
}

bool set_quant_params(const std::string& config_json_path, SampleINT8Params& params)
{
    std::ifstream input_file(config_json_path);
    nlohmann::json json_file;
    input_file >> json_file;

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
    params.dlaCore = json_file.contains("dlaCore") ? static_cast<int>(json_file["dlaCore"]) : -1;
    params.is_save_jpg_after_bgr2rgb     = 0 != static_cast<int>(json_file["is_save_jpg_after_bgr2rgb"]);
    return true;
}
void process_color_image(const SampleINT8Params& params, float* preprocess_data)
{
    size_t src_pixel_num_pre = 0;
    Npp8u *src_ptr_d = nullptr;
    size_t dst_pixel_num = params.imageHeight * params.imageWidth * params.imageChannels;
    Npp8u *dst_ptr_d = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * dst_pixel_num));
    Npp32f *dst_float_ptr_d = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num));
    Npp32f *dst_chw_float_ptr_d = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num));

    std::ifstream input(params.img_list_file);
    std::string line;
    size_t img_count = 0;
    while (true)
    {
        std::getline(input, line);
        if (line.empty())
            break;
        std::string img_path = trim(line);

        /************Device memory allocator and initialization***********/
        cv::Mat img;
        img = cv::imread(line, cv::IMREAD_COLOR); // img is bgr
        int src_width = img.cols;
        int src_height = img.rows;
        int src_channels = 3;

        size_t src_pixel_num = src_height * src_width * src_channels;
        if(src_pixel_num_pre < src_pixel_num)
        {
            cudaFastFree(src_ptr_d);
            src_pixel_num_pre = src_pixel_num;
            src_ptr_d = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * src_pixel_num_pre));
        }
        CUDACHECK(cudaMemcpy(src_ptr_d, img.data, sizeof(Npp8u) * src_pixel_num, cudaMemcpyHostToDevice));

        /**********************getAffineTransform*************************/
        NppiRect oSrcROI = {.x=0, .y=0, .width=src_width, .height=src_height};
        double aQuad[4][2] = {{0,               0.0},
                              {1.0 * params.imageWidth, 0.0},
                              {1.0 * params.imageWidth, 1.0 * params.imageHeight},
                              {0.0,             1.0 * params.imageHeight}};

        if (params.isFixResize)
        {
            float roi_width = src_width;
            float roi_height = src_height;

            float scale_wh = 1.0 * std::fmax(1.0 * params.imageHeight, 1.0 * params.imageWidth) /
                             std::fmax(1.0 * src_height, 1.0 * src_width);
            float roi_new_width = roi_width * scale_wh;
            float roi_new_height = roi_height * scale_wh;

            /**
                roi_height > roi_width                roi_height < roi_width
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                0000000000000000000000
                ****************000000                0000000000000000000000
                ****************000000                0000000000000000000000
             **/
            float x = 0;
            float y = 0;
            if (params.isSymmetryPad)
            {
                /**
                roi_height > roi_width                roi_height < roi_width
                000****************000                00000000000000000000
                000****************000                00000000000000000000
                000****************000                ********************
                000****************000                ********************
                000****************000                ********************
                000****************000                00000000000000000000
                000****************000                00000000000000000000
                 **/
                x = (params.imageWidth - roi_new_width) / 2.0f;
                y = (params.imageHeight - roi_new_height) / 2.0f;
            }
            aQuad[0][0] = x, aQuad[0][1] = y;
            aQuad[1][0] = x + roi_new_width, aQuad[1][1] = y;
            aQuad[2][0] = x + roi_new_width, aQuad[2][1] = y + roi_new_height;
            aQuad[3][0] = x, aQuad[3][1] = y + roi_new_height;
        }

        double aCoeffs[2][3];
        nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);

        /**********************warpAffine**********************/
        nppiWarpAffine_8u_C3R(src_ptr_d, {src_width, src_height},
                              sizeof(Npp8u) * src_width * src_channels,
                              {0, 0, src_width, src_height},
                              dst_ptr_d, sizeof(Npp8u) * params.imageWidth * src_channels,
                              {0, 0, params.imageWidth, params.imageHeight},
                              aCoeffs,
                              NPPI_INTER_LINEAR);

        if(params.is_save_jpg_after_bgr2rgb && img_count < 10)
        {
            std::vector<uint8_t> img_after_warpAffine_data;
            img_after_warpAffine_data.resize(dst_pixel_num);
            CUDACHECK(cudaMemcpy(img_after_warpAffine_data.data(), dst_ptr_d,
                                 sizeof(Npp8u) * dst_pixel_num, cudaMemcpyDeviceToHost));
            cv::Mat img_after_warpAffine = cv::Mat(params.imageHeight, params.imageWidth, CV_8UC3,
                                                   img_after_warpAffine_data.data());
            cv::imwrite("img_after_warpAffine_count" + std::to_string(img_count) + ".jpg", img_after_warpAffine);
        }

        /**********************bgr2rgb*************************/
        if(params.isBGR2RGB)
        {
            const int aDstOrder[3] = {2, 1, 0};
            nppiSwapChannels_8u_C3IR(dst_ptr_d,
                                     sizeof(Npp8u) * params.imageWidth * src_channels,
                                     {params.imageWidth, params.imageHeight},
                                     aDstOrder);

            if(params.is_save_jpg_after_bgr2rgb && img_count < 10)
            {
                std::vector<uint8_t> img_after_bgr2rgb_data;
                img_after_bgr2rgb_data.resize(dst_pixel_num);
                CUDACHECK(cudaMemcpy(img_after_bgr2rgb_data.data(), dst_ptr_d,
                                     sizeof(Npp8u) * dst_pixel_num, cudaMemcpyDeviceToHost));
                cv::Mat img_after_bgr2rgb = cv::Mat(params.imageHeight, params.imageWidth, CV_8UC3, img_after_bgr2rgb_data.data());
                cv::imwrite("img_after_bgr2rgb_count" + std::to_string(img_count) + ".jpg", img_after_bgr2rgb);
            }
        }

        /********************uint8 -> float********************/
        nppiConvert_8u32f_C3R(dst_ptr_d, sizeof(Npp8u) * params.imageWidth * src_channels,
                              dst_float_ptr_d,
                              sizeof(Npp32f) * params.imageWidth * src_channels,
                              {params.imageWidth, params.imageHeight}
        );

        /*********************(x - a) / b**********************/
        /*1.-------- y = (x - a) --------*/
        nppiSubC_32f_C3IR(params.means, dst_float_ptr_d,
                          sizeof(Npp32f) * params.imageWidth * src_channels, {params.imageWidth, params.imageHeight});

        /*2.---------- y * s ----------*/
        nppiMulC_32f_C3IR(params.scales, dst_float_ptr_d,
                          sizeof(Npp32f) * params.imageWidth * src_channels, {params.imageWidth, params.imageHeight});

        /**********************hwc2chw*************************/
        if(params.isHWC2CHW)
        {
            Npp32f *const aDst[3] = {dst_chw_float_ptr_d,
                                     dst_chw_float_ptr_d + params.imageWidth * params.imageHeight,
                                     dst_chw_float_ptr_d + 2 * params.imageWidth * params.imageHeight};
            nppiCopy_32f_C3P3R(dst_float_ptr_d,
                               sizeof(Npp32f) * params.imageWidth * src_channels,
                               aDst,
                               sizeof(Npp32f) * params.imageWidth,
                               {params.imageWidth, params.imageHeight});

            CUDACHECK(cudaMemcpy(preprocess_data + img_count * dst_pixel_num, dst_chw_float_ptr_d,
                                 sizeof(Npp32f) * dst_pixel_num, cudaMemcpyDeviceToHost));
        }
        else
        {
            CUDACHECK(cudaMemcpy(preprocess_data + img_count * dst_pixel_num, dst_float_ptr_d,
                                 sizeof(Npp32f) * dst_pixel_num, cudaMemcpyDeviceToHost));
        }
        ++img_count;
    }

    CUDACHECK(cudaFree(src_ptr_d));
    CUDACHECK(cudaFree(dst_ptr_d));
    CUDACHECK(cudaFree(dst_float_ptr_d));
    CUDACHECK(cudaFree(dst_chw_float_ptr_d));
}

void process_gray_image(const SampleINT8Params& params, float* preprocess_data)
{
    size_t src_pixel_num_pre = 0;
    Npp8u *src_ptr_d = nullptr;
    size_t dst_pixel_num = params.imageHeight * params.imageWidth;
    Npp8u *dst_ptr_d = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * dst_pixel_num));
    Npp32f *dst_float_ptr_d = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num));
    Npp32f *dst_chw_float_ptr_d = reinterpret_cast<Npp32f*>(cudaFastMalloc(sizeof(Npp32f) * dst_pixel_num));

    std::ifstream input(params.img_list_file);
    std::string line;

    size_t img_count = 0;
    while (true)
    {
        std::getline(input, line);
        if (line.empty())
            break;
        std::string img_path = trim(line);

        /************Device memory allocator and initialization***********/
        cv::Mat img;
        img = cv::imread(line, cv::IMREAD_GRAYSCALE); // img is gray
        int src_width = img.cols;
        int src_height = img.rows;
        int src_channels = 1;
        size_t src_pixel_num = src_height * src_width * src_channels;
        if(src_pixel_num_pre < src_pixel_num)
        {
            cudaFastFree(src_ptr_d);
            src_pixel_num_pre = src_pixel_num;
            src_ptr_d = reinterpret_cast<Npp8u*>(cudaFastMalloc(sizeof(Npp8u) * src_pixel_num_pre));
        }
        CUDACHECK(cudaMemcpy(src_ptr_d, img.data, sizeof(Npp8u) * src_pixel_num, cudaMemcpyHostToDevice));


        /**********************getAffineTransform*************************/
        NppiRect oSrcROI = {.x=0, .y=0, .width=src_width, .height=src_height};
        double aQuad[4][2] = {{0,               0.0},
                              {1.0 * params.imageWidth, 0.0},
                              {1.0 * params.imageWidth, 1.0 * params.imageHeight},
                              {0.0,             1.0 * params.imageHeight}};

        if (params.isFixResize)
        {
            float roi_width = src_width;
            float roi_height = src_height;

            float scale_wh = 1.0 * std::fmax(1.0 * params.imageHeight, 1.0 * params.imageWidth) /
                             std::fmax(1.0 * src_height, 1.0 * src_width);
            float roi_new_width = roi_width * scale_wh;
            float roi_new_height = roi_height * scale_wh;

            /**
                roi_height > roi_width                roi_height < roi_width
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                **********************
                ****************000000                0000000000000000000000
                ****************000000                0000000000000000000000
                ****************000000                0000000000000000000000
             **/
            float x = 0;
            float y = 0;
            if (params.isSymmetryPad)
            {
                /**
                roi_height > roi_width                roi_height < roi_width
                000****************000                00000000000000000000
                000****************000                00000000000000000000
                000****************000                ********************
                000****************000                ********************
                000****************000                ********************
                000****************000                00000000000000000000
                000****************000                00000000000000000000
                 **/
                x = (params.imageWidth - roi_new_width) / 2.0f;
                y = (params.imageHeight - roi_new_height) / 2.0f;
            }
            aQuad[0][0] = x, aQuad[0][1] = y;
            aQuad[1][0] = x + roi_new_width, aQuad[1][1] = y;
            aQuad[2][0] = x + roi_new_width, aQuad[2][1] = y + roi_new_height;
            aQuad[3][0] = x, aQuad[3][1] = y + roi_new_height;
        }

        double aCoeffs[2][3];
        nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);

        /**********************warpAffine**********************/
        nppiWarpAffine_8u_C1R(src_ptr_d, {src_width, src_height},
                              sizeof(Npp8u) * src_width * src_channels,
                              {0, 0, src_width, src_height},
                              dst_ptr_d, sizeof(Npp8u) * params.imageWidth * src_channels,
                              {0, 0, params.imageWidth, params.imageHeight},
                              aCoeffs,
                              NPPI_INTER_LINEAR);

        if(params.is_save_jpg_after_bgr2rgb && img_count < 10)
        {
            std::vector<uint8_t> img_after_warpAffine_data;
            img_after_warpAffine_data.resize(dst_pixel_num);
            CUDACHECK(cudaMemcpy(img_after_warpAffine_data.data(), dst_ptr_d,
                                 sizeof(Npp8u) * dst_pixel_num, cudaMemcpyDeviceToHost));
            cv::Mat img_after_warpAffine = cv::Mat(params.imageHeight, params.imageWidth, CV_8UC1,
                                                   img_after_warpAffine_data.data());
            cv::imwrite("img_after_warpAffine_count" + std::to_string(img_count) + ".jpg", img_after_warpAffine);
        }

        /********************uint8 -> float********************/
        nppiConvert_8u32f_C1R(dst_ptr_d, sizeof(Npp8u) * params.imageWidth * src_channels,
                              dst_float_ptr_d,
                              sizeof(Npp32f) * params.imageWidth * src_channels,
                              {params.imageWidth, params.imageHeight});

        /*********************(x - a) / b**********************/
        /*1.-------- y = (x - a) --------*/
        nppiSubC_32f_C1IR(params.means[0], dst_float_ptr_d,
                          sizeof(Npp32f) * params.imageWidth * src_channels, {params.imageWidth, params.imageHeight});

        /*2.---------- y * s ----------*/
        nppiMulC_32f_C1IR(params.scales[0], dst_float_ptr_d,
                          sizeof(Npp32f) * params.imageWidth * src_channels, {params.imageWidth, params.imageHeight});

        CUDACHECK(cudaMemcpy(preprocess_data + img_count * dst_pixel_num, dst_float_ptr_d,
                             sizeof(Npp32f) * dst_pixel_num, cudaMemcpyDeviceToHost));
        ++img_count;
    }

    CUDACHECK(cudaFree(src_ptr_d));
    CUDACHECK(cudaFree(dst_ptr_d));
    CUDACHECK(cudaFree(dst_float_ptr_d));
    CUDACHECK(cudaFree(dst_chw_float_ptr_d));
}

void preprocess_func(const SampleINT8Params& params, std::vector<float>& net_input_preprocessed_data)
{
    std::ifstream input(params.img_list_file);
    std::string line;
    int img_count = 0;
    while (true)
    {
        std::getline(input, line);
        if (line.empty())
            break;
        ++img_count;
    }
    size_t dst_pixel_num = params.imageHeight * params.imageWidth * params.imageChannels;
    net_input_preprocessed_data.resize((img_count + 1) * dst_pixel_num);
    float* preprocess_data = net_input_preprocessed_data.data();
    if(3 == params.imageChannels)
    {
        process_color_image(params, preprocess_data);
    }
    else if(1 == params.imageChannels)
    {
        process_gray_image(params, preprocess_data);
    }
    else
    {
        throw std::invalid_argument("Only supported channel number is 1 or 3");
    }
}

size_t get_image_num(const SampleINT8Params& params)
{
    int img_count = 0;
    if(fs::exists(params.img_list_file))
    {
        std::ifstream input(params.img_list_file);
        std::string line;
        while (true)
        {
            std::getline(input, line);
            if (line.empty())
                break;
            ++img_count;
        }
        return img_count;
    }

    if(fs::exists(params.preprocessed_binary_file_path))
    {
        std::ifstream in_file(params.preprocessed_binary_file_path, std::ios::binary);
        in_file.seekg(0, std::ios::end);
        size_t file_size = in_file.tellg(); // unit is bytes
        file_size /= sizeof(float); // the float number
        return file_size / (params.imageChannels * params.imageHeight * params.imageWidth);
    }
    return 0;
}
