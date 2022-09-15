//
// Created by jnulzl on 2022/9/6.
//
// ref to 1. Image-Processing Specific API Conventions https://docs.nvidia.com/cuda/npp/nppi_conventions_lb.html
// ref to 2. NPP Image Processing: https://docs.nvidia.com/cuda/npp/group__nppi.html

#include <iostream>
#include <opencv2/opencv.hpp>
#include <npp.h>

#define CHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                 \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

// #define NPP_DEBUG
// #define NPP_LONE_DEBUG

/**
image roi:
    00000000000000000000---------------
    00000000000000000000---------------
    ----(x0,y0)*************(x1,y1)----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----***************************----
    ----(x3,y3)************(x2,y2)-----
    00000000000000000000---------------
    00000000000000000000---------------
 */

int main(int argc, const char* argv[])
{
    cv::Mat img;
    std::string img_path;
    int des_size;
    int roi_x, roi_y, roi_width, roi_height;
    bool isFixResize;
    bool isSymmetryPad;
    if(argc < 3 || argc > 9)
    {
        std::cout << "Usage:\n\t " << argv[0] << " img_path des_size [isFixResize isSymmetryPad [roi_x roi_y roi_w roi_h]]" << std::endl;
        return -1;
    }
    else if (3 == argc)
    {
        img_path = argv[1];
        img = cv::imread(img_path); // img is bgr
        des_size = std::atoi(argv[2]);
        isFixResize = true;
        isSymmetryPad = false;
        roi_x = 0;
        roi_y = 0;
        roi_width = img.cols;
        roi_height = img.rows;
    }
    else if (4 == argc)
    {
        img_path = argv[1];
        img = cv::imread(img_path); // img is bgr
        des_size = std::atoi(argv[2]);
        isFixResize = std::atoi(argv[3]);
        isSymmetryPad = false;
        roi_x = 0;
        roi_y = 0;
        roi_width = img.cols;
        roi_height = img.rows;
    }
    else if (5 == argc)
    {
        img_path = argv[1];
        img = cv::imread(img_path); // img is bgr
        des_size = std::atoi(argv[2]);
        isFixResize = 0 == std::atoi(argv[3]) ? false : true;
        isSymmetryPad = 0 == std::atoi(argv[4]) ? false : true;
        roi_x = 0;
        roi_y = 0;
        roi_width = img.cols;
        roi_height = img.rows;
    }
    else if (9 == argc)
    {
        img_path = argv[1];
        img = cv::imread(img_path); // img is bgr
        des_size = std::atoi(argv[2]);
        isFixResize = 0 == std::atoi(argv[3]) ? false : true;
        isSymmetryPad = 0 == std::atoi(argv[4]) ? false : true;
        roi_x = std::atoi(argv[5]);
        roi_y = std::atoi(argv[6]);
        roi_width = std::atoi(argv[7]);
        roi_height = std::atoi(argv[8]);
    }
    else
    {
        std::cout << "Usage:\n\t " << argv[0] << " img_path des_size [isFixResize isSymmetryPad [roi_x roi_y roi_w roi_h]]" << std::endl;
        return -1;
    }

    int src_channels = img.channels();
    int src_width = img.cols;
    int src_height = img.rows;

    int des_width = des_size;
    int des_height = des_size;

    
    /************Device memory allocator and initialization***********/
    size_t src_pixel_num = src_height * src_width * src_channels;
    Npp8u* src_ptr_d;
    CHECK(cudaMalloc(&src_ptr_d, sizeof(Npp8u) * (src_pixel_num + 64)));
    CHECK(cudaMemcpy(src_ptr_d, img.data, sizeof(Npp8u) * src_pixel_num, cudaMemcpyHostToDevice));

    size_t dst_pixel_num = des_height * des_width * src_channels;
    Npp8u* dst_ptr_d;
    CHECK(cudaMalloc(&dst_ptr_d, sizeof(Npp8u) * (dst_pixel_num + 64)));

    Npp32f* dst_float_ptr_d;
    CHECK(cudaMalloc(&dst_float_ptr_d, sizeof(Npp32f) * (dst_pixel_num + 64)));

    Npp32f * dst_chw_float_ptr_d;
    CHECK(cudaMalloc(&dst_chw_float_ptr_d, sizeof(Npp32f) * (dst_pixel_num + 64)));

    /**********************getAffineTransform*************************/
    NppiRect oSrcROI = {.x=roi_x, .y=roi_y, .width=roi_width, .height=roi_height};
    double aQuad[4][2] = {{0, 0.0}, {1.0 * des_width, 0.0},
                          {1.0 * des_width, 1.0 * des_height},
                          {0.0, 1.0 * des_height}};

    if(isFixResize)
    {
        float roi_width = src_width;
        float roi_height = src_height;

        float scale_wh = 1.0 * std::fmax(1.0 * des_height, 1.0 * des_width) /
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
        if(isSymmetryPad)
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
            x = (des_width - roi_new_width) / 2.0f;
            y = (des_height - roi_new_height) / 2.0f;
        }
        aQuad[0][0] = x, aQuad[0][1] = y;
        aQuad[1][0] = x + roi_new_width, aQuad[1][1] = y;
        aQuad[2][0] = x + roi_new_width, aQuad[2][1] = y + roi_new_height;
        aQuad[3][0] = x, aQuad[3][1] = y + roi_new_height;
    }
#ifdef NPP_DEBUG
    std::cout << "Src pooints is :" << std::endl;
    std::cout << oSrcROI.x                 << " " << oSrcROI.y                  << std::endl;
    std::cout << oSrcROI.x + oSrcROI.width << " " << oSrcROI.y                  << std::endl;
    std::cout << oSrcROI.x + oSrcROI.width << " " << oSrcROI.y + oSrcROI.height << std::endl;
    std::cout << oSrcROI.x                 << " " << oSrcROI.y + oSrcROI.height << std::endl;

    std::cout << "Dst pooints is :" << std::endl;
    std::cout << aQuad[0][0] << " " << aQuad[0][1] << std::endl;
    std::cout << aQuad[1][0] << " " << aQuad[1][1] << std::endl;
    std::cout << aQuad[2][0] << " " << aQuad[2][1] << std::endl;
    std::cout << aQuad[3][0] << " " << aQuad[3][1] << std::endl;
#endif

    double aCoeffs[2][3];
    // nppiGetAffineTransform(NppiRect oSrcROI, const double aQuad[4][2], double aCoeffs[2][3]);
    nppiGetAffineTransform(oSrcROI, aQuad, aCoeffs);
#ifdef NPP_DEBUG
    std::cout << "Transform matrix is :" << std::endl;
    std::cout << aCoeffs[0][0] << " " << aCoeffs[0][1] << " " << aCoeffs[0][2] << std::endl;
    std::cout << aCoeffs[1][0] << " " << aCoeffs[1][1] << " " << aCoeffs[1][2] << std::endl;
#endif

    /**********************warpAffine**********************/
    // nppiWarpAffine_8u_C3R(const Npp8u * pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
    //                            Npp8u * pDst, int nDstStep, NppiRect oDstROI,
    //                      const double aCoeffs[2][3], int eInterpolation)
    nppiWarpAffine_8u_C3R(src_ptr_d, {src_width, src_height},
                          sizeof(Npp8u) * src_width * src_channels,
                          {0, 0, src_width, src_height},
                          dst_ptr_d, sizeof(Npp8u) * des_width * src_channels,
                          {0, 0, des_width, des_height},
                          aCoeffs,
                          NPPI_INTER_LINEAR);
#ifdef NPP_DEBUG
    std::vector<uint8_t> img_after_warpAffine_data;
    img_after_warpAffine_data.resize(dst_pixel_num);
    CHECK(cudaMemcpy(img_after_warpAffine_data.data(), dst_ptr_d,
                     sizeof(Npp8u) * dst_pixel_num, cudaMemcpyDeviceToHost));
    cv::Mat img_after_warpAffine = cv::Mat(des_height, des_height, CV_8UC3, img_after_warpAffine_data.data());
    cv::imwrite("img_after_warpAffine.png", img_after_warpAffine);
#endif

    /**********************bgr2rgb*************************/
    // nppiSwapChannels_8u_C3IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const int aDstOrder[3])
    const int aDstOrder[3] = {2, 1, 0};
    nppiSwapChannels_8u_C3IR(dst_ptr_d,
                             sizeof(Npp8u) * des_width * src_channels,
                             {des_width, des_height},
                             aDstOrder);
#ifdef NPP_DEBUG
    std::vector<uint8_t> img_after_bgr2rgb_data;
    img_after_bgr2rgb_data.resize(dst_pixel_num);
    CHECK(cudaMemcpy(img_after_bgr2rgb_data.data(), dst_ptr_d,
                     sizeof(Npp8u) * dst_pixel_num, cudaMemcpyDeviceToHost));
    cv::Mat img_after_bgr2rgb = cv::Mat(des_height, des_height, CV_8UC3, img_after_bgr2rgb_data.data());
    cv::imwrite("img_after_bgr2rgb.png", img_after_bgr2rgb);
#endif

    /********************uint8 -> float********************/
    // nppiConvert_8u32f_C3R(const Npp8u  * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI)
    nppiConvert_8u32f_C3R(dst_ptr_d, sizeof(Npp8u) * des_width * src_channels,
                          dst_float_ptr_d,
                          sizeof(Npp32f) * des_width * src_channels,
                          {des_width, des_height}
                          );

    /*********************(x - a) / b**********************/
    /*1.-------- y = (x - a) --------*/
    // nppiSubC_32f_C3IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI)
    const Npp32f means[3] = {0.0f, 0.0f, 0.0f};
    nppiSubC_32f_C3IR(means, dst_float_ptr_d, sizeof(Npp32f) * des_width * src_channels, {des_width, des_height});

    /*2.---------- y * s ----------*/
    // nppiMulC_32f_C3IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI)
    const Npp32f scales[3] = {0.00392157f, 0.00392157f, 0.00392157f};
    nppiMulC_32f_C3IR(scales, dst_float_ptr_d, sizeof(Npp32f) * des_width * src_channels, {des_width, des_height});

//    /*2.---------- y / b ----------*/
//    // nppiDivC_32f_C3IR(const Npp32f  aConstants[3], Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI)
//    const Npp32f scales[3] = {255.0f, 255.0f, 255.0f};
//    nppiDivC_32f_C3IR(scales, dst_float_ptr_d, sizeof(Npp32f) * des_width * src_channels, {des_width, des_height});

    /**********************hwc2chw*************************/
    // nppiCopy_8u_C3P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * const aDst[3], int nDstStep, NppiSize oSizeROI)
    Npp32f * const aDst[3] = {dst_chw_float_ptr_d,
                              dst_chw_float_ptr_d + des_width * des_height,
                              dst_chw_float_ptr_d + 2 * des_width * des_height};
    nppiCopy_32f_C3P3R(dst_float_ptr_d,
                       sizeof(Npp32f) * des_width * src_channels,
                       aDst,
                       sizeof(Npp32f) * des_width,
                       {des_width, des_height});
#ifdef NPP_LONE_DEBUG
    std::vector<float> img_after_bgr2rgb_float_normalize_chw_data;
    img_after_bgr2rgb_float_normalize_chw_data.resize(src_channels * des_width * des_height);
    CHECK(cudaMemcpy(img_after_bgr2rgb_float_normalize_chw_data.data(), dst_chw_float_ptr_d,
                     sizeof(Npp32f) * img_after_bgr2rgb_float_normalize_chw_data.size(), cudaMemcpyDeviceToHost));
    std::cout << "img_chw_float_data : " << std::endl;
    for (int idx = 0; idx < img_after_bgr2rgb_float_normalize_chw_data.size(); ++idx)
    {
        std::printf("%.5f\n", img_after_bgr2rgb_float_normalize_chw_data[idx]);
    }
#endif

    CHECK(cudaFree(src_ptr_d));
    CHECK(cudaFree(dst_ptr_d));
    CHECK(cudaFree(dst_float_ptr_d));
    CHECK(cudaFree(dst_chw_float_ptr_d));
    return 0;
}