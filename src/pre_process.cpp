//
// Created by jnulzl on 2020/6/20.
//
#include <stdio.h>
#include <vector>
#include <cmath>
#include <string.h>
#include "pre_process.h"

namespace AI_UTILS {

    void resize_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride,
                            unsigned char* dst, int w, int h, int stride);

    void resize_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride,
                            unsigned char* dst, int w, int h, int stride);

    void resize_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride,
                            unsigned char* dst, int w, int h, int stride);

    void resize_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride,
                            unsigned char* dst, int w, int h, int stride);

    void resize_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch,
                                  unsigned char* dst, int w, int h);

    void getAffineTransform(const float* src, const float* dst, int num_point, float* trans)
    {
        /**
         * [
         *  trans_m[0].x trans_m[1].x trans_m[2].x
         *  trans_m[0].y trans_m[1].y trans_m[2].y
         * ]
         *
         */
        float x0 = src[0], y0 = src[1], x1 = src[2], y1 = src[3], x2 = src[4], y2 = src[5];
        float u0 = dst[0], v0 = dst[1], u1 = dst[2], v1 = dst[3], u2 = dst[4], v2 = dst[5];

        float b  = x0 * (y1 - y2) - x1 * (y0 - y2) + x2 * (y0 - y1);
        trans[0] = (u0 * (y1 - y2) - u1 * (y0 - y2) + u2 * (y0 - y1)) / b;  //x1
        trans[1] = -(u0 * (x1 - x2) - u1 * (x0 - x2) + u2 * (x0 - x1)) / b; // x2
        trans[2] = (u0 * (x1 * y2 - x2 * y1) -u1 * (x0 * y2 - x2 * y0) + u2 * (x0 * y1 - x1 * y0)) / b; // x3

        trans[3] = (v0 * (y1 - y2) - v1 * (y0 - y2) + v2 * (y0 - y1)) / b;  //y1
        trans[4] = -(v0 * (x1 - x2) - v1 * (x0 - x2) + v2 * (x0 - x1)) / b; //y2
        trans[5] = (v0 * (x1 * y2 - x2 * y1) - v1 * (x0 * y2 - x2 * y0) + v2 * (x0 * y1 - x1 * y0)) / b; // y3

        return;
    }

    void get_affine_transform(const float* points_from, const float* points_to, int num_point, float* tm);

    void invert_affine_transform(const float* tm, float* tm_inv);

    void warpaffine_bilinear_c1(const unsigned char* src, int srcw, int srch, int srcstride,
                                unsigned char* dst, int w, int h, int stride,
                                const float* tm, int type, unsigned int v);

    void warpaffine_bilinear_c2(const unsigned char* src, int srcw, int srch, int srcstride,
                                unsigned char* dst, int w, int h, int stride,
                                const float* tm, int type, unsigned int v);

    void warpaffine_bilinear_c3(const unsigned char* src, int srcw, int srch, int srcstride,
                                unsigned char* dst, int w, int h, int stride,
                                const float* tm, int type, unsigned int v);

    void warpaffine_bilinear_c4(const unsigned char* src, int srcw, int srch, int srcstride,
                                unsigned char* dst, int w, int h, int stride,
                                const float* tm, int type, unsigned int v);

    void warpaffine_bilinear_yuv420sp(const unsigned char* src, int srcw, int srch, unsigned char* dst, int w, int h,
                                      const float* tm, int type, unsigned int v);
}

void ai_utils_resize_c1_keep_scale(const uint8_t *src, int src_height, int src_width, int src_stride,
                    uint8_t *des, int des_height, int des_width, int des_stride)
{
    if (src_height == src_width) {
        AI_UTILS::resize_bilinear_c1(src, src_width, src_height, src_stride,
                                 des, des_width, des_height, des_stride);
        return;
    }

    int num_channels = 1;
    float scale_wh = 1.0 * std::fmax(1.0 * des_height, 1.0 * des_width) /
                     std::fmax(1.0 * src_height, 1.0 * src_width);
    int src_new_height = scale_wh * src_height;
    int src_new_width = scale_wh * src_width;

    int src_resize_tmpsize = num_channels * src_new_width * src_new_height;
    std::vector<uint8_t> src_resize_tmp;
    if(src_resize_tmp.size() < src_resize_tmpsize)
    {
        src_resize_tmp.resize(src_resize_tmpsize + 64);
    }

    AI_UTILS::resize_bilinear_c1(src, src_width, src_height, src_stride,
                             src_resize_tmp.data(), src_new_width, src_new_height, num_channels * src_new_width);

//    size_t left = (des_width - src_new_width) / 2;
//    size_t top = (des_height - src_new_height) / 2;
    size_t left = 0;
    size_t top = 0;
    if(src_new_width < des_width)
    {
        for (int idx = 0; idx < des_height; ++idx)
        {
            memcpy(des + idx * des_stride + left * num_channels,
                   src_resize_tmp.data() + idx * num_channels * src_new_width,
                   sizeof(uint8_t) * num_channels * src_new_width);
        }
    }
    else
    {
        memcpy(des + top * des_stride + left * num_channels, src_resize_tmp.data(),
               sizeof(uint8_t) * src_new_width * src_new_height * num_channels);
    }

    return;
}


void ai_utils_resize_c3_keep_scale(const uint8_t *src, int src_height, int src_width, int src_stride,
                    uint8_t *des, int des_height, int des_width, int des_stride)
{
    if (src_height == src_width) {
        AI_UTILS::resize_bilinear_c3(src, src_width, src_height, src_stride,
                                     des, des_width, des_height, des_stride);
        return;
    }

    int num_channels = 3;
    float scale_wh = 1.0 * std::fmax(1.0 * des_height, 1.0 * des_width) /
                     std::fmax(1.0 * src_height, 1.0 * src_width);
    int src_new_height = scale_wh * src_height;
    int src_new_width = scale_wh * src_width;

    int src_resize_tmpsize = num_channels * src_new_width * src_new_height;
    std::vector<uint8_t> src_resize_tmp;
    if(src_resize_tmp.size() < src_resize_tmpsize)
    {
        src_resize_tmp.resize(src_resize_tmpsize + 64);
    }

    AI_UTILS::resize_bilinear_c3(src, src_width, src_height, src_stride,
                                 src_resize_tmp.data(), src_new_width, src_new_height, num_channels * src_new_width);

//    size_t left = (des_width - src_new_width) / 2;
//    size_t top = (des_height - src_new_height) / 2;
    size_t left = 0;
    size_t top = 0;
    if(src_new_width < des_width)
    {
        for (int idx = 0; idx < des_height; ++idx)
        {
            memcpy(des + idx * des_stride + left * num_channels,
                   src_resize_tmp.data() + idx * num_channels * src_new_width,
                   sizeof(uint8_t) * num_channels * src_new_width);
        }
    }
    else
    {
        memcpy(des + top * des_stride + left * num_channels, src_resize_tmp.data(),
               sizeof(uint8_t) * src_new_width * src_new_height * num_channels);
    }

    return;
}


void ai_utils_resize(const uint8_t *src, int src_height, int src_width, int src_stride,
                     uint8_t *des, int des_height, int des_width, int des_stride, int num_channels)
{
    if(1 == num_channels)
    {
        AI_UTILS::resize_bilinear_c1(src, src_width, src_height, src_stride,
                                     des, des_width, des_height, des_stride);
    }
    else if(3 == num_channels)
    {
        AI_UTILS::resize_bilinear_c3(src, src_width, src_height, src_stride,
                                     des, des_width, des_height, des_stride);
    }
    return;
}

void ai_utils_resize_keep_scale(const uint8_t *src, int src_height, int src_width, int src_stride,
                     uint8_t *des, int des_height, int des_width, int des_stride, int num_channels)
{
    if(1 == num_channels)
    {
        ai_utils_resize_c1_keep_scale(src, src_height, src_width, src_stride,
                                      des, des_height, des_width, des_stride);
    }
    else if(3 == num_channels)
    {
        ai_utils_resize_c3_keep_scale(src, src_height, src_width, src_stride,
                                      des, des_height, des_width, des_stride);
    }
    return;
}

void ai_utils_resize_with_affine(const uint8_t *src, int src_height, int src_width, int src_stride,
                     uint8_t *des, int des_height, int des_width, int des_stride, int num_channels,
                     int is_resize_with_pad)
{
    float src_points[6], des_points[6];
    src_points[0] = 0;
    src_points[1] = 0;
    src_points[2] = src_width;
    src_points[3] = 0;
    src_points[4] = src_width;
    src_points[5] = src_height;

    des_points[0] = 0;
    des_points[1] = 0;
    des_points[2] = des_width;
    des_points[3] = 0;
    des_points[4] = des_width;
    des_points[5] = des_height;

    if(is_resize_with_pad)
    {
        float roi_width = src_width;
        float roi_height = src_height;

        float scale_wh = 1.0 * std::fmax(1.0 * des_height, 1.0 * des_width) /
                         std::fmax(1.0 * src_height, 1.0 * src_width);
        float roi_new_width = roi_width * scale_wh;
        float roi_new_height = roi_height * scale_wh;

//        float x = (des_width - roi_new_width) / 2.0f;
//        float y = (des_height - roi_new_height) / 2.0f;
        float x = 0;
        float y = 0;

        des_points[0] = x;
        des_points[1] = y;
        des_points[2] = x + roi_new_width;
        des_points[3] = y;
        des_points[4] = x + roi_new_width;
        des_points[5] = y + roi_new_height;
    }

    float tm[6] = {0.0f};
    AI_UTILS::getAffineTransform(des_points, src_points, 3, tm);
    //AI_UTILS::get_affine_transform(des_points, src_points, 3, tm);

    if(1 == num_channels)
    {
        AI_UTILS::warpaffine_bilinear_c1(src, src_width, src_height, src_stride,
                                     des, des_width, des_height, des_stride, tm, 0, 0);
    }
    else if(3 == num_channels)
    {
        AI_UTILS::warpaffine_bilinear_c3(src, src_width, src_height, src_stride,
                                     des, des_width, des_height, des_stride, tm, 0, 0);
    }
    return;
}

void color_normalize_scale_and_chw(const uint8_t* src, int src_heights, int src_width, int src_stride,
                                    const float* means_rgb, const float* scales_rgb, float* des,
                                   int swap_b_r_channels, int to_chw)
{
    if(swap_b_r_channels)
    {
        if(to_chw)
        {
            // swap bule and red channels and hwc -> chw
            int des_stride = src_heights * src_width;
            for (int idx = 0; idx < src_heights; ++idx)
            {
                for (int idy = 0; idy < src_width; ++idy)
                {
                    float r = (1.0f * src[idx * src_stride + 3 * idy + 2] - means_rgb[0]) * scales_rgb[0];
                    float g = (1.0f * src[idx * src_stride + 3 * idy + 1] - means_rgb[1]) * scales_rgb[1];
                    float b = (1.0f * src[idx * src_stride + 3 * idy + 0] - means_rgb[2]) * scales_rgb[2];
                    des[idx * src_width + idy] = r;
                    des[idx * src_width + idy + des_stride] = g;
                    des[idx * src_width + idy + des_stride * 2] = b;
                }
            }
        }
        else
        {
            // only swap bule and red channels
            for (int idx = 0; idx < src_heights; ++idx)
            {
                for (int idy = 0; idy < src_width; ++idy)
                {
                    float r = (1.0f * src[idx * src_stride + 3 * idy + 2] - means_rgb[0]) * scales_rgb[0];
                    float g = (1.0f * src[idx * src_stride + 3 * idy + 1] - means_rgb[1]) * scales_rgb[1];
                    float b = (1.0f * src[idx * src_stride + 3 * idy + 0] - means_rgb[2]) * scales_rgb[2];
                    des[idx * src_stride + 3 * idy + 0] = r;
                    des[idx * src_stride + 3 * idy + 1] = g;
                    des[idx * src_stride + 3 * idy + 2] = b;
                }
            }
        }
    }
    else
    {
        if(to_chw)
        {
            // only hwc -> chw
            int des_stride = src_heights * src_width;
            for (int idx = 0; idx < src_heights; ++idx)
            {
                for (int idy = 0; idy < src_width; ++idy)
                {
                    float r = (1.0f * src[idx * src_stride + 3 * idy + 0] - means_rgb[0]) * scales_rgb[0];
                    float g = (1.0f * src[idx * src_stride + 3 * idy + 1] - means_rgb[1]) * scales_rgb[1];
                    float b = (1.0f * src[idx * src_stride + 3 * idy + 2] - means_rgb[2]) * scales_rgb[2];
                    des[idx * src_width + idy] = r;
                    des[idx * src_width + idy + des_stride] = g;
                    des[idx * src_width + idy + des_stride * 2] = b;
                }
            }
        }
        else
        {
            for (int idx = 0; idx < src_heights; ++idx)
            {
                for (int idy = 0; idy < src_width; ++idy)
                {
                    float r = (1.0f * src[idx * src_stride + 3 * idy + 0] - means_rgb[0]) * scales_rgb[0];
                    float g = (1.0f * src[idx * src_stride + 3 * idy + 1] - means_rgb[1]) * scales_rgb[1];
                    float b = (1.0f * src[idx * src_stride + 3 * idy + 2] - means_rgb[2]) * scales_rgb[2];
                    des[idx * src_stride + 3 * idy + 0] = r;
                    des[idx * src_stride + 3 * idy + 1] = g;
                    des[idx * src_stride + 3 * idy + 2] = b;
                }
            }
        }
    }
}

