//
// Created by jnulzl on 2020/6/20.
//

#ifndef AL_ALG_PRE_PROCESS_H
#define AL_ALG_PRE_PROCESS_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details             : gray image resize with bilinear interpolation
 * @param src           : gray image data pointer
 * @param src_height    : height of gray image
 * @param src_width     : width of gray image
 * @param src_stride    : stride of gray image
 * @param des           : resized image data pointer
 * @param des_height    : height of resized image
 * @param des_width     : width of resized image
 * @param des_stride    : stride of resized image
 * @param num_channels  : number of src or des image's channels
 * @param keep_scale    : if keep original image scale? 0/1
 */
void ai_utils_resize_with_affine(const uint8_t *src, int src_height, int src_width, int src_stride,
                                 uint8_t *des, int des_height, int des_width, int des_stride, int num_channels,
                                 int keep_scale);

/**
 * @details             : gray image resize with bilinear interpolation
 * @param src           : gray image data pointer
 * @param src_height    : height of gray image
 * @param src_width     : width of gray image
 * @param src_stride    : stride of gray image
 * @param des           : resized image data pointer
 * @param des_height    : height of resized image
 * @param des_width     : width of resized image
 * @param des_stride    : stride of resized image
 * @param num_channels  : number of src or des image's channels
 */
void ai_utils_resize(const uint8_t *src, int src_height, int src_width, int src_stride,
                        uint8_t *des, int des_height, int des_width, int des_stride, int num_channels);

/**
 * @details             : gray image resize with bilinear interpolation and keep original image scale
 * @param src           : gray image data pointer
 * @param src_height    : height of gray image
 * @param src_width     : width of gray image
 * @param src_stride    : stride of gray image
 * @param des           : resized image data pointer
 * @param des_height    : height of resized image
 * @param des_width     : width of resized image
 * @param des_stride    : stride of resized image
 * @param num_channels  : number of src or des image's channels
 */
void ai_utils_resize_keep_scale(const uint8_t *src, int src_height, int src_width, int src_stride,
                    uint8_t *des, int des_height, int des_width, int des_stride, int num_channels);

/**
 * @details                 : (pixel - mean) * scale, BGR -> RGB(or reverse), H x W x C -> C x H x W
 * @param src               : color image data pointer
 * @param src_heights       : height of color image
 * @param src_width         : width of color image
 * @param src_stride        : stride of color image
 * @param means_rgb         : mean vale of each channels(BGR order or RGB order)
 * @param scales_rgb        : scale vale of each channels(BGR order or RGB order)
 * @param des               : preprocessed data pointer
 * @param swap_b_r_channels : if swap B and R channels, default is 1
 * @param to_chw            : if transform to C x H x W order, default is 1
 */
void color_normalize_scale_and_chw(const uint8_t* src, int src_heights, int src_width, int src_stride,
                                   const float* means_rgb, const float* scales_rgb, float* des,
                                   int swap_b_r_channels = 1, int to_chw = 1);

#ifdef __cplusplus
}
#endif


#endif //AL_ALG_PRE_PROCESS_H
