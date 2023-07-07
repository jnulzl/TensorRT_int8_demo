//
// Created by jnulzl on 2020/6/20.
//

#ifndef YOLOV5_POST_PROCESS_H
#define YOLOV5_POST_PROCESS_H

#include <cstdint>
#include "data_type.h"

#define MAX_DET_NUM 100

/**
 * @details           : decode yolo net output to objs
 *
 * @param src         : net output feature map whose shape eg : 3 x 40 x 40 x 6
 * @param src_n       : feature map src's batch, eg : 3
 * @param src_c       : feature map src's channels, eg : 40
 * @param src_h       : feature map src's height, eg : 40
 * @param src_w       : feature map src's width, eg : 6
 * @param stride      : net stride, eg : 8
 * @param anchor_grid : yolo anchors, eg : [10, 13, 16, 30, 33, 23]
 */
void decode_net_output(float* src, int src_n, int src_c, int src_h, int src_w,
                        int stride, const float* anchor_grid);


void non_max_suppression(float* src, int src_height, int src_width, float conf_thres, float nms_thresh,
                         int net_input_height, int net_input_width, int img_height, int img_width,
                         BoxInfo* dets, int* keep_indexs, int* num_keep);
#endif //YOLOV5_POST_PROCESS_H
