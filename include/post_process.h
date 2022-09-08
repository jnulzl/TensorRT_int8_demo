//
// Created by jnulzl on 2020/6/20.
//

#ifndef YOLOV5_POST_PROCESS_H
#define YOLOV5_POST_PROCESS_H

#include <cstdint>
#include "data_type.h"
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

    /**
     * @details          : non max suppression for detect results
     *
     * @param src        : detect results from decode_net_output whose shape eg : N x (5 + num_cls)
     * @param src_height : numbers of detect, eg : N
     * @param src_width  : each detect obj length, eg : 6
     *                     (center x, center y, width, height, obj_conf, cls_0_conf, cls_1_conf, ...,  cls_M_conf)
     * @param conf_thres : detect threshold, eg : 0.5
     * @param nms_thresh : NMS threshold, eg : 0.6
     * @param dets       : detect results after non max suppression
     *
     */
    void non_max_suppression(float* src, int src_height, int src_width,
                            float conf_thres, float nms_thresh,
                            std::vector<BoxInfo>& dets);

    /**
     * @details                : convert obj coordinate to origin image scale
     *
     * @param dets             : detect results after non max suppression
     * @param net_input_height : height of net input, eg : 320
     * @param net_input_width  : width of net input, eg : 320
     * @param img_height       : height of origin image, eg : 480
     * @param img_width        : width of origin image, eg : 640
     */
    void postprocess(std::vector<BoxInfo>& dets, float net_input_height, float net_input_width,
                    float img_height, float img_width);

    template <typename Dtype>
    void Permute(const Dtype* bottom_data, const std::vector<int>& bottom_data_shape, const std::vector<int>& permute_order,
                 const int num_axes, Dtype* top_data, std::vector<int>& top_data_shape);

    template <typename Dtype>
    void Sigmod(Dtype* bottom_data, const int bottom_data_num);
#endif //YOLOV5_POST_PROCESS_H
