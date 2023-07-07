//
// Created by jnulzl on 2020/6/20.
//
#include <string.h>
#include <vector>
#include <algorithm>

#include "post_process.h"

std::vector<int> make_grid(int nx, int ny) {
    std::vector<int> data;
    long num = 2 * nx * ny;
    data.resize(num);
    for (int  idy = 0; idy < ny; idy++)
    {
        for (int idx = 0; idx < nx; idx++)
        {
            long index = 2 * idx + 2 * nx * idy;
            data[index] = idx;
            data[index + 1] = idy;
        }
    }
    return data;
}

void decode_net_output(float* src, int src_n, int src_c, int src_h, int src_w, 
                    int stride, const float* anchor_grid) {
    int chw = src_c * src_h * src_w;
    int ch = src_c * src_h;
    std::vector<int> grid = make_grid(src_h, src_c);
    for (long bs = 0; bs < src_n; bs++)
    {
        for (long index = 0; index < ch; index++)
        {
            long step = bs * chw + index * src_w;
            float x = src[step + 0];
            float y = src[step + 1];
            float w = src[step + 2];
            float h = src[step + 3];
            // xy
            src[step + 0] = (x * 2 - 0.5 + grid[index * 2 + 0]) * stride;  // x
            src[step + 1] = (y * 2 - 0.5 + grid[index * 2 + 1]) * stride;  // y
            // wh
            src[step + 2] = w * 2 * w * 2 * anchor_grid[2 * bs + 0];  // w
            src[step + 3] = h * 2 * h * 2 * anchor_grid[2 * bs + 1];  // h
        }
    }
}

int cmpfunc (const void * a, const void * b)
{
    BoxInfo* boxa = (BoxInfo*)a;
    BoxInfo* boxb = (BoxInfo*)b;
    return (boxa->score - boxb->score)  > 0 ? 0 : 1;
}

void nms(BoxInfo* boxes, int* delete_obj_index, int num_box, int num_cls, float nms_thresh,
         int* keep_indexs, int* num_keep)
{
    qsort(boxes, num_box, sizeof(BoxInfo), cmpfunc);
    memset(delete_obj_index, 0, sizeof(int) * num_box);

    for (int cls = 0; cls < num_cls; ++cls)
    {
        for (int idx = 0; idx < num_box; ++idx)
        {
            if(delete_obj_index[idx] || cls != boxes[idx].label)
            {
                continue;
            }
            for (int idy = idx + 1; idy < num_box; ++idy)
            {
                if(delete_obj_index[idy] || cls != boxes[idy].label)
                {
                    continue;
                }
                float xx1 = std::max(boxes[idx].x1, boxes[idy].x1);
                float yy1 = std::max(boxes[idx].y1, boxes[idy].y1);
                float xx2 = std::max(boxes[idx].x2, boxes[idy].x2);
                float yy2 = std::max(boxes[idx].y2, boxes[idy].y2);
                float w = std::max(0.0f, xx2 - xx1 + 1);
                float h = std::max(0.0f, yy2 - yy1 + 1);
                float inter = w * h;
                float iou = inter / (boxes[idx].area + boxes[idy].area - inter);
                if(iou > nms_thresh)
                {
                    delete_obj_index[idy] = 1;
                }
            }
        }
    }

    *num_keep = 0;
    for (int i = 0; i < num_box; ++i)
    {
        if(delete_obj_index[i])
        {
            continue;
        }
        keep_indexs[(*num_keep)++] = i;
        if(*num_keep >= MAX_DET_NUM)
            break;
    }
}

void non_max_suppression(float* src, int src_height, int src_width, float conf_thres, float nms_thresh,
                         int net_input_height, int net_input_width, int img_height, int img_width,
                         BoxInfo* dets, int* keep_indexs, int* num_keep)
{
    float gain = 1.0f * std::max(net_input_height, net_input_width) / std::max(img_height, img_width);

    int num_det = 0;
    for (long idx = 0; idx < src_height; idx++)
    {
        const int step = idx * src_width;
        /* cx, cy, width, height, obj_conf, cls_conf0, cls_conf1, ..., cls_conf{n} */
        float obj_conf = src[step + 4];
        if (obj_conf < conf_thres)
            continue;
        float max_score = -1.0f;
        int   obj_cls = -1;
        for (int  idy = 5; idy < src_width; idy++) {
            float tmp = src[step + idy] * obj_conf;
            if (tmp > max_score) {
                max_score = tmp;
                obj_cls = idy - 5;
            }
        }
        if (max_score < conf_thres)
            continue;
        /* xywh2xyxy */
        float cx       = src[step + 0];
        float cy       = src[step + 1];
        float width_2  = src[step + 2] / 2;
        float height_2 = src[step + 3] / 2;
        dets[num_det].x1 = (cx - width_2) / gain;
        dets[num_det].y1 = (cy - height_2) / gain;
        dets[num_det].x2 = (cx + width_2) / gain;
        dets[num_det].y2 = (cy + height_2) /gain;

        dets[num_det].x1 = dets[num_det].x1 > 0 ? dets[num_det].x1 : 0;
        dets[num_det].y1 = dets[num_det].y1 > 0 ? dets[num_det].y1 : 0;
        dets[num_det].x2 = dets[num_det].x2 < img_width ? dets[num_det].x2 : img_width - 1;
        dets[num_det].y2 = dets[num_det].y2 < img_height ? dets[num_det].y2 : img_height - 1;

        dets[num_det].score = max_score;
        dets[num_det].area = (dets[num_det].x2 - dets[num_det].x1 + 1) * (dets[num_det].y2 - dets[num_det].y1 + 1);
        dets[num_det].label = obj_cls;
        num_det++;
    }

    nms(dets, (int*)(src), num_det, src_width - 5, nms_thresh, keep_indexs, num_keep);
}
