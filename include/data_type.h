//
// Created by jnulzl on 2020/5/24.
//

#ifndef AI_ALG_DATA_TYPE_H
#define AI_ALG_DATA_TYPE_H

#include <iostream>
#include <cmath>

#define ANY_POINTER_CAST(impl, T) reinterpret_cast<T*>(impl)
typedef void* AW_ANY_POINTER;

struct BaseConfig {
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::string weights_path;
    std::string deploy_path;
    float means[3];
    float scales[3];
    int mean_length;
    int net_inp_channels;
    int net_inp_width;
    int net_inp_height;
    int num_threads = 2;
#ifdef USE_CUDA
    int batch_size = 1;
    int device_id = 0;
#ifdef USE_TENSORRT
    int dlaCore = -1;
        bool fp16 = false;
        bool int8 = false;
#endif
#endif
};

struct SegConfig : public BaseConfig {
    // hrnet or u2net
    int model_type;
};

struct YoloConfig : public BaseConfig {
    int num_cls = 1;
    float conf_thres;
    float nms_thresh;
    std::vector<int> strides;
    std::vector<std::vector<float>> anchor_grids;
};

template <typename DATATYPE>
struct _NetTensor
{
    enum class DimensionType : int {
        /** for tensorflow net type. uses NHWC as data format. */
        NHWC = 1,
        /** for caffe net type. uses NCHW as data format. */
        NCHW,
    };
    DATATYPE* data = nullptr;
    DimensionType format = DimensionType::NCHW;
    size_t batch = 1;
    size_t channels = 1;
    size_t height = 1;
    size_t width = 1;
    char reserve[8];
};
typedef _NetTensor<float> NetFloatTensor;
typedef _NetTensor<uint8_t> NetUINT8Tensor;

typedef struct {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef struct
{
    uint8_t* cls;
    float* probs;
    int height;
    int width;
    char reserve[8];
}SegmentResult;

typedef enum:int
{
    IMG_BGR = 0,
    IMG_RGB = 1,
    IMG_GRAY = 2
}InputDataType;


template <typename DATATYPE>
struct _Rect
{
    DATATYPE xmin;
    DATATYPE ymin;
    DATATYPE xmax;
    DATATYPE ymax;

    DATATYPE width;
    DATATYPE height;
};
typedef _Rect<float> RectFloat;
typedef _Rect<int> RectInt;

template <typename DATATYPE>
struct _Point
{
    DATATYPE x;
    DATATYPE y;
};
typedef _Point<float> PointFloat;
typedef _Point<int> PointInt;


typedef struct
{
	RectFloat rect;
	int id;
	char reserve[8];
}RectWithID;

#define PI (3.141592653589793)

#endif //AI_ALG_DATA_TYPE_H
