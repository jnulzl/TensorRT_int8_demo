# TensorRT int8 Demo with yolov5

`本仓库以yolov5为例记录了TensorRT int8使用方法，主要包括：`模型int8量化`和`int8模型推理部署`两部分，详细见下面`

## 开发环境

- Ubuntu 16.04 + 

- CUDA 11.x

- TensorRT 7.x / TensorRT 7.x

- cudnn 8.x

- OpenCV 3.x

## 模型int8量化

- 编译onnx2trtInt8和demo

```shell
# build onnx2trtInt8 
>>mkdir build && cd build
>>export PATH="YOUR_CUDA_HOME"/bin:$PATH
>>cmake —DCMAKE_BUILD_TYPE=Release -DCUDA_HOME="YOUR_CUDA_HOME" -DCUDNN_HOME="YOUR_CUDNN_HOME" -DTRT_HOME="YOUR_TENSORRT_HOME" -DOpenCV_DIR="YOUR_OPENCV_DIR" ..
>>make VERBOSE=1 -j8 
>>
# build complete you will get onnx2trtInt8 and demo in $YOUR_ROOT/bin/Linux directory
```
- onnx2trtInt8使用

```
>>./onnx2trtInt8
Usage:
	 ./onnx2trtInt8 quant_int8_conifg.json
```

在终端运行`./onnx2trtInt8`后可以看到，输入参数为一个`quant_int8_conifg.json`，其内容如下：

```json
{
  "onnxFilePath": "YOUR_ONNX_MODEL_PATH",
  "engine_file_save_path": "YOU_WILL_SAVED_TRT_MODEL_PATH",
  // img_list_file 和 preprocessed_binary_file_path同时可用的话优先使用前者.
  "img_list_file": "CALIBRATION_IMAGE_TXT_LIST", //
  "preprocessed_binary_file_path": "preprocessed_binary_file_path",  
  // 推理时候的最大batch(目前只测试了batchsize=1的情况)
  "maxInferenceBatchSize": 4, 
  // 网络输入通道数，一般为1或者3
  "netInputChannels": 3, 
  "netInputHeight": 640, 
  "netInputWidth": 640,
  // 校准bathsize，一定要小于img_list_file或者preprocessed_binary_file_path中的图像数量
  "calBatchSize": 16,  
  // 网络输入tensor名字，一般只有一个
  "input": [           
    "input"
  ],
  // 网络输出tensor名字，可有一个或多个
  "output": [          
    "output1",
    "output2",
    "output3"
  ],
  "networkName": "_yolo_face_int8",
  // 图像预处理时的均值，如果netInputChannels=1，则只第一个数有效
  "means": [    
    0.0,
    0.0,
    0.0
  ],
  // 图像预处理时的缩放系数，如果netInputChannels=1，则只第一个数有效
  "scales": [  
    0.00392157,
    0.00392157,
    0.00392157
  ],
  // 图像是否等比例缩放，如果为1，则根据isSymmetryPad在边缘补0
  "isFixResize": 1, 
  //isSymmetryPad=1，则缩放时候在图像左右或者上下对称补0;否则为0，则在图像右面或者下面补0
  "isSymmetryPad": 0, 
  // 是否进行bgr -> rgb(或者rgb -> bgr)的转换
  "isBGR2RGB": 1,
  // 是否进行hwc -> chw的转换
  "isHWC2CHW": 1,
  "dlaCore" : -1,  
  "is_save_jpg_after_bgr2rgb": false
}
```

详细例子可进入`$YOUR_ROOT/bin/Linux`，然后执行:

- 生成`preprocessed_binary_file_path`文件

```shell
# 可根据个人实际prepreocess情况修改crop_yolo_resize.py中的代码
>>python crop_yolo_resize.py
0 widerface_val_subset/000000000065.png 768 1024
1 widerface_val_subset/000000000015.png 681 1024
2 widerface_val_subset/000000000053.png 820 1024
3 widerface_val_subset/000000000040.png 683 1024
4 widerface_val_subset/000000000042.png 1050 1024
5 widerface_val_subset/000000000074.png 768 1024
6 widerface_val_subset/000000000062.png 755 1024
7 widerface_val_subset/000000000019.png 768 1024
8 widerface_val_subset/000000000026.png 819 1024
9 widerface_val_subset/000000000041.png 642 1024
10 widerface_val_subset/000000000070.png 731 1024
11 widerface_val_subset/000000000063.png 768 1024
12 widerface_val_subset/000000000014.png 807 1024
......
47 widerface_val_subset/000000000047.png 1530 1024
all_imgs :  (48, 3, 640, 640)
```

运行完成之后可得到`yolo_face_preprocessed_bin_file_input_size640_num48.bin`文件

- 利用`img_list_file`直接量化

```shell
>>./onnx2trtInt8 quant_int8_conifg.json 
[09/08/2022-20:09:08] [I] [TRT] [MemUsageChange] Init CUDA: CPU +536, GPU +0, now: CPU 547, GPU 3366 (MiB)
[09/08/2022-20:09:08] [I] Current environment support int8!
&&&& RUNNING SampleInt8Demo [TensorRT v8001] # ./onnx2trtInt8 quant_int8_conifg.json
[09/08/2022-20:09:08] [I] Building and running a GPU inference engine for INT8 sample
[09/08/2022-20:09:08] [I] [TRT] [MemUsageChange] Init CUDA: CPU +0, GPU +0, now: CPU 547, GPU 3366 (MiB)
[09/08/2022-20:09:08] [I] [TRT] ----------------------------------------------------------------
[09/08/2022-20:09:08] [I] [TRT] Input filename:   ../../models/yolov5/face/face_640_input_one_output.onnx
[09/08/2022-20:09:08] [I] [TRT] ONNX IR version:  0.0.6
[09/08/2022-20:09:08] [I] [TRT] Opset version:    11
[09/08/2022-20:09:08] [I] [TRT] Producer name:    pytorch
[09/08/2022-20:09:08] [I] [TRT] Producer version: 1.5
[09/08/2022-20:09:08] [I] [TRT] Domain:           
[09/08/2022-20:09:08] [I] [TRT] Model version:    0
[09/08/2022-20:09:08] [I] [TRT] Doc string:       
[09/08/2022-20:09:08] [I] [TRT] ----------------------------------------------------------------
[09/08/2022-20:09:08] [W] [TRT] onnx2trt_utils.cpp:364: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
......
```

运行完成之后可得到`$YOUR_ROOT/models/yolov5/face/face_640_input_one_output_int8.trt`


## int8模型推理部署

上一步已经在`$YOUR_ROOT/bin/Linux`目录中编译出`demo`，这里直接可以运行量化得到的文件，如下所示

```shell
>>cd $YOUR_ROOT/bin/Linux
>>./demo ../../models/yolov5/face/face_640_input_one_output_int8.trt  640 ../../data/test0.mp4
```

## 其它说明

- **如果你用的CUDA, OpenCV, TensorRT或者cudnn与上述描述的不一致可能存在细微的api差异，可在此基础上稍微修改即可**。

- **[YOLOv6-TensorRT in C++](https://github.com/meituan/YOLOv6/tree/main/deploy/TensorRT)与本仓库类似，也可以参考**。
