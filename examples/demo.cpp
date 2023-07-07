//
// Created by lizhaoliang-os on 2020/6/23.

#include <iostream>
#include <string>
#include <chrono>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "Module_yolov5.h"

int main(int argc, char* argv[])
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::string weights_path;
    std::string deploy_path;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.push_back("input");
    output_names.push_back("output");

    std::string input_src;

    YoloConfig config_tmp;
    float means_rgb[3] = {0, 0, 0};
    float scales_rgb[3] = {0.0039215, 0.0039215, 0.0039215}; // 1.0 / 255

    config_tmp.means[0] = means_rgb[0];
    config_tmp.means[1] = means_rgb[1];
    config_tmp.means[2] = means_rgb[2];
    config_tmp.scales[0] = scales_rgb[0];
    config_tmp.scales[1] = scales_rgb[1];
    config_tmp.scales[2] = scales_rgb[2];

    config_tmp.mean_length = 3;
    config_tmp.net_inp_channels = 3;

    config_tmp.num_cls = 1;
    config_tmp.model_include_preprocess = 1;
    config_tmp.conf_thres = 0.5;
    config_tmp.nms_thresh = 0.3;
    config_tmp.strides = {8, 16, 32};
    config_tmp.anchor_grids = { {10, 13, 16, 30, 33, 23} , {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326} };

    if(argc < 4)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " onnx_model_path input_size video_path"
                  << std::endl;
        return -1;
    }

    weights_path = std::string(argv[1]);
    deploy_path = std::string(argv[1]);
    input_src = argv[3];
    config_tmp.net_inp_width = std::atoi(argv[2]);
    config_tmp.net_inp_height = config_tmp.net_inp_width;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = weights_path;
    config_tmp.deploy_path = deploy_path;

    CModule_yolov5 yolov5("tensorrt");
    yolov5.init(config_tmp);

    cv::VideoCapture cap(input_src);
    if (!cap.isOpened())
    {
        cap.open(0);

        if (!cap.isOpened())
        {
            std::cout << "Unable open video/camera " << input_src << std::endl;
            return -1;
        }
    }
    std::chrono::time_point<std::chrono::system_clock> begin_time = std::chrono::system_clock::now();
    long frame_id = 0;    
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (!frame.data)
        {
            break;
        }
        cv::Mat img_origin = frame.clone();
        cv::Mat img_show = frame.clone();

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        yolov5.process(frame.data, frame.rows, frame.cols);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const BoxInfos* res = yolov5.get_result();
        std::cout << "frame_id:" << frame_id << " Detected obj num : " <<  res->size << " TensorRT process each frame time = " << std::chrono::duration_cast<std::chrono::microseconds>(finishTP1 - startTP).count() << " us" << std::endl;
        //show result
        for (size_t idx = 0; idx < res->size; idx++)
        {
            int xmin    = res->boxes[idx].x1;
            int ymin    = res->boxes[idx].y1;
            int xmax    = res->boxes[idx].x2;
            int ymax    = res->boxes[idx].y2;
            float score = res->boxes[idx].score;
            int label   = res->boxes[idx].label;
            //std::cout << "xyxy : " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << score << " " << label << std::endl;
            cv::rectangle(img_show, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
            cv::putText(img_show, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
            cv::putText(img_show, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
        }
        cv::imwrite("res/" + std::to_string(frame_id) + ".jpg",img_show);
        if(frame_id > 20)
            break;
        frame_id++;
    }
    std::chrono::time_point<std::chrono::system_clock> end_time = std::chrono::system_clock::now();
    std::cout << "TensorRT process average each frame time = " << std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin_time).count() / frame_id << " us" << std::endl;
    cap.release();

    return 0;
}
