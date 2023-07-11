//
// Created by lizhaoliang-os on 2020/6/23.

#include <iostream>
#include <string>
#include <chrono>

#include <thread>
#include <mutex>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"

#include "Module_yolov5.h"


void open_video(const std::string& video_path, cv::VideoCapture& cap)
{
    cap.open(video_path);
    if (!cap.isOpened())
    {
        std::cout << "Unable open video/camera " << video_path << std::endl;
        return;
    }
}


void yolov6_thread_func(const std::string& input_name, const std::string& output_name,
                        const std::string& trt_engine_file_path, int net_inp_width, int net_inp_height,
                        int det_num_cls, int batch_size, int device_id,
                        const std::string& input_src, int is_save_res, const std::string& output_prefix)
{
    std::string project_root = std::string(PROJECT_ROOT);

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    input_names.push_back(input_name);
    output_names.push_back(output_name);

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
    config_tmp.model_include_preprocess = 0;
    config_tmp.conf_thres = 0.5;
    config_tmp.nms_thresh = 0.3;
    config_tmp.strides = {8, 16, 32};
    config_tmp.anchor_grids = { {10, 13, 16, 30, 33, 23} , {30, 61, 62, 45, 59, 119}, {116, 90, 156, 198, 373, 326} };

    config_tmp.net_inp_width = net_inp_width;
    config_tmp.net_inp_height = net_inp_height;
    config_tmp.num_cls = det_num_cls;
    config_tmp.batch_size = batch_size;
    config_tmp.device_id = device_id;
    config_tmp.input_names = input_names;
    config_tmp.output_names = output_names;
    config_tmp.weights_path = trt_engine_file_path;
    config_tmp.deploy_path = trt_engine_file_path;

    CModule_yolov5 yolov5("tensorrt");
    yolov5.init(config_tmp);

    std::vector<cv::VideoCapture> caps;
    caps.resize(config_tmp.batch_size);
    for (int bs = 0; bs < config_tmp.batch_size; ++bs)
    {
        open_video(input_src, caps[bs]);
    }

    long frame_id = 0;
    std::vector<ImageInfoUint8> img_batch;
    img_batch.resize(config_tmp.batch_size);

    std::vector<cv::Mat> frames;
    frames.resize(config_tmp.batch_size);
    while (true)
    {
        int count_empty_frame = 0;
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            caps[bs] >> frames[bs];
            if (!frames[bs].data)
            {
                count_empty_frame++;
            }
        }
        if(config_tmp.batch_size == count_empty_frame)
        {
            break;
        }
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            if (frames[bs].data)
            {
                img_batch[bs].data = frames[bs].data;
                img_batch[bs].img_height = frames[bs].rows;
                img_batch[bs].img_width = frames[bs].cols;
                img_batch[bs].img_data_type = InputDataType::IMG_BGR;
            }
            else
            {
                img_batch[bs].data = nullptr;
                img_batch[bs].img_height = 0;
                img_batch[bs].img_width = 0;
                img_batch[bs].img_data_type = InputDataType::IMG_BGR;
            }
        }

        std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
        yolov5.process_batch(img_batch.data(), config_tmp.batch_size);
        std::chrono::time_point<std::chrono::system_clock> finishTP1 = std::chrono::system_clock::now();

        const BoxInfos* res = yolov5.get_result();
        std::cout << output_prefix << " Thread id = " << std::this_thread::get_id() << " Frame = " << frame_id << " Batch = " << config_tmp.batch_size << " TensorRT process time = " << std::chrono::duration_cast<std::chrono::microseconds>(finishTP1 - startTP).count() << " us" << std::endl;
        for (int bs = 0; bs < config_tmp.batch_size; ++bs)
        {
            if(img_batch[bs].data)
            {
                std::cout << output_prefix << " Video " << bs << " detected " << res[bs].size << " objs" << std::endl;
            }
            else
            {
                std::cout << output_prefix << " Video is end!" << std::endl;
            }
        }

        if(1 == is_save_res)
        {
            //show result
            for (int bs = 0; bs < config_tmp.batch_size; ++bs)
            {
                if(img_batch[bs].data)
                {
                    cv::Mat img_show = cv::Mat(img_batch[bs].img_height, img_batch[bs].img_width, CV_8UC3, img_batch[bs].data);
                    for (size_t idx = 0; idx < res[bs].size; idx++)
                    {
                        int xmin    = res[bs].boxes[idx].x1;
                        int ymin    = res[bs].boxes[idx].y1;
                        int xmax    = res[bs].boxes[idx].x2;
                        int ymax    = res[bs].boxes[idx].y2;
                        float score = res[bs].boxes[idx].score;
                        int label   = res[bs].boxes[idx].label;
                        //std::cout << "xyxy : " << xmin << " " << ymin << " " << xmax << " " << ymax << " " << score << " " << label << std::endl;
                        cv::rectangle(img_show, cv::Point2i(xmin, ymin), cv::Point2i(xmax, ymax), cv::Scalar(255, 0, 0), 2);
                        cv::putText(img_show, std::to_string(label), cv::Point(xmin, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 255), 2);
                        cv::putText(img_show, std::to_string(score), cv::Point(xmax, ymin), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 2);
                    }
                    cv::imwrite("res/" + output_prefix + "_video_" + std::to_string(bs) + "_frame" + std::to_string(frame_id) +
                                "_bs" + std::to_string(config_tmp.batch_size) + "_obj" + std::to_string(res[bs].size) +  ".jpg",img_show);
                }
            }
        }

        frame_id++;
    }

    for (int bs = 0; bs < config_tmp.batch_size; ++bs)
    {
        caps[bs].release();
    }
}


int main(int argc, char* argv[])
{
    if(argc < 8)
    {
        std::cout << "Usage:\n\t "
                  << argv[0] << " trt_dingshi trt_ceshi batch_size device_id video_dingshi_path video_ceshi_path is_save_res"
                  << std::endl;
        return -1;
    }
    std::string input_name = "images";
    std::string output_name = "outputs";

    std::string weights_path_dingshi = std::string(argv[1]);
    std::string weights_path_ceshi = std::string(argv[2]);
    int batch_size = std::atoi(argv[3]);
    int device_id = std::atoi(argv[4]);
    std::string dingshi_video_path = std::string(argv[5]);
    std::string ceshi_video_path = std::string(argv[6]);
    int is_save_res = std::atoi(argv[7]);

    int net_inp_width = 640;
    int net_inp_height = 640;
//    yolov6_thread_func(input_name, output_name, weights_path, net_inp_width, net_inp_height,
//                       det_num_cls, batch_size, device_id, input_src, is_save_res);

//    std::vector<std::thread> threads;
//    for (int t_id = 0; t_id < ; ++t_id) {
//
//    }
    std::thread t1_dingshi(yolov6_thread_func, input_name, output_name, weights_path_dingshi, net_inp_width, net_inp_height,
                       8, batch_size, device_id, dingshi_video_path, is_save_res, "Dingshi");

    std::thread t2_ceshi(yolov6_thread_func, input_name, output_name, weights_path_ceshi, net_inp_width, net_inp_height,
                   31, batch_size, device_id, ceshi_video_path, is_save_res, "Ceshi");

    t1_dingshi.join();
    t2_ceshi.join();
    return 0;
}
