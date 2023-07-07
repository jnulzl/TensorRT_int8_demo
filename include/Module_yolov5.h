#ifndef MODULE_YOLOV5_H
#define MODULE_YOLOV5_H

#include <string>
#include <vector>
#include "data_type.h"
#include "alg_define.h"

class AIWORKS_PUBLIC CModule_yolov5
{
public:
	CModule_yolov5(const std::string& engine_name);
	~CModule_yolov5();

	void init(const YoloConfig& config);

    void process(const uint8_t* src, int src_height, int src_width, InputDataType inputDataType = InputDataType::IMG_BGR);

    const BoxInfos* get_result();

private:
	AW_ANY_POINTER impl_;
};

#endif // MODULE_YOLOV5_H

