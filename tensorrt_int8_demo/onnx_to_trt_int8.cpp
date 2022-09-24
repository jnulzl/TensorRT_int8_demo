//! Created by jnulzl on 2022/9/6.
//!
//! onnx_to_txt_int8.cpp
//! This file contains the implementation of the sample. It creates the network using
//! the onnx model.

#include "OnnxBatchStream.hpp"
#include "npp_image_preprocess.h"

int main(int argc, char** argv)
{
    if(2 != argc)
    {
        std::cout << "Usage:\n\t " <<  argv[0] << " quant_int8_conifg.json" << std::endl;
        return -1;
    }
    if(!is_support_int8())
    {
        return 0;
    }

    SampleINT8Params params;
    set_quant_params(argv[1], params);
    SampleINT8 sample(params);

    auto sampleTest = sample::gLogger.defineTest("SampleInt8Demo", argc, argv);
    sample::gLogger.reportTestStart(sampleTest);
    sample::gLogInfo << "Building and running a GPU inference engine for INT8 sample" << std::endl;

    std::vector<std::string> dataTypeNames = {"INT8"};
    std::vector<DataType> dataTypes = {DataType::kINT8};
    for (size_t i = 0; i < dataTypes.size(); i++)
    {
        if (!sample.build(dataTypes[i]))
        {
#if NV_TENSORRT_MAJOR >= 8
            if (!samplesCommon::isDataTypeSupported(dataTypes[i]))
#else
            if (!is_support_int8())
#endif
            {
                sample::gLogWarning << "Skipping " << dataTypeNames[i]
                                    << " since the platform does not support this data type." << std::endl;
                continue;
            }
            return sample::gLogger.reportFail(sampleTest);
        }
    }
    return sample::gLogger.reportPass(sampleTest);
}
