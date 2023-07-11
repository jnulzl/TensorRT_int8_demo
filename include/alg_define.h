//
// Created by lizhaoliang-os on 2020/6/23.
//

#ifndef ALG_DEFINE_H
#define ALG_DEFINE_H

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cstdio>

/*****************************some common function*********************************/
// the alignment of all the allocated buffers
#define AI_MALLOC_ALIGN 64

// we have some optimized kernels that may overread buffer a bit in loop
// it is common to interleave next-loop data load with arithmetic instructions
// allocating more bytes keeps us safe from SEGV_ACCERR failure
#define AI_MALLOC_OVERREAD 64

// Aligns a pointer to the specified number of bytes
// ptr Aligned pointer
// n Alignment size that must be a power of two
template<typename _Tp>
static  _Tp* alignPtr(_Tp* ptr, int n = (int)sizeof(_Tp))
{
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

// Aligns a buffer size to the specified number of bytes
// The function returns the minimum number that is greater or equal to sz and is divisible by n
// sz Buffer size to align
// n Alignment size that must be a power of two
static size_t alignSize(size_t sz, int n)
{
    return (sz + n - 1) & -n;
}

static void* fastMalloc(size_t size)
{
#if _MSC_VER
    return _aligned_malloc(size, NCNN_MALLOC_ALIGN);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
    void* ptr = 0;
    if (posix_memalign(&ptr, AI_MALLOC_ALIGN, size + AI_MALLOC_OVERREAD))
        ptr = 0;
    return ptr;
#elif __ANDROID__ && __ANDROID_API__ < 17
    return memalign(NCNN_MALLOC_ALIGN, size + NCNN_MALLOC_OVERREAD);
#else
    unsigned char* udata = (unsigned char*)malloc(size + sizeof(void*) + NCNN_MALLOC_ALIGN + NCNN_MALLOC_OVERREAD);
    if (!udata)
        return 0;
    unsigned char** adata = alignPtr((unsigned char**)udata + 1, NCNN_MALLOC_ALIGN);
    adata[-1] = udata;
    return adata;
#endif
}

static void fastFree(void* ptr)
{
    if (ptr)
    {
#if _MSC_VER
        _aligned_free(ptr);
#elif (defined(__unix__) || defined(__APPLE__)) && _POSIX_C_SOURCE >= 200112L || (__ANDROID__ && __ANDROID_API__ >= 17)
        free(ptr);
#elif __ANDROID__ && __ANDROID_API__ < 17
        free(ptr);
#else
        unsigned char* udata = ((unsigned char**)ptr)[-1];
        free(udata);
#endif
    }
}

#ifdef USE_CUDA
#include <cuda_runtime.h>

#define CUDACHECK(status)                                                                                                  \
    do                                                                                                                 \
    {                                                                                                                  \
        auto ret = (status);                                                                                           \
        if (ret != 0)                                                                                                  \
        {                                                                                                              \
            std::cerr << "Cuda failure: " << ret << std::endl;                                                 \
            abort();                                                                                                   \
        }                                                                                                              \
    } while (0)

static void* cudaFastMalloc(size_t size, size_t device_id)
{
    CUDACHECK(cudaSetDevice(device_id));
    void* devPtr;
    CUDACHECK(cudaMalloc(&devPtr, alignSize(size + sizeof(void*) + AI_MALLOC_OVERREAD, AI_MALLOC_ALIGN)));
    return devPtr;
}

static void cudaFastFree(void* devPtr, size_t device_id)
{
    CUDACHECK(cudaSetDevice(device_id));
    if (devPtr)
    {
        CUDACHECK(cudaFree(devPtr));
    }
}

#endif // USE_CUDA

/**************************************************************/

#if defined(__APPLE__)
    #include <TargetConditionals.h>
    #if TARGET_OS_IPHONE
        #define AIWORKS_BUILD_FOR_IOS
    #endif
#endif

#if defined(__aarch64__) || defined(__arm__)
    #include <android/log.h>
    #define AIWORKS_ERROR(format, ...) __android_log_print(ANDROID_LOG_ERROR, "MNNJNI", format, ##__VA_ARGS__)
    #define AIWORKS_PRINT(format, ...) __android_log_print(ANDROID_LOG_INFO, "MNNJNI", format, ##__VA_ARGS__)
#else
    #define AIWORKS_PRINT(format, ...) printf("%d, %s:", __LINE__, __FUNCTION__); \
                                        printf(format, ##__VA_ARGS__)
    #define AIWORKS_ERROR(format, ...) printf("%d, %s:", __LINE__, __FUNCTION__); \
                                        printf(format, ##__VA_ARGS__)
#endif

#ifdef DEBUG
#define AIWORKS_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            AIWORKS_ERROR("Error for %s, %d\n", __FILE__, __LINE__); \
            assert(res);                                         \
        }                                                        \
    }
#else
#define AIWORKS_ASSERT(x)                                            \
    {                                                            \
        int res = (x);                                           \
        if (!res) {                                              \
            AIWORKS_ERROR("Error for %d\n", __LINE__); \
        }                                                        \
    }
#endif

#define AIWORKS_FUNC_PRINT(x) AIWORKS_PRINT(#x "=%d in %s, %d \n", x, __func__, __LINE__);
#define AIWORKS_FUNC_PRINT_ALL(x, type) AIWORKS_PRINT(#x "=" #type " %" #type " in %s, %d \n", x, __func__, __LINE__);

#define AIWORKS_CHECK(success, log) \
if(!(success)){ \
AIWORKS_ERROR("Check failed: %s ==> %s\n", #success, #log); \
}


#if defined(_MSC_VER)
    #if defined(BUILDING_AIWORKS_DLL)
        #define AIWORKS_PUBLIC __declspec(dllexport)
    #elif defined(USING_AIWORKS_DLL)
        #define AIWORKS_PUBLIC __declspec(dllimport)
    #else
        #define AIWORKS_PUBLIC
    #endif
#else
    #define AIWORKS_PUBLIC __attribute__((visibility("default")))
#endif

#define ALG_ENGINE_IMPL(alg, lower_name)  \
    CModule_##alg##_##lower_name##_impl()

#endif //ALG_DEFINE_H
