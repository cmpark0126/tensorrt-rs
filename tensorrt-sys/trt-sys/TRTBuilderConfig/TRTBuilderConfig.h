//
// Created by cmpark on 12/05/22.
//

#ifndef LIBTRT_TRTBUILDERCONFIG_H
#define LIBTRT_TRTBUILDERCONFIG_H

#include <NvInfer.h>

#include <stddef.h>
#include <stdint.h>

void add_optimization_profile(nvinfer1::IBuilderConfig * config, nvinfer1::IOptimizationProfile const* profile);

void set_memory_pool_limit(nvinfer1::IBuilderConfig * config, nvinfer1::MemoryPoolType pool, std::size_t poolSize);

#endif //LIBTRT_TRTBUILDERCONFIG_H
