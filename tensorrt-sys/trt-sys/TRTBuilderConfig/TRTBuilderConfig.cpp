//
// Created by cmpark on 12/05/22.
//
#include <memory>
#include <NvInfer.h>

void add_optimization_profile(nvinfer1::IBuilderConfig * config, nvinfer1::IOptimizationProfile const* profile) {
    config->addOptimizationProfile(profile);
    return;
}

void set_memory_pool_limit(nvinfer1::IBuilderConfig * config, nvinfer1::MemoryPoolType pool, std::size_t poolSize) {
    return config->setMemoryPoolLimit(pool, poolSize);
}
