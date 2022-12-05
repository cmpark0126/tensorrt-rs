//
// Created by mason on 11/27/19.
//
#include <memory>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "TRTBuilder.h"
#include "../TRTLogger/TRTLoggerInternal.hpp"

void builder_set_max_batch_size(nvinfer1::IBuilder* builder, int32_t batch_size) {
    builder->setMaxBatchSize(batch_size);
}

int32_t builder_get_max_batch_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxBatchSize();
}

bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder){
    return builder->platformHasFastFp16();
}

bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder) {
    return builder->platformHasFastInt8();
}

int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder) {
    return builder->getMaxBatchSize();
}

int builder_get_nb_dla_cores(nvinfer1::IBuilder* builder) {
    return builder->getNbDLACores();
}

nvinfer1::IBuilder *create_infer_builder(Logger_t *logger) {
    initLibNvInferPlugins(&logger->getLogger(), "");
    return nvinfer1::createInferBuilder(logger->getLogger());
}


void destroy_builder(nvinfer1::IBuilder* builder) {
    builder->destroy();
}

nvinfer1::INetworkDefinition *create_network_v2(nvinfer1::IBuilder *builder, uint32_t flags) {
    return builder->createNetworkV2(flags);
}

nvinfer1::IBuilderConfig *create_builder_config(nvinfer1::IBuilder* builder) {
    return builder->createBuilderConfig();
}

nvinfer1::IOptimizationProfile* create_optimization_profile(nvinfer1::IBuilder* builder) {
    return builder->createOptimizationProfile();
}

nvinfer1::IHostMemory* builder_build_serialized_network(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, nvinfer1::IBuilderConfig* config) {
    return builder->buildSerializedNetwork(*network, *config);
}