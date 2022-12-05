//
// Created by mason on 11/27/19.
//

#ifndef LIBTRT_TRTBUILDER_H
#define LIBTRT_TRTBUILDER_H

#include <NvInfer.h>
#include "../TRTLogger/TRTLogger.h"
#include "../TRTEnums.h"

#include <stddef.h>
#include <stdint.h>

nvinfer1::IBuilder *create_infer_builder(Logger_t *logger);
void destroy_builder(nvinfer1::IBuilder* builder);
void builder_set_max_batch_size(nvinfer1::IBuilder* builder, int32_t batch_size);
int32_t builder_get_max_batch_size(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_fp16(nvinfer1::IBuilder* builder);
bool builder_platform_has_fast_int8(nvinfer1::IBuilder* builder);
int builder_get_max_dla_batch_size(nvinfer1::IBuilder* builder);
int builder_get_nb_dla_cores(nvinfer1::IBuilder* builder);
nvinfer1::INetworkDefinition *create_network_v2(nvinfer1::IBuilder* builder, uint32_t flags);
nvinfer1::IBuilderConfig *create_builder_config(nvinfer1::IBuilder* builder);
nvinfer1::IOptimizationProfile* create_optimization_profile(nvinfer1::IBuilder* builder);
nvinfer1::IHostMemory* builder_build_serialized_network(nvinfer1::IBuilder* builder, nvinfer1::INetworkDefinition* network, nvinfer1::IBuilderConfig* config);

#endif //LIBTRT_TRTBUILDER_H
