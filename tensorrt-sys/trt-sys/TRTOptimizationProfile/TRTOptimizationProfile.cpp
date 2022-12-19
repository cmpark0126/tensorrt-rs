//
// Created by cmpark on 12/05/22.
//
#include <memory>
#include <NvInfer.h>

void optimization_profile_set_dimensions(nvinfer1::IOptimizationProfile* profile, char const* inputName, nvinfer1::OptProfileSelector select, nvinfer1::Dims dims) {
    profile->setDimensions(inputName, select, dims);
    return;
}
