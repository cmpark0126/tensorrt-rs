//
// Created by cmpark on 12/05/22.
//

#ifndef LIBTRT_TRTOPTIMIZATIONPROFILE_H
#define LIBTRT_TRTOPTIMIZATIONPROFILE_H

#include <NvInfer.h>
#include "../TRTLogger/TRTLogger.h"
#include "../TRTEnums.h"

#include <stddef.h>
#include <stdint.h>

void optimization_profile_set_dimensions(nvinfer1::IOptimizationProfile* profile, char const* inputName, nvinfer1::OptProfileSelector select, nvinfer1::Dims dims);

#endif //LIBTRT_TRTOPTIMIZATIONPROFILE_H
