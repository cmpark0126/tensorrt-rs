//
// Created by mason on 4/30/20.
//
#include <cstring>
#include <cstdlib>
#include "TRTDims.h"

nvinfer1::Dims2 create_dims2(int dim1, int dim2) {
    return nvinfer1::Dims2(dim1, dim2);
}

nvinfer1::Dims3 create_dims3(int dim1, int dim2, int dim3) {
    return nvinfer1::Dims3(dim1, dim2, dim3);
}

nvinfer1::Dims4 create_dims4(int dim1, int dim2, int dim3, int dim4) {
    return nvinfer1::Dims4(dim1, dim2, dim3, dim4);
}
