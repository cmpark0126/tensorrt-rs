use crate::dims::Dim;
use std::ffi::{CStr, CString};
use tensorrt_sys::*;

pub struct OptimizationProfile {
    pub internal_optimization_profile: *mut tensorrt_sys::nvinfer1_IOptimizationProfile,
}

impl OptimizationProfile {
    pub fn set_min_dimensions<D: Dim>(&self, input_name: &str, dims: D) {
        unsafe {
            optimization_profile_set_dimensions(
                self.internal_optimization_profile,
                CString::new(input_name).unwrap().as_ptr(),
                nvinfer1_OptProfileSelector_kMIN,
                dims.get_internal_dims(),
            )
        }
    }

    pub fn set_opt_dimensions<D: Dim>(&self, input_name: &str, dims: D) {
        unsafe {
            optimization_profile_set_dimensions(
                self.internal_optimization_profile,
                CString::new(input_name).unwrap().as_ptr(),
                nvinfer1_OptProfileSelector_kOPT,
                dims.get_internal_dims(),
            )
        }
    }

    pub fn set_max_dimensions<D: Dim>(&self, input_name: &str, dims: D) {
        unsafe {
            optimization_profile_set_dimensions(
                self.internal_optimization_profile,
                CString::new(input_name).unwrap().as_ptr(),
                nvinfer1_OptProfileSelector_kMAX,
                dims.get_internal_dims(),
            )
        }
    }
}
