use crate::optimization_profile::OptimizationProfile;
use tensorrt_sys::*;

pub struct BuilderConfig {
    pub internal_builder_config: *mut nvinfer1_IBuilderConfig,
}

impl BuilderConfig {
    pub fn add_optimization_profile(&self, profile: OptimizationProfile) {
        unsafe {
            add_optimization_profile(
                self.internal_builder_config,
                profile.internal_optimization_profile,
            )
        }
    }

    pub fn set_max_workspace_size(&self, size: usize) {
        unsafe {
            set_memory_pool_limit(
                self.internal_builder_config,
                nvinfer1_MemoryPoolType_kWORKSPACE,
                size,
            )
        }
    }
}
