#[cfg(test)]
mod tests;

use std::marker::PhantomData;

use crate::engine::Engine;
use crate::network::Network;
use crate::runtime::Logger;
use num_derive::FromPrimitive;
use tensorrt_sys::create_network_v2;

use tensorrt_sys::*;

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum DeviceType {
    GPU,
    DLA,
}

#[repr(C)]
#[derive(Eq, PartialEq, Debug, FromPrimitive)]
pub enum EngineCapability {
    Default,
    SafeGpu,
    SafeDla,
}

pub struct Builder<'a> {
    pub(crate) internal_builder: *mut tensorrt_sys::nvinfer1_IBuilder,
    pub(crate) logger: PhantomData<&'a Logger>,
}

bitflags! {
    pub struct NetworkBuildFlags: u32 {
        const DEFAULT = 0b0;
        const EXPLICIT_BATCH = 0b1;
        const EXPLICIT_PRECISION = 0b10;
    }
}

impl<'a> Builder<'a> {
    pub fn new(logger: &'a Logger) -> Self {
        let internal_builder = unsafe { create_infer_builder(logger.internal_logger) };
        let logger = PhantomData;
        Self {
            internal_builder,
            logger,
        }
    }

    pub fn get_max_batch_size(&self) -> i32 {
        unsafe { builder_get_max_batch_size(self.internal_builder) as i32 }
    }

    pub fn set_max_batch_size(&self, bs: i32) {
        unsafe { builder_set_max_batch_size(self.internal_builder, bs as i32) }
    }

    pub fn platform_has_fast_fp16(&self) -> bool {
        unsafe { builder_platform_has_fast_fp16(self.internal_builder) }
    }

    pub fn platform_has_fast_int8(&self) -> bool {
        unsafe { builder_platform_has_fast_int8(self.internal_builder) }
    }

    pub fn get_max_dla_batch_size(&self) -> i32 {
        unsafe { builder_get_max_dla_batch_size(self.internal_builder) }
    }

    pub fn get_nb_dla_cores(&self) -> i32 {
        unsafe { builder_get_nb_dla_cores(self.internal_builder) }
    }

    pub fn create_network_v2(&self, flags: NetworkBuildFlags) -> Network {
        let internal_network = unsafe { create_network_v2(self.internal_builder, flags.bits()) };
        Network { internal_network }
    }

    pub fn build_cuda_engine(&self, network: &Network) -> Engine {
        let internal_engine =
            unsafe { build_cuda_engine(self.internal_builder, network.internal_network) };
        Engine { internal_engine }
    }
}

impl<'a> Drop for Builder<'a> {
    fn drop(&mut self) {
        unsafe { destroy_builder(self.internal_builder) };
    }
}
