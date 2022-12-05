use std::error;
use std::fmt::Formatter;
use tensorrt_sys::*;

mod private {
    pub trait DimsPrivate {
        fn get_internal_dims(&self) -> super::nvinfer1_Dims;
    }
}

pub trait Dim: private::DimsPrivate {
    fn nb_dims(&self) -> i32 {
        self.get_internal_dims().nbDims
    }

    fn d(&self) -> [i32; 8] {
        self.get_internal_dims().d
    }
}

#[repr(C)]
pub enum DimensionType {
    Spacial,
    Channel,
    Index,
    Sequence,
}

#[repr(transparent)]
pub struct Dims(pub nvinfer1_Dims);

impl private::DimsPrivate for Dims {
    fn get_internal_dims(&self) -> nvinfer1_Dims {
        self.0
    }
}
impl Dim for Dims {}

#[repr(transparent)]
pub struct Dims2(nvinfer1_Dims2);

impl Dims2 {
    pub fn new(dim1: i32, dim2: i32) -> Dims2 {
        let internal_dims = unsafe { create_dims2(dim1, dim2) };

        Dims2(internal_dims)
    }
}

impl private::DimsPrivate for Dims2 {
    fn get_internal_dims(&self) -> nvinfer1_Dims {
        self.0._base
    }
}

impl Dim for Dims2 {}

#[repr(transparent)]
pub struct Dims3(nvinfer1_Dims3);

impl Dims3 {
    pub fn new(dim1: i32, dim2: i32, dim3: i32) -> Dims3 {
        let internal_dims = unsafe { create_dims3(dim1, dim2, dim3) };
        Dims3(internal_dims)
    }
}

impl private::DimsPrivate for Dims3 {
    fn get_internal_dims(&self) -> nvinfer1_Dims {
        self.0._base
    }
}

impl Dim for Dims3 {}

#[repr(transparent)]
pub struct Dims4(nvinfer1_Dims4);

impl Dims4 {
    pub fn new(dim1: i32, dim2: i32, dim3: i32, dim4: i32) -> Dims4 {
        let internal_dims = unsafe { create_dims4(dim1, dim2, dim3, dim4) };
        Dims4(internal_dims)
    }
}

impl private::DimsPrivate for Dims4 {
    fn get_internal_dims(&self) -> nvinfer1_Dims {
        self.0._base
    }
}

impl Dim for Dims4 {}

#[derive(Debug, Clone)]
pub struct DimsShapeError {
    message: String,
}

impl DimsShapeError {
    pub fn new(message: &str) -> Self {
        DimsShapeError {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for DimsShapeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl error::Error for DimsShapeError {}
