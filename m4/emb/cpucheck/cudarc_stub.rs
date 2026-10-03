//! Stand-in for the two cudarc marker traits iquant.rs names, so the unfixed file (impls without
//! `#[cfg(feature = "cuda")]`) can still be compiled for the "before" run.
pub mod driver {
    pub unsafe trait DeviceRepr {}
    pub unsafe trait ValidAsZeroBits {}
}
