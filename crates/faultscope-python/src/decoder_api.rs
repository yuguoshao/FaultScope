use crate::*;
#[cfg(feature = "decoder-fusion-blossom")]
use faultscope_core::NativeFusionBlossomDecoder as CoreNativeFusionBlossomDecoder;
use faultscope_core::{
    BinaryLinearDecodingProblem, CorrectionMaskBatch, DetectorEventShotBatchView,
    DetectorMaskBatchView, FaultScopeNativeCorrectionMaskBatchMutViewV1,
    FaultScopeNativeDecoderI64SliceV1, FaultScopeNativeDecoderMaskMutViewV1,
    FaultScopeNativeDecoderMaskViewV1, FaultScopeNativeDecoderStatusV1,
    FaultScopeNativeDecoderStringViewV1, FaultScopeNativeDecoderV1,
    FaultScopeNativeDetectorEventShotBatchViewV1, FaultScopeNativeDetectorMaskBatchViewV1,
    FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, GraphlikeDecodingProblem, IndexedDem,
    NativeBatchDecoder as CoreNativeBatchDecoder,
    NativeCompositeDecoder as CoreNativeCompositeDecoder,
    NativeGraphlikeDetectorCopyDecoder as CoreNativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder as CoreNativeNoCorrectionDecoder, PackedDetectorShotBatchView,
    PackedObservableShotBatch, SparseBinaryMatrix, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_CAPSULE_METHOD, NATIVE_DECODER_PLUGIN_CAPSULE_NAME,
    NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE, NATIVE_DECODER_PLUGIN_STATUS_OK,
};
use std::ffi::CString;
use std::mem;
use std::ptr::NonNull;
use std::slice;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[pyfunction]
pub(crate) fn available_native_decoders(py: Python<'_>) -> PyResult<PyObject> {
    #[cfg(feature = "decoder-fusion-blossom")]
    let names = ["no-correction", "graphlike-detector-copy", "fusion-blossom"];
    #[cfg(not(feature = "decoder-fusion-blossom"))]
    let names = ["no-correction", "graphlike-detector-copy"];
    Ok(PyTuple::new(py, names)?.into())
}

/// Type-erased native batch decoder handle.
#[pyclass(name = "NativeBatchDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeBatchDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native decoder that composes independent child decoders.
#[pyclass(name = "NativeCompositeDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeCompositeDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native decoder that always returns an empty correction.
#[pyclass(name = "NativeNoCorrectionDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeNoCorrectionDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native graphlike decoder that copies detector masks to observables.
#[pyclass(
    name = "NativeGraphlikeDetectorCopyDecoder",
    module = "faultscope._native"
)]
pub(crate) struct PyNativeGraphlikeDetectorCopyDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[cfg(feature = "decoder-fusion-blossom")]
#[pyclass(name = "NativeFusionBlossomDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeFusionBlossomDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[pymethods]
impl PyNativeBatchDecoder {
    #[staticmethod]
    #[pyo3(signature = (observable_ids=None, detector_ids=None))]
    pub(crate) fn no_correction(
        observable_ids: Option<Vec<i64>>,
        detector_ids: Option<Vec<i64>>,
    ) -> Self {
        Self {
            inner: Arc::new(CoreNativeNoCorrectionDecoder::with_detector_ids(
                detector_ids.unwrap_or_default(),
                observable_ids.unwrap_or_default(),
            )),
            python_decode_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.detector_ids())
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.observable_ids())
    }

    pub(crate) fn decode_batch_masks(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.python_decode_calls.fetch_add(1, Ordering::Relaxed);
        decode_batch_masks_with_native_decoder(py, self.inner.clone(), batch)
    }

    #[getter]
    pub(crate) fn python_decode_call_count(&self) -> usize {
        self.python_decode_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "NativeBatchDecoder(name={:?}, detector_ids={:?}, observable_ids={:?})",
            self.inner.name(),
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

#[pymethods]
impl PyNativeCompositeDecoder {
    #[new]
    pub(crate) fn new(decoders: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut children = Vec::new();
        for (index, decoder) in PyIterator::from_object(decoders)?.enumerate() {
            let decoder = decoder?;
            let child = native_decoder_from_py(&decoder)?.ok_or_else(|| {
                PyTypeError::new_err(format!(
                    "NativeCompositeDecoder child {index} is not a native decoder"
                ))
            })?;
            children.push(child);
        }
        let inner = CoreNativeCompositeDecoder::new(children)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(inner),
            python_decode_calls: Arc::new(AtomicUsize::new(0)),
        })
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.detector_ids())
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.observable_ids())
    }

    pub(crate) fn decode_batch_masks(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.python_decode_calls.fetch_add(1, Ordering::Relaxed);
        decode_batch_masks_with_native_decoder(py, self.inner.clone(), batch)
    }

    #[getter]
    pub(crate) fn python_decode_call_count(&self) -> usize {
        self.python_decode_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "NativeCompositeDecoder(detector_ids={:?}, observable_ids={:?})",
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

#[pymethods]
impl PyNativeNoCorrectionDecoder {
    #[new]
    #[pyo3(signature = (observable_ids=None, detector_ids=None))]
    pub(crate) fn new(observable_ids: Option<Vec<i64>>, detector_ids: Option<Vec<i64>>) -> Self {
        Self {
            inner: Arc::new(CoreNativeNoCorrectionDecoder::with_detector_ids(
                detector_ids.unwrap_or_default(),
                observable_ids.unwrap_or_default(),
            )),
            python_decode_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.detector_ids())
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.observable_ids())
    }

    pub(crate) fn decode_batch_masks(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.python_decode_calls.fetch_add(1, Ordering::Relaxed);
        decode_batch_masks_with_native_decoder(py, self.inner.clone(), batch)
    }

    #[getter]
    pub(crate) fn python_decode_call_count(&self) -> usize {
        self.python_decode_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "NativeNoCorrectionDecoder(name={:?}, detector_ids={:?}, observable_ids={:?})",
            self.inner.name(),
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

#[pymethods]
impl PyNativeGraphlikeDetectorCopyDecoder {
    #[staticmethod]
    pub(crate) fn from_dem(py: Python<'_>, dem: PyRef<'_, PyDetectorErrorModel>) -> PyResult<Self> {
        let core_dem = dem.to_core_dem(py)?;
        Self::from_core_dem(core_dem)
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, *, detectors=None, observables=None))]
    pub(crate) fn from_circuit(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        detectors: Option<&Bound<'_, PyAny>>,
        observables: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let generator = core_dem_generator_from_circuit(py, circuit, detectors, observables)?;
        let core_dem = generator
            .generate()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Self::from_core_dem(core_dem)
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.detector_ids())
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.observable_ids())
    }

    pub(crate) fn decode_batch_masks(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.python_decode_calls.fetch_add(1, Ordering::Relaxed);
        decode_batch_masks_with_native_decoder(py, self.inner.clone(), batch)
    }

    #[getter]
    pub(crate) fn python_decode_call_count(&self) -> usize {
        self.python_decode_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "NativeGraphlikeDetectorCopyDecoder(name={:?}, detector_ids={:?}, observable_ids={:?})",
            self.inner.name(),
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

impl PyNativeGraphlikeDetectorCopyDecoder {
    fn from_core_dem(core_dem: faultscope_core::DetectorErrorModel) -> PyResult<Self> {
        let problem = core_dem
            .compile_graphlike_problem()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let backend = CoreNativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(problem)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(backend),
            python_decode_calls: Arc::new(AtomicUsize::new(0)),
        })
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
#[pymethods]
impl PyNativeFusionBlossomDecoder {
    #[staticmethod]
    #[pyo3(signature = (dem, *, options=None))]
    pub(crate) fn from_dem(
        py: Python<'_>,
        dem: PyRef<'_, PyDetectorErrorModel>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        reject_fusion_blossom_options(options)?;
        let core_dem = dem.to_core_dem(py)?;
        Self::from_core_dem(core_dem)
    }

    #[staticmethod]
    #[pyo3(signature = (circuit, *, detectors=None, observables=None, options=None))]
    pub(crate) fn from_circuit(
        py: Python<'_>,
        circuit: &Bound<'_, PyAny>,
        detectors: Option<&Bound<'_, PyAny>>,
        observables: Option<&Bound<'_, PyAny>>,
        options: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        reject_fusion_blossom_options(options)?;
        let generator = core_dem_generator_from_circuit(py, circuit, detectors, observables)?;
        let core_dem = generator
            .generate()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Self::from_core_dem(core_dem)
    }

    #[getter]
    pub(crate) fn name(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.detector_ids())
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, self.inner.observable_ids())
    }

    pub(crate) fn decode_batch_masks(
        &self,
        py: Python<'_>,
        batch: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        self.python_decode_calls.fetch_add(1, Ordering::Relaxed);
        decode_batch_masks_with_native_decoder(py, self.inner.clone(), batch)
    }

    #[getter]
    pub(crate) fn python_decode_call_count(&self) -> usize {
        self.python_decode_calls.load(Ordering::Relaxed)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "NativeFusionBlossomDecoder(name={:?}, detector_ids={:?}, observable_ids={:?})",
            self.inner.name(),
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
impl PyNativeFusionBlossomDecoder {
    fn from_core_dem(core_dem: faultscope_core::DetectorErrorModel) -> PyResult<Self> {
        let problem = core_dem
            .compile_graphlike_problem()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let backend = CoreNativeFusionBlossomDecoder::from_graphlike_problem(problem)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(backend),
            python_decode_calls: Arc::new(AtomicUsize::new(0)),
        })
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
fn reject_fusion_blossom_options(options: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    if options.is_some_and(|options| !options.is_none()) {
        return Err(PyValueError::new_err(
            "fusion-blossom options are reserved until the backend dependency is linked",
        ));
    }
    Ok(())
}

struct OwnedPyObjectPtr {
    ptr: *mut pyo3::ffi::PyObject,
}

impl OwnedPyObjectPtr {
    fn new(obj: &Bound<'_, PyAny>) -> Self {
        unsafe {
            pyo3::ffi::Py_INCREF(obj.as_ptr());
        }
        Self { ptr: obj.as_ptr() }
    }

    #[cfg(test)]
    fn noop() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }
}

impl Drop for OwnedPyObjectPtr {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        Python::with_gil(|_| unsafe {
            pyo3::ffi::Py_DECREF(self.ptr);
        });
    }
}

unsafe impl Send for OwnedPyObjectPtr {}
unsafe impl Sync for OwnedPyObjectPtr {}

struct ExternalDecoderOwner {
    descriptor: NonNull<FaultScopeNativeDecoderV1>,
    // Retained solely to keep the descriptor and prototype state alive.
    #[allow(dead_code)]
    capsule: OwnedPyObjectPtr,
}

unsafe impl Send for ExternalDecoderOwner {}
unsafe impl Sync for ExternalDecoderOwner {}

impl ExternalDecoderOwner {
    fn struct_size(&self) -> usize {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).struct_size).read() }
    }

    fn state(&self) -> *mut std::ffi::c_void {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).state).read() }
    }

    fn drop_state_callback(&self) -> Option<unsafe extern "C" fn(*mut std::ffi::c_void)> {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).drop_state).read() }
    }

    fn name_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderStringViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).name).read() }
    }

    fn detector_ids_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderI64SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).detector_ids).read() }
    }

    fn observable_ids_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderI64SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).observable_ids).read() }
    }

    fn decode_batch_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const FaultScopeNativeDetectorMaskBatchViewV1,
            *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).decode_batch).read() }
    }

    fn decode_packed_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const FaultScopeNativePackedDetectorShotBatchViewV1,
            *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        if self.struct_size()
            < mem::offset_of!(FaultScopeNativeDecoderV1, decode_detector_event_batch)
        {
            return None;
        }
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).decode_packed_batch).read() }
    }

    fn decode_detector_event_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const FaultScopeNativeDetectorEventShotBatchViewV1,
            *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        if self.struct_size() < native_decoder_v1_decode_event_field_end() {
            return None;
        }
        unsafe {
            std::ptr::addr_of!((*self.descriptor.as_ptr()).decode_detector_event_batch).read()
        }
    }

    fn create_worker_state_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut *mut std::ffi::c_void,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        if self.struct_size() < native_decoder_v1_create_worker_state_field_end() {
            return None;
        }
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).create_worker_state).read() }
    }
}

struct ExternalNativeBatchDecoder {
    owner: Arc<ExternalDecoderOwner>,
    state: NonNull<std::ffi::c_void>,
    owns_worker_state: bool,
    name: String,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

unsafe impl Send for ExternalNativeBatchDecoder {}
unsafe impl Sync for ExternalNativeBatchDecoder {}

impl ExternalNativeBatchDecoder {
    fn from_capsule(capsule: &Bound<'_, PyAny>) -> PyResult<Self> {
        let capsule_name = CString::new(NATIVE_DECODER_PLUGIN_CAPSULE_NAME)
            .expect("native decoder capsule name must not contain NUL");
        let is_valid =
            unsafe { pyo3::ffi::PyCapsule_IsValid(capsule.as_ptr(), capsule_name.as_ptr()) == 1 };
        if !is_valid {
            return Err(PyValueError::new_err(format!(
                "native decoder capsule must be named {NATIVE_DECODER_PLUGIN_CAPSULE_NAME:?}"
            )));
        }
        let pointer =
            unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule.as_ptr(), capsule_name.as_ptr()) };
        let descriptor =
            NonNull::new(pointer.cast::<FaultScopeNativeDecoderV1>()).ok_or_else(|| {
                PyValueError::new_err("native decoder capsule contained a null descriptor pointer")
            })?;
        Self::from_descriptor(descriptor, OwnedPyObjectPtr::new(capsule))
    }

    fn from_descriptor(
        descriptor: NonNull<FaultScopeNativeDecoderV1>,
        capsule: OwnedPyObjectPtr,
    ) -> PyResult<Self> {
        validate_external_decoder_descriptor(descriptor)?;
        let owner = Arc::new(ExternalDecoderOwner {
            descriptor,
            capsule,
        });
        let state = NonNull::new(owner.state()).expect("external descriptor state was validated");
        let name = unsafe { call_decoder_name(&owner, state).map_err(PyValueError::new_err)? };
        let detector_ids = unsafe {
            call_decoder_ids(state, owner.detector_ids_callback()).map_err(PyValueError::new_err)?
        };
        let observable_ids = unsafe {
            call_decoder_ids(state, owner.observable_ids_callback())
                .map_err(PyValueError::new_err)?
        };
        Ok(Self {
            owner,
            state,
            owns_worker_state: false,
            name,
            detector_ids,
            observable_ids,
        })
    }
}

impl Drop for ExternalNativeBatchDecoder {
    fn drop(&mut self) {
        if self.owns_worker_state {
            if let Some(drop_state) = self.owner.drop_state_callback() {
                unsafe {
                    drop_state(self.state.as_ptr());
                }
            }
        }
    }
}

fn native_decoder_v1_decode_event_field_end() -> usize {
    mem::offset_of!(FaultScopeNativeDecoderV1, decode_detector_event_batch)
        + mem::size_of::<
            Option<
                unsafe extern "C" fn(
                    *mut std::ffi::c_void,
                    *const FaultScopeNativeDetectorEventShotBatchViewV1,
                    *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
                ) -> FaultScopeNativeDecoderStatusV1,
            >,
        >()
}

fn native_decoder_v1_create_worker_state_field_end() -> usize {
    mem::offset_of!(FaultScopeNativeDecoderV1, create_worker_state)
        + mem::size_of::<
            Option<
                unsafe extern "C" fn(
                    *const std::ffi::c_void,
                    *mut *mut std::ffi::c_void,
                ) -> FaultScopeNativeDecoderStatusV1,
            >,
        >()
}

impl CoreNativeBatchDecoder for ExternalNativeBatchDecoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> faultscope_core::NpResult<Arc<dyn CoreNativeBatchDecoder>> {
        let create_worker_state = self.owner.create_worker_state_callback().ok_or_else(|| {
            faultscope_core::NpError::new(format!(
                "{} does not support collection worker instances",
                self.name
            ))
        })?;
        let drop_state = self.owner.drop_state_callback().ok_or_else(|| {
            faultscope_core::NpError::new(format!(
                "{} worker-state factory requires a drop_state callback",
                self.name
            ))
        })?;
        let mut worker_state = std::ptr::null_mut();
        let status =
            unsafe { create_worker_state(self.owner.state().cast_const(), &mut worker_state) };
        if let Err(error) = status_to_np_result(status) {
            if let Some(worker_state) = NonNull::new(worker_state) {
                unsafe {
                    drop_state(worker_state.as_ptr());
                }
            }
            return Err(error);
        }
        let worker_state = NonNull::new(worker_state).ok_or_else(|| {
            faultscope_core::NpError::new(format!(
                "{} worker-state factory returned a null state pointer",
                self.name
            ))
        })?;
        let worker_metadata = unsafe {
            (|| {
                let name = call_decoder_name(&self.owner, worker_state)
                    .map_err(faultscope_core::NpError::new)?;
                let detector_ids =
                    call_decoder_ids(worker_state, self.owner.detector_ids_callback())
                        .map_err(faultscope_core::NpError::new)?;
                let observable_ids =
                    call_decoder_ids(worker_state, self.owner.observable_ids_callback())
                        .map_err(faultscope_core::NpError::new)?;
                Ok::<_, faultscope_core::NpError>((name, detector_ids, observable_ids))
            })()
        };
        let (name, detector_ids, observable_ids) = match worker_metadata {
            Ok(metadata) => metadata,
            Err(error) => {
                unsafe {
                    drop_state(worker_state.as_ptr());
                }
                return Err(error);
            }
        };
        if name != self.name
            || detector_ids != self.detector_ids
            || observable_ids != self.observable_ids
        {
            unsafe {
                drop_state(worker_state.as_ptr());
            }
            return Err(faultscope_core::NpError::new(format!(
                "{} worker-state factory returned mismatched decoder metadata",
                self.name
            )));
        }
        Ok(Arc::new(Self {
            owner: Arc::clone(&self.owner),
            state: worker_state,
            owns_worker_state: true,
            name,
            detector_ids,
            observable_ids,
        }))
    }

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> faultscope_core::NpResult<CorrectionMaskBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(faultscope_core::NpError::new(format!(
                "{} received detector masks in an unexpected order",
                self.name
            )));
        }
        let word_count = faultscope_core::word_count(detectors.shots);
        let input_masks = detectors
            .masks
            .iter()
            .map(|mask| FaultScopeNativeDecoderMaskViewV1 {
                words: mask.words.as_ptr(),
                word_count: mask.words.len(),
            })
            .collect::<Vec<_>>();
        let input = FaultScopeNativeDetectorMaskBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            masks: input_masks.as_ptr(),
            shots: detectors.shots,
            word_count,
        };
        let mut output_masks = vec![Mask::zero(word_count); self.observable_ids.len()];
        let mut output_views = output_masks
            .iter_mut()
            .map(|mask| FaultScopeNativeDecoderMaskMutViewV1 {
                words: mask.words.as_mut_ptr(),
                word_count: mask.words.len(),
            })
            .collect::<Vec<_>>();
        let mut output = FaultScopeNativeCorrectionMaskBatchMutViewV1 {
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            masks: output_views.as_mut_ptr(),
            shots: detectors.shots,
            word_count,
        };
        let decode_batch = self
            .owner
            .decode_batch_callback()
            .expect("external decoder descriptor was validated");
        let status = unsafe { decode_batch(self.state.as_ptr(), &input, &mut output) };
        status_to_np_result(status)?;
        CorrectionMaskBatch::new(self.observable_ids.clone(), output_masks, detectors.shots)
    }

    fn supports_packed_batch(&self) -> bool {
        self.owner.decode_packed_callback().is_some()
    }

    fn decode_packed_batch(
        &self,
        detectors: PackedDetectorShotBatchView<'_>,
    ) -> faultscope_core::NpResult<PackedObservableShotBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(faultscope_core::NpError::new(format!(
                "{} received packed detector shots in an unexpected order",
                self.name
            )));
        }
        let decode_packed_batch = self.owner.decode_packed_callback().ok_or_else(|| {
            faultscope_core::NpError::new(format!(
                "{} does not support packed-row batch decode",
                self.name
            ))
        })?;
        let observable_byte_count = self.observable_ids.len().div_ceil(8);
        let mut output_data = vec![0; detectors.shots * observable_byte_count];
        let input = FaultScopeNativePackedDetectorShotBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            data: detectors.data.as_ptr(),
            shots: detectors.shots,
            detector_byte_count: detectors.detector_byte_count,
        };
        let mut output = FaultScopeNativePackedObservableShotBatchMutViewV1 {
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            data: output_data.as_mut_ptr(),
            shots: detectors.shots,
            observable_byte_count,
        };
        let status = unsafe { decode_packed_batch(self.state.as_ptr(), &input, &mut output) };
        status_to_np_result(status)?;
        PackedObservableShotBatch::new(self.observable_ids.clone(), output_data, detectors.shots)
    }

    fn supports_detector_event_batch(&self) -> bool {
        self.owner.decode_detector_event_callback().is_some()
    }

    fn decode_detector_event_batch(
        &self,
        detectors: DetectorEventShotBatchView<'_>,
    ) -> faultscope_core::NpResult<PackedObservableShotBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(faultscope_core::NpError::new(format!(
                "{} received detector events in an unexpected detector order",
                self.name
            )));
        }
        let decode_detector_event_batch =
            self.owner.decode_detector_event_callback().ok_or_else(|| {
                faultscope_core::NpError::new(format!(
                    "{} does not support detector-event batch decode",
                    self.name
                ))
            })?;
        let observable_byte_count = self.observable_ids.len().div_ceil(8);
        let mut output_data = vec![0; detectors.shots * observable_byte_count];
        let input = FaultScopeNativeDetectorEventShotBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            offsets: detectors.offsets.as_ptr(),
            offsets_len: detectors.offsets.len(),
            events: detectors.events.as_ptr(),
            event_count: detectors.events.len(),
            shots: detectors.shots,
        };
        let mut output = FaultScopeNativePackedObservableShotBatchMutViewV1 {
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            data: output_data.as_mut_ptr(),
            shots: detectors.shots,
            observable_byte_count,
        };
        let status =
            unsafe { decode_detector_event_batch(self.state.as_ptr(), &input, &mut output) };
        status_to_np_result(status)?;
        PackedObservableShotBatch::new(self.observable_ids.clone(), output_data, detectors.shots)
    }
}

pub(crate) fn native_decoder_from_py(
    decoder: &Bound<'_, PyAny>,
) -> PyResult<Option<Arc<dyn CoreNativeBatchDecoder>>> {
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeBatchDecoder>>() {
        return Ok(Some(decoder.inner.clone()));
    }
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeCompositeDecoder>>() {
        return Ok(Some(decoder.inner.clone()));
    }
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeNoCorrectionDecoder>>() {
        return Ok(Some(decoder.inner.clone()));
    }
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeGraphlikeDetectorCopyDecoder>>() {
        return Ok(Some(decoder.inner.clone()));
    }
    #[cfg(feature = "decoder-fusion-blossom")]
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeFusionBlossomDecoder>>() {
        return Ok(Some(decoder.inner.clone()));
    }
    match decoder.getattr(NATIVE_DECODER_PLUGIN_CAPSULE_METHOD) {
        Ok(method) => {
            let capsule = method.call0()?;
            let external = ExternalNativeBatchDecoder::from_capsule(&capsule)?;
            Ok(Some(Arc::new(external)))
        }
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(decoder.py()) => {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

fn validate_external_decoder_descriptor(
    descriptor: NonNull<FaultScopeNativeDecoderV1>,
) -> PyResult<()> {
    let abi_version = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).abi_version).read() };
    if abi_version != NATIVE_DECODER_PLUGIN_ABI_VERSION {
        return Err(PyValueError::new_err(format!(
            "native decoder plugin ABI version {} is unsupported; expected {}",
            abi_version, NATIVE_DECODER_PLUGIN_ABI_VERSION
        )));
    }
    let struct_size = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).struct_size).read() };
    let minimum_descriptor_size = mem::offset_of!(FaultScopeNativeDecoderV1, decode_packed_batch);
    if struct_size < minimum_descriptor_size {
        return Err(PyValueError::new_err(format!(
            "native decoder descriptor has size {}; expected at least {}",
            struct_size, minimum_descriptor_size
        )));
    }
    let flags = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).flags).read() };
    if flags & NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE == 0 {
        return Err(PyValueError::new_err(
            "native decoder descriptor must declare thread-safe decode callbacks",
        ));
    }
    let state = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).state).read() };
    if state.is_null() {
        return Err(PyValueError::new_err(
            "native decoder descriptor has a null state pointer",
        ));
    }
    let name = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).name).read() };
    let detector_ids = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).detector_ids).read() };
    let observable_ids =
        unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).observable_ids).read() };
    let decode_batch = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).decode_batch).read() };
    if name.is_none()
        || detector_ids.is_none()
        || observable_ids.is_none()
        || decode_batch.is_none()
    {
        return Err(PyValueError::new_err(
            "native decoder descriptor is missing required callbacks",
        ));
    }
    Ok(())
}

unsafe fn call_decoder_name(
    owner: &ExternalDecoderOwner,
    state: NonNull<std::ffi::c_void>,
) -> Result<String, String> {
    let mut out = FaultScopeNativeDecoderStringViewV1::empty();
    let callback = owner
        .name_callback()
        .expect("external decoder descriptor was validated");
    let status = callback(state.as_ptr().cast_const(), &mut out);
    status_to_decoder_result(status)?;
    string_view_to_string_result(out)
}

unsafe fn call_decoder_ids(
    state: NonNull<std::ffi::c_void>,
    callback: Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderI64SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
) -> Result<Vec<i64>, String> {
    let mut out = FaultScopeNativeDecoderI64SliceV1::empty();
    let callback = callback.expect("external decoder descriptor was validated");
    let status = callback(state.as_ptr().cast_const(), &mut out);
    status_to_decoder_result(status)?;
    if out.len == 0 {
        return Ok(Vec::new());
    }
    if out.ptr.is_null() {
        return Err("native decoder callback returned a null id pointer".to_string());
    }
    Ok(slice::from_raw_parts(out.ptr, out.len).to_vec())
}

fn status_to_decoder_result(status: FaultScopeNativeDecoderStatusV1) -> Result<(), String> {
    if status.code == NATIVE_DECODER_PLUGIN_STATUS_OK {
        return Ok(());
    }
    Err(format!(
        "native decoder plugin error: {}",
        status_message(status)
    ))
}

fn status_to_np_result(status: FaultScopeNativeDecoderStatusV1) -> faultscope_core::NpResult<()> {
    if status.code == NATIVE_DECODER_PLUGIN_STATUS_OK {
        return Ok(());
    }
    Err(faultscope_core::NpError::new(format!(
        "native decoder plugin error: {}",
        status_message(status)
    )))
}

fn status_message(status: FaultScopeNativeDecoderStatusV1) -> String {
    string_view_to_string_result(status.message)
        .unwrap_or_else(|_| format!("status code {}", status.code))
}

fn string_view_to_string_result(
    view: FaultScopeNativeDecoderStringViewV1,
) -> Result<String, String> {
    if view.len == 0 {
        return Ok(String::new());
    }
    if view.ptr.is_null() {
        return Err("native decoder callback returned a null string pointer".to_string());
    }
    let bytes = unsafe { slice::from_raw_parts(view.ptr.cast::<u8>(), view.len) };
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

pub(crate) fn detector_mask_view_from_map(
    detectors: &HashMap<i64, Mask>,
    detector_ids: &[i64],
    _shots: usize,
) -> PyResult<Vec<Mask>> {
    detector_ids
        .iter()
        .map(|detector_id| {
            detectors
                .get(detector_id)
                .cloned()
                .ok_or_else(|| PyValueError::new_err(format!("missing detector id {detector_id}")))
        })
        .collect()
}

pub(crate) fn correction_batch_to_py(
    py: Python<'_>,
    corrections: &CorrectionMaskBatch,
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    for (observable_id, mask) in corrections.observable_ids.iter().zip(&corrections.masks) {
        out.set_item(*observable_id, mask_to_py(py, mask)?)?;
    }
    Ok(out.into())
}

fn decode_batch_masks_with_native_decoder(
    py: Python<'_>,
    decoder: Arc<dyn CoreNativeBatchDecoder>,
    batch: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let shots = batch.getattr("shots")?.extract::<usize>()?;
    let detectors = batch.getattr("detectors")?;
    let detectors = detectors
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("batch.detectors must be a dict"))?;
    let detector_masks = py_detector_masks_to_vec(detectors, decoder.detector_ids(), shots)?;
    let view = DetectorMaskBatchView::new(decoder.detector_ids(), &detector_masks, shots)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let corrections = decoder
        .decode_batch_checked(view)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    correction_batch_to_py(py, &corrections)
}

fn py_detector_masks_to_vec(
    detectors: &Bound<'_, PyDict>,
    detector_ids: &[i64],
    shots: usize,
) -> PyResult<Vec<Mask>> {
    let words = faultscope_core::word_count(shots);
    detector_ids
        .iter()
        .map(|detector_id| {
            let value = detectors.get_item(*detector_id)?.ok_or_else(|| {
                PyValueError::new_err(format!("missing detector id {detector_id}"))
            })?;
            py_int_to_mask(&value, words, shots)
        })
        .collect()
}

/// Indexed detector-error-model edge used by decoder builders.
#[pyclass(name = "IndexedDemEdge", module = "faultscope._native", frozen)]
pub(crate) struct PyIndexedDemEdge {
    edge: faultscope_core::IndexedDemEdge,
}

#[pymethods]
impl PyIndexedDemEdge {
    #[getter]
    pub(crate) fn probability(&self) -> f64 {
        self.edge.probability
    }

    #[getter]
    pub(crate) fn weight(&self) -> f64 {
        self.edge.weight
    }

    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize_local(py, &self.edge.detectors)
    }

    #[getter]
    pub(crate) fn observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize_local(py, &self.edge.observables)
    }

    #[getter]
    pub(crate) fn original_edge_index(&self) -> usize {
        self.edge.original_edge_index
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "IndexedDemEdge(original_edge_index={}, detectors={:?}, observables={:?}, probability={:.6})",
            self.edge.original_edge_index,
            self.edge.detectors,
            self.edge.observables,
            self.edge.probability
        )
    }
}

/// Decoder-oriented indexed detector error model.
#[pyclass(name = "IndexedDem", module = "faultscope._native", frozen)]
pub(crate) struct PyIndexedDem {
    pub(crate) indexed: IndexedDem,
}

#[pymethods]
impl PyIndexedDem {
    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.indexed.detector_ids)
    }

    #[getter]
    pub(crate) fn detector_coords(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_f64_tuples(py, &self.indexed.detector_coords)
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.indexed.observable_ids)
    }

    #[getter]
    pub(crate) fn detector_count(&self) -> usize {
        self.indexed.detector_ids.len()
    }

    #[getter]
    pub(crate) fn observable_count(&self) -> usize {
        self.indexed.observable_ids.len()
    }

    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.indexed.edges.len()
    }

    #[getter]
    pub(crate) fn edges(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .indexed
            .edges
            .iter()
            .cloned()
            .map(|edge| Py::new(py, PyIndexedDemEdge { edge }))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?.into())
    }

    #[getter]
    pub(crate) fn edge_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        indexed_edge_summary_to_py(py, &self.indexed.edges)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "IndexedDem(detector_count={}, observable_count={}, edge_count={})",
            self.indexed.detector_ids.len(),
            self.indexed.observable_ids.len(),
            self.indexed.edges.len()
        )
    }
}

/// Graphlike decoder edge with fault-observable support.
#[pyclass(name = "GraphlikeEdge", module = "faultscope._native", frozen)]
pub(crate) struct PyGraphlikeEdge {
    edge: faultscope_core::GraphlikeEdge,
}

#[pymethods]
impl PyGraphlikeEdge {
    #[getter]
    pub(crate) fn detectors(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize_local(py, &self.edge.detectors)
    }

    #[getter]
    pub(crate) fn fault_observables(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize_local(py, &self.edge.fault_observables)
    }

    #[getter]
    pub(crate) fn probability(&self) -> f64 {
        self.edge.probability
    }

    #[getter]
    pub(crate) fn weight(&self) -> f64 {
        self.edge.weight
    }

    #[getter]
    pub(crate) fn dem_edge_index(&self) -> usize {
        self.edge.dem_edge_index
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "GraphlikeEdge(dem_edge_index={}, detectors={:?}, fault_observables={:?}, probability={:.6})",
            self.edge.dem_edge_index,
            self.edge.detectors,
            self.edge.fault_observables,
            self.edge.probability
        )
    }
}

/// Graphlike matching problem compiled from a detector error model.
#[pyclass(
    name = "GraphlikeDecodingProblem",
    module = "faultscope._native",
    frozen
)]
pub(crate) struct PyGraphlikeDecodingProblem {
    pub(crate) problem: GraphlikeDecodingProblem,
}

#[pymethods]
impl PyGraphlikeDecodingProblem {
    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.problem.detector_ids)
    }

    #[getter]
    pub(crate) fn detector_coords(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_f64_tuples(py, &self.problem.detector_coords)
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.problem.observable_ids)
    }

    #[getter]
    pub(crate) fn detector_count(&self) -> usize {
        self.problem.detector_ids.len()
    }

    #[getter]
    pub(crate) fn observable_count(&self) -> usize {
        self.problem.observable_ids.len()
    }

    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.problem.edges.len()
    }

    #[getter]
    pub(crate) fn edges(&self, py: Python<'_>) -> PyResult<PyObject> {
        let items = self
            .problem
            .edges
            .iter()
            .cloned()
            .map(|edge| Py::new(py, PyGraphlikeEdge { edge }))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?.into())
    }

    #[getter]
    pub(crate) fn edge_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        graphlike_edge_summary_to_py(py, &self.problem.edges)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "GraphlikeDecodingProblem(detector_count={}, observable_count={}, edge_count={})",
            self.problem.detector_ids.len(),
            self.problem.observable_ids.len(),
            self.problem.edges.len()
        )
    }
}

/// Sparse binary matrix represented by coordinate entries.
#[pyclass(name = "SparseBinaryMatrix", module = "faultscope._native", frozen)]
pub(crate) struct PySparseBinaryMatrix {
    matrix: SparseBinaryMatrix,
}

#[pymethods]
impl PySparseBinaryMatrix {
    #[getter]
    pub(crate) fn row_count(&self) -> usize {
        self.matrix.row_count
    }

    #[getter]
    pub(crate) fn col_count(&self) -> usize {
        self.matrix.col_count
    }

    #[getter]
    pub(crate) fn entry_count(&self) -> usize {
        self.matrix.entries.len()
    }

    #[getter]
    pub(crate) fn entries(&self, py: Python<'_>) -> PyResult<PyObject> {
        let entries = self
            .matrix
            .entries
            .iter()
            .map(|(row, col)| PyTuple::new(py, [*row, *col]))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries.iter())?.into())
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "SparseBinaryMatrix(row_count={}, col_count={}, entry_count={})",
            self.matrix.row_count,
            self.matrix.col_count,
            self.matrix.entries.len()
        )
    }
}

/// Binary-linear decoder problem compiled from a detector error model.
#[pyclass(
    name = "BinaryLinearDecodingProblem",
    module = "faultscope._native",
    frozen
)]
pub(crate) struct PyBinaryLinearDecodingProblem {
    pub(crate) problem: BinaryLinearDecodingProblem,
}

#[pymethods]
impl PyBinaryLinearDecodingProblem {
    #[getter]
    pub(crate) fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.problem.detector_ids)
    }

    #[getter]
    pub(crate) fn detector_coords(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_f64_tuples(py, &self.problem.detector_coords)
    }

    #[getter]
    pub(crate) fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_i64(py, &self.problem.observable_ids)
    }

    #[getter]
    pub(crate) fn detector_count(&self) -> usize {
        self.problem.detector_ids.len()
    }

    #[getter]
    pub(crate) fn observable_count(&self) -> usize {
        self.problem.observable_ids.len()
    }

    #[getter]
    pub(crate) fn edge_count(&self) -> usize {
        self.problem.edge_count
    }

    #[getter]
    pub(crate) fn h(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(Py::new(
            py,
            PySparseBinaryMatrix {
                matrix: self.problem.h.clone(),
            },
        )?
        .into_any())
    }

    #[getter]
    pub(crate) fn f(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(Py::new(
            py,
            PySparseBinaryMatrix {
                matrix: self.problem.f.clone(),
            },
        )?
        .into_any())
    }

    #[getter]
    pub(crate) fn probabilities(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.problem.probabilities.iter().copied())?.into())
    }

    #[getter]
    pub(crate) fn log_likelihood_ratios(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.problem.log_likelihood_ratios.iter().copied())?.into())
    }

    #[getter]
    pub(crate) fn dem_edge_indices(&self, py: Python<'_>) -> PyResult<PyObject> {
        tuple_usize_local(py, &self.problem.dem_edge_indices)
    }

    #[getter]
    pub(crate) fn edge_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        binary_edge_summary_to_py(py, &self.problem)
    }

    pub(crate) fn __repr__(&self) -> String {
        format!(
            "BinaryLinearDecodingProblem(detector_count={}, observable_count={}, edge_count={})",
            self.problem.detector_ids.len(),
            self.problem.observable_ids.len(),
            self.problem.edge_count
        )
    }
}

fn tuple_i64(py: Python<'_>, values: &[i64]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}

fn tuple_f64_tuples(py: Python<'_>, values: &[Vec<f64>]) -> PyResult<PyObject> {
    let items = values
        .iter()
        .map(|coords| PyTuple::new(py, coords.iter().copied()).map(|item| item.into_any().unbind()))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?.into())
}

fn tuple_usize_local(py: Python<'_>, values: &[usize]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}

fn indexed_edge_summary_to_py(
    py: Python<'_>,
    edges: &[faultscope_core::IndexedDemEdge],
) -> PyResult<PyObject> {
    let rows = edges
        .iter()
        .map(|edge| {
            let row = PyDict::new(py);
            row.set_item("dem_edge_index", edge.original_edge_index)?;
            row.set_item(
                "detectors",
                PyTuple::new(py, edge.detectors.iter().copied())?,
            )?;
            row.set_item(
                "observables",
                PyTuple::new(py, edge.observables.iter().copied())?,
            )?;
            row.set_item("probability", edge.probability)?;
            row.set_item("weight", edge.weight)?;
            Ok(row.into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    tuple_py_objects(py, rows)
}

fn graphlike_edge_summary_to_py(
    py: Python<'_>,
    edges: &[faultscope_core::GraphlikeEdge],
) -> PyResult<PyObject> {
    let rows = edges
        .iter()
        .map(|edge| {
            let row = PyDict::new(py);
            row.set_item("dem_edge_index", edge.dem_edge_index)?;
            row.set_item(
                "detectors",
                PyTuple::new(py, edge.detectors.iter().copied())?,
            )?;
            row.set_item(
                "fault_observables",
                PyTuple::new(py, edge.fault_observables.iter().copied())?,
            )?;
            row.set_item("probability", edge.probability)?;
            row.set_item("weight", edge.weight)?;
            Ok(row.into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    tuple_py_objects(py, rows)
}

fn binary_edge_summary_to_py(
    py: Python<'_>,
    problem: &BinaryLinearDecodingProblem,
) -> PyResult<PyObject> {
    let rows = problem
        .dem_edge_indices
        .iter()
        .enumerate()
        .map(|(edge_index, dem_edge_index)| {
            let row = PyDict::new(py);
            row.set_item("edge_index", edge_index)?;
            row.set_item("dem_edge_index", *dem_edge_index)?;
            row.set_item("probability", problem.probabilities[edge_index])?;
            row.set_item(
                "log_likelihood_ratio",
                problem.log_likelihood_ratios[edge_index],
            )?;
            Ok(row.into())
        })
        .collect::<PyResult<Vec<PyObject>>>()?;
    tuple_py_objects(py, rows)
}

fn tuple_py_objects(py: Python<'_>, items: Vec<PyObject>) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, items.iter().map(|item| item.clone_ref(py)))?.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::alloc::{alloc, dealloc, Layout};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const TEST_NAME: &str = "test-external";
    const TEST_METADATA_ERROR: &str = "test metadata failure";
    const TEST_DETECTOR_IDS: [i64; 1] = [7];
    const TEST_OBSERVABLE_IDS: [i64; 1] = [11];

    struct TestState {
        next_worker: AtomicU64,
        value: AtomicU64,
        drops: Arc<AtomicUsize>,
        worker_pointers: Arc<Mutex<Vec<usize>>>,
        metadata_error: bool,
    }

    impl TestState {
        fn prototype(drops: Arc<AtomicUsize>, worker_pointers: Arc<Mutex<Vec<usize>>>) -> Self {
            Self {
                next_worker: AtomicU64::new(0),
                value: AtomicU64::new(0),
                drops,
                worker_pointers,
                metadata_error: false,
            }
        }
    }

    unsafe extern "C" fn test_drop_state(state: *mut c_void) {
        if !state.is_null() {
            let state = Box::from_raw(state.cast::<TestState>());
            state.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    unsafe extern "C" fn test_name(
        _state: *const c_void,
        out: *mut FaultScopeNativeDecoderStringViewV1,
    ) -> FaultScopeNativeDecoderStatusV1 {
        *out = FaultScopeNativeDecoderStringViewV1 {
            ptr: TEST_NAME.as_ptr().cast(),
            len: TEST_NAME.len(),
        };
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_detector_ids(
        state: *const c_void,
        out: *mut FaultScopeNativeDecoderI64SliceV1,
    ) -> FaultScopeNativeDecoderStatusV1 {
        if (*state.cast::<TestState>()).metadata_error {
            return FaultScopeNativeDecoderStatusV1 {
                code: faultscope_core::NATIVE_DECODER_PLUGIN_STATUS_ERROR,
                message: FaultScopeNativeDecoderStringViewV1 {
                    ptr: TEST_METADATA_ERROR.as_ptr().cast(),
                    len: TEST_METADATA_ERROR.len(),
                },
            };
        }
        *out = FaultScopeNativeDecoderI64SliceV1 {
            ptr: TEST_DETECTOR_IDS.as_ptr(),
            len: TEST_DETECTOR_IDS.len(),
        };
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_observable_ids(
        _state: *const c_void,
        out: *mut FaultScopeNativeDecoderI64SliceV1,
    ) -> FaultScopeNativeDecoderStatusV1 {
        *out = FaultScopeNativeDecoderI64SliceV1 {
            ptr: TEST_OBSERVABLE_IDS.as_ptr(),
            len: TEST_OBSERVABLE_IDS.len(),
        };
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_decode_batch(
        state: *mut c_void,
        _input: *const FaultScopeNativeDetectorMaskBatchViewV1,
        output: *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
    ) -> FaultScopeNativeDecoderStatusV1 {
        let state = &*state.cast::<TestState>();
        let value = state.value.fetch_add(1, Ordering::SeqCst) + 1;
        let output = &mut *output;
        *(*output.masks).words = value;
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_create_worker_state(
        factory_state: *const c_void,
        out_state: *mut *mut c_void,
    ) -> FaultScopeNativeDecoderStatusV1 {
        let factory = &*factory_state.cast::<TestState>();
        let worker_id = factory.next_worker.fetch_add(1, Ordering::SeqCst) + 1;
        let worker = Box::new(TestState {
            next_worker: AtomicU64::new(0),
            value: AtomicU64::new(worker_id * 10),
            drops: Arc::clone(&factory.drops),
            worker_pointers: Arc::clone(&factory.worker_pointers),
            metadata_error: false,
        });
        let worker = Box::into_raw(worker).cast::<c_void>();
        factory
            .worker_pointers
            .lock()
            .expect("worker pointer mutex poisoned")
            .push(worker as usize);
        *out_state = worker;
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_create_null_worker_state(
        _factory_state: *const c_void,
        _out_state: *mut *mut c_void,
    ) -> FaultScopeNativeDecoderStatusV1 {
        FaultScopeNativeDecoderStatusV1::ok()
    }

    unsafe extern "C" fn test_create_metadata_error_worker_state(
        factory_state: *const c_void,
        out_state: *mut *mut c_void,
    ) -> FaultScopeNativeDecoderStatusV1 {
        let factory = &*factory_state.cast::<TestState>();
        let worker = Box::new(TestState {
            next_worker: AtomicU64::new(0),
            value: AtomicU64::new(0),
            drops: Arc::clone(&factory.drops),
            worker_pointers: Arc::clone(&factory.worker_pointers),
            metadata_error: true,
        });
        *out_state = Box::into_raw(worker).cast::<c_void>();
        FaultScopeNativeDecoderStatusV1::ok()
    }

    fn test_descriptor(
        state: *mut c_void,
        struct_size: usize,
        create_worker_state: Option<
            unsafe extern "C" fn(
                *const c_void,
                *mut *mut c_void,
            ) -> FaultScopeNativeDecoderStatusV1,
        >,
    ) -> FaultScopeNativeDecoderV1 {
        FaultScopeNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size,
            flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
            state,
            drop_state: Some(test_drop_state),
            name: Some(test_name),
            detector_ids: Some(test_detector_ids),
            observable_ids: Some(test_observable_ids),
            decode_batch: Some(test_decode_batch),
            decode_packed_batch: None,
            decode_detector_event_batch: None,
            create_worker_state,
        }
    }

    unsafe fn external_from_test_descriptor(
        descriptor: *mut FaultScopeNativeDecoderV1,
    ) -> ExternalNativeBatchDecoder {
        let descriptor = NonNull::new(descriptor).expect("test descriptor must not be null");
        ExternalNativeBatchDecoder::from_descriptor(descriptor, OwnedPyObjectPtr::noop()).unwrap()
    }

    unsafe fn drop_test_descriptor(descriptor: *mut FaultScopeNativeDecoderV1) {
        let descriptor = Box::from_raw(descriptor);
        descriptor.drop_state.unwrap()(descriptor.state);
    }

    #[test]
    fn external_worker_instance_legacy_descriptor_is_unsupported() {
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_pointers = Arc::new(Mutex::new(Vec::new()));
        let state = Box::into_raw(Box::new(TestState::prototype(
            Arc::clone(&drops),
            worker_pointers,
        )))
        .cast::<c_void>();
        let legacy_size = mem::offset_of!(FaultScopeNativeDecoderV1, create_worker_state);
        let legacy_descriptor = test_descriptor(state, legacy_size, None);
        let layout =
            Layout::from_size_align(legacy_size, mem::align_of::<FaultScopeNativeDecoderV1>())
                .unwrap();
        let descriptor = unsafe { alloc(layout).cast::<FaultScopeNativeDecoderV1>() };
        assert!(!descriptor.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&legacy_descriptor as *const FaultScopeNativeDecoderV1).cast::<u8>(),
                descriptor.cast::<u8>(),
                legacy_size,
            );
        }
        let decoder = unsafe { external_from_test_descriptor(descriptor) };

        let error = match decoder.create_worker_instance() {
            Ok(_) => panic!("legacy external descriptor unexpectedly created a worker"),
            Err(error) => error,
        };

        assert_eq!(
            error.to_string(),
            "test-external does not support collection worker instances"
        );
        drop(decoder);
        unsafe {
            test_drop_state(state);
            dealloc(descriptor.cast::<u8>(), layout);
        }
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn external_worker_instances_are_independent_and_drop_once() {
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_pointers = Arc::new(Mutex::new(Vec::new()));
        let state = Box::into_raw(Box::new(TestState::prototype(
            Arc::clone(&drops),
            Arc::clone(&worker_pointers),
        )))
        .cast::<c_void>();
        let descriptor = Box::into_raw(Box::new(test_descriptor(
            state,
            mem::size_of::<FaultScopeNativeDecoderV1>(),
            Some(test_create_worker_state),
        )));
        let decoder = unsafe { external_from_test_descriptor(descriptor) };

        let first = decoder.create_worker_instance().unwrap();
        let second = decoder.create_worker_instance().unwrap();
        let third = first.create_worker_instance().unwrap();
        let masks = [Mask { words: vec![1] }];
        let first_view = DetectorMaskBatchView::new(first.detector_ids(), &masks, 1).unwrap();
        let second_view = DetectorMaskBatchView::new(second.detector_ids(), &masks, 1).unwrap();
        let third_view = DetectorMaskBatchView::new(third.detector_ids(), &masks, 1).unwrap();
        assert_eq!(first.decode_batch(first_view).unwrap().masks[0].words, [11]);
        let first_view = DetectorMaskBatchView::new(first.detector_ids(), &masks, 1).unwrap();
        assert_eq!(first.decode_batch(first_view).unwrap().masks[0].words, [12]);
        assert_eq!(
            second.decode_batch(second_view).unwrap().masks[0].words,
            [21]
        );
        assert_eq!(third.decode_batch(third_view).unwrap().masks[0].words, [31]);
        let pointers = worker_pointers.lock().unwrap();
        assert_eq!(pointers.len(), 3);
        assert_ne!(pointers[0], pointers[1]);
        assert_ne!(pointers[0], pointers[2]);
        assert_ne!(pointers[1], pointers[2]);
        drop(pointers);

        drop(first);
        drop(second);
        drop(third);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
        drop(decoder);
        unsafe { drop_test_descriptor(descriptor) };
        assert_eq!(drops.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn external_worker_instance_rejects_null_factory_state() {
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_pointers = Arc::new(Mutex::new(Vec::new()));
        let state = Box::into_raw(Box::new(TestState::prototype(
            Arc::clone(&drops),
            worker_pointers,
        )))
        .cast::<c_void>();
        let descriptor = Box::into_raw(Box::new(test_descriptor(
            state,
            mem::size_of::<FaultScopeNativeDecoderV1>(),
            Some(test_create_null_worker_state),
        )));
        let decoder = unsafe { external_from_test_descriptor(descriptor) };

        let error = match decoder.create_worker_instance() {
            Ok(_) => panic!("null worker-state factory unexpectedly created a worker"),
            Err(error) => error,
        };

        assert_eq!(
            error.to_string(),
            "test-external worker-state factory returned a null state pointer"
        );
        drop(decoder);
        unsafe { drop_test_descriptor(descriptor) };
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn external_worker_instance_drops_state_when_metadata_validation_fails() {
        let drops = Arc::new(AtomicUsize::new(0));
        let worker_pointers = Arc::new(Mutex::new(Vec::new()));
        let state = Box::into_raw(Box::new(TestState::prototype(
            Arc::clone(&drops),
            worker_pointers,
        )))
        .cast::<c_void>();
        let descriptor = Box::into_raw(Box::new(test_descriptor(
            state,
            mem::size_of::<FaultScopeNativeDecoderV1>(),
            Some(test_create_metadata_error_worker_state),
        )));
        let decoder = unsafe { external_from_test_descriptor(descriptor) };

        let error = match decoder.create_worker_instance() {
            Ok(_) => panic!("invalid worker metadata unexpectedly created a worker"),
            Err(error) => error,
        };

        assert!(error.to_string().contains(TEST_METADATA_ERROR));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(decoder);
        unsafe { drop_test_descriptor(descriptor) };
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }
}
