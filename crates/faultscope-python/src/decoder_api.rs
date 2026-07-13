use crate::*;
#[cfg(feature = "decoder-fusion-blossom")]
use faultscope_core::NativeFusionBlossomDecoder as CoreNativeFusionBlossomDecoder;
use faultscope_core::{
    BinaryLinearDecodingProblem, CorrectionMaskBatch, DetectorEventShotBatchView,
    DetectorMaskBatchView, FaultScopeNativeCorrectionMaskBatchMutViewV1,
    FaultScopeNativeDecoderFactoryV2, FaultScopeNativeDecoderI64SliceV1,
    FaultScopeNativeDecoderMaskMutViewV1, FaultScopeNativeDecoderMaskViewV1,
    FaultScopeNativeDecoderStatusV1, FaultScopeNativeDecoderStringViewV1,
    FaultScopeNativeDecoderWorkerV2, FaultScopeNativeDetectorEventShotBatchViewV1,
    FaultScopeNativeDetectorMaskBatchViewV1, FaultScopeNativePackedDetectorShotBatchViewV1,
    FaultScopeNativePackedObservableShotBatchMutViewV1, GraphlikeDecodingProblem, IndexedDem,
    NativeCompositeDecoder as CoreNativeCompositeDecoder,
    NativeDecoderFactory as CoreNativeDecoderFactory,
    NativeDecoderWorker as CoreNativeDecoderWorker,
    NativeGraphlikeDetectorCopyDecoder as CoreNativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder as CoreNativeNoCorrectionDecoder, PackedDetectorShotBatchView,
    PackedObservableShotBatch, SparseBinaryMatrix, NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE,
    NATIVE_DECODER_PLUGIN_ABI_VERSION, NATIVE_DECODER_PLUGIN_CAPSULE_METHOD,
    NATIVE_DECODER_PLUGIN_CAPSULE_NAME, NATIVE_DECODER_PLUGIN_STATUS_OK,
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

/// Type-erased native decoder factory handle.
#[pyclass(name = "NativeBatchDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeBatchDecoder {
    pub(crate) inner: Arc<dyn CoreNativeDecoderFactory>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native decoder that composes independent child decoders.
#[pyclass(name = "NativeCompositeDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeCompositeDecoder {
    pub(crate) inner: Arc<dyn CoreNativeDecoderFactory>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native decoder that always returns an empty correction.
#[pyclass(name = "NativeNoCorrectionDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeNoCorrectionDecoder {
    pub(crate) inner: Arc<dyn CoreNativeDecoderFactory>,
    python_decode_calls: Arc<AtomicUsize>,
}

/// Native graphlike decoder that copies detector masks to observables.
#[pyclass(
    name = "NativeGraphlikeDetectorCopyDecoder",
    module = "faultscope._native"
)]
pub(crate) struct PyNativeGraphlikeDetectorCopyDecoder {
    pub(crate) inner: Arc<dyn CoreNativeDecoderFactory>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[cfg(feature = "decoder-fusion-blossom")]
#[pyclass(name = "NativeFusionBlossomDecoder", module = "faultscope._native")]
pub(crate) struct PyNativeFusionBlossomDecoder {
    pub(crate) inner: Arc<dyn CoreNativeDecoderFactory>,
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
    descriptor: NonNull<FaultScopeNativeDecoderFactoryV2>,
    // Retained solely to keep the descriptor and factory state alive.
    #[allow(dead_code)]
    capsule: OwnedPyObjectPtr,
}

unsafe impl Send for ExternalDecoderOwner {}
unsafe impl Sync for ExternalDecoderOwner {}

impl ExternalDecoderOwner {
    fn state(&self) -> *mut std::ffi::c_void {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).factory_state).read() }
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

    fn create_worker_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderWorkerV2,
            usize,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        unsafe { std::ptr::addr_of!((*self.descriptor.as_ptr()).create_worker).read() }
    }
}

struct ExternalNativeDecoderFactory {
    owner: Arc<ExternalDecoderOwner>,
    name: String,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

unsafe impl Send for ExternalNativeDecoderFactory {}
unsafe impl Sync for ExternalNativeDecoderFactory {}

impl ExternalNativeDecoderFactory {
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
        let descriptor = NonNull::new(pointer.cast::<FaultScopeNativeDecoderFactoryV2>())
            .ok_or_else(|| {
                PyValueError::new_err("native decoder capsule contained a null descriptor pointer")
            })?;
        Self::from_descriptor(descriptor, OwnedPyObjectPtr::new(capsule))
    }

    fn from_descriptor(
        descriptor: NonNull<FaultScopeNativeDecoderFactoryV2>,
        capsule: OwnedPyObjectPtr,
    ) -> PyResult<Self> {
        validate_external_factory_descriptor(descriptor)?;
        let owner = Arc::new(ExternalDecoderOwner {
            descriptor,
            capsule,
        });
        let state = NonNull::new(owner.state()).expect("external factory state was validated");
        let name = unsafe {
            call_decoder_name(state, owner.name_callback()).map_err(PyValueError::new_err)?
        };
        let detector_ids = unsafe {
            call_decoder_ids(state, owner.detector_ids_callback()).map_err(PyValueError::new_err)?
        };
        let observable_ids = unsafe {
            call_decoder_ids(state, owner.observable_ids_callback())
                .map_err(PyValueError::new_err)?
        };
        Ok(Self {
            owner,
            name,
            detector_ids,
            observable_ids,
        })
    }

    fn clean_failed_worker(
        &self,
        worker: &mut FaultScopeNativeDecoderWorkerV2,
        error: faultscope_core::NpError,
    ) -> faultscope_core::NpError {
        let Some(worker_state) = NonNull::new(worker.worker_state) else {
            return error;
        };
        let Some(drop_worker_state) = worker.drop_worker_state else {
            return faultscope_core::NpError::new(format!(
                "{}; ABI v2 factory {} returned unsafe non-null partial worker state without a drop callback",
                error, self.name
            ));
        };
        unsafe {
            drop_worker_state(worker_state.as_ptr());
        }
        worker.worker_state = std::ptr::null_mut();
        error
    }

    fn validate_metadata_still_matches(&self) -> faultscope_core::NpResult<()> {
        let state = NonNull::new(self.owner.state())
            .expect("external factory state was validated and remains owned");
        let name = unsafe {
            call_decoder_name(state, self.owner.name_callback())
                .map_err(faultscope_core::NpError::new)?
        };
        let detector_ids = unsafe {
            call_decoder_ids(state, self.owner.detector_ids_callback())
                .map_err(faultscope_core::NpError::new)?
        };
        let observable_ids = unsafe {
            call_decoder_ids(state, self.owner.observable_ids_callback())
                .map_err(faultscope_core::NpError::new)?
        };
        if name != self.name
            || detector_ids != self.detector_ids
            || observable_ids != self.observable_ids
        {
            return Err(faultscope_core::NpError::new(format!(
                "ABI v2 factory {} returned inconsistent decoder metadata while creating a worker",
                self.name
            )));
        }
        Ok(())
    }
}

impl CoreNativeDecoderFactory for ExternalNativeDecoderFactory {
    fn name(&self) -> &str {
        &self.name
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker(&self) -> faultscope_core::NpResult<Box<dyn CoreNativeDecoderWorker>> {
        let mut worker = FaultScopeNativeDecoderWorkerV2 {
            struct_size: mem::size_of::<FaultScopeNativeDecoderWorkerV2>(),
            worker_state: std::ptr::null_mut(),
            drop_worker_state: None,
            decode_batch: None,
            decode_packed_batch: None,
            decode_detector_event_batch: None,
        };
        let create_worker = self
            .owner
            .create_worker_callback()
            .expect("external factory descriptor was validated");
        let capacity = mem::size_of::<FaultScopeNativeDecoderWorkerV2>();
        let status =
            unsafe { create_worker(self.owner.state().cast_const(), &mut worker, capacity) };
        if let Err(error) = status_to_np_result(status) {
            return Err(self.clean_failed_worker(&mut worker, error));
        }
        if let Err(error) = validate_external_worker_descriptor(&worker, &self.name) {
            return Err(self.clean_failed_worker(&mut worker, error));
        }
        if let Err(error) = self.validate_metadata_still_matches() {
            return Err(self.clean_failed_worker(&mut worker, error));
        }
        Ok(Box::new(ExternalNativeDecoderWorker {
            _owner: Arc::clone(&self.owner),
            worker,
            name: self.name.clone(),
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
    }
}

struct ExternalNativeDecoderWorker {
    _owner: Arc<ExternalDecoderOwner>,
    worker: FaultScopeNativeDecoderWorkerV2,
    name: String,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

unsafe impl Send for ExternalNativeDecoderWorker {}

impl Drop for ExternalNativeDecoderWorker {
    fn drop(&mut self) {
        if let (Some(worker_state), Some(drop_worker_state)) = (
            NonNull::new(self.worker.worker_state),
            self.worker.drop_worker_state,
        ) {
            unsafe {
                drop_worker_state(worker_state.as_ptr());
            }
            self.worker.worker_state = std::ptr::null_mut();
        }
    }
}

impl ExternalNativeDecoderWorker {
    fn decode_packed_callback(
        &self,
    ) -> Option<
        unsafe extern "C" fn(
            *mut std::ffi::c_void,
            *const FaultScopeNativePackedDetectorShotBatchViewV1,
            *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    > {
        if self.worker.struct_size < external_worker_packed_field_end() {
            return None;
        }
        self.worker.decode_packed_batch
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
        if self.worker.struct_size < external_worker_event_field_end() {
            return None;
        }
        self.worker.decode_detector_event_batch
    }
}

impl CoreNativeDecoderWorker for ExternalNativeDecoderWorker {
    fn name(&self) -> &str {
        &self.name
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(
        &mut self,
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
            .worker
            .decode_batch
            .expect("external worker descriptor was validated");
        let expected_masks = output_views.as_mut_ptr();
        let status = unsafe { decode_batch(self.worker.worker_state, &input, &mut output) };
        status_to_np_result(status)?;
        if output.observable_ids != self.observable_ids.as_ptr()
            || output.observable_count != self.observable_ids.len()
            || output.masks != expected_masks
            || output.shots != detectors.shots
            || output.word_count != word_count
            || output_views.iter().zip(&output_masks).any(|(view, mask)| {
                view.words != mask.words.as_ptr().cast_mut() || view.word_count != mask.words.len()
            })
        {
            return Err(faultscope_core::NpError::new(format!(
                "ABI v2 worker {} returned an invalid mask output shape",
                self.name
            )));
        }
        CorrectionMaskBatch::new(self.observable_ids.clone(), output_masks, detectors.shots)
    }

    fn supports_packed_batch(&self) -> bool {
        self.decode_packed_callback().is_some()
    }

    fn decode_packed_batch(
        &mut self,
        detectors: PackedDetectorShotBatchView<'_>,
    ) -> faultscope_core::NpResult<PackedObservableShotBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(faultscope_core::NpError::new(format!(
                "{} received packed detector shots in an unexpected order",
                self.name
            )));
        }
        let decode_packed_batch = self.decode_packed_callback().ok_or_else(|| {
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
        let expected_data = output_data.as_mut_ptr();
        let status = unsafe { decode_packed_batch(self.worker.worker_state, &input, &mut output) };
        status_to_np_result(status)?;
        if output.observable_ids != self.observable_ids.as_ptr()
            || output.observable_count != self.observable_ids.len()
            || output.data != expected_data
            || output.shots != detectors.shots
            || output.observable_byte_count != observable_byte_count
        {
            return Err(faultscope_core::NpError::new(format!(
                "ABI v2 worker {} returned an invalid packed-row output shape",
                self.name
            )));
        }
        PackedObservableShotBatch::new(self.observable_ids.clone(), output_data, detectors.shots)
    }

    fn supports_detector_event_batch(&self) -> bool {
        self.decode_detector_event_callback().is_some()
    }

    fn decode_detector_event_batch(
        &mut self,
        detectors: DetectorEventShotBatchView<'_>,
    ) -> faultscope_core::NpResult<PackedObservableShotBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(faultscope_core::NpError::new(format!(
                "{} received detector events in an unexpected detector order",
                self.name
            )));
        }
        let decode_detector_event_batch =
            self.decode_detector_event_callback().ok_or_else(|| {
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
        let expected_data = output_data.as_mut_ptr();
        let status =
            unsafe { decode_detector_event_batch(self.worker.worker_state, &input, &mut output) };
        status_to_np_result(status)?;
        if output.observable_ids != self.observable_ids.as_ptr()
            || output.observable_count != self.observable_ids.len()
            || output.data != expected_data
            || output.shots != detectors.shots
            || output.observable_byte_count != observable_byte_count
        {
            return Err(faultscope_core::NpError::new(format!(
                "ABI v2 worker {} returned an invalid detector-event output shape",
                self.name
            )));
        }
        PackedObservableShotBatch::new(self.observable_ids.clone(), output_data, detectors.shots)
    }
}

pub(crate) fn native_decoder_from_py(
    decoder: &Bound<'_, PyAny>,
) -> PyResult<Option<Arc<dyn CoreNativeDecoderFactory>>> {
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
            let external = ExternalNativeDecoderFactory::from_capsule(&capsule)?;
            Ok(Some(Arc::new(external)))
        }
        Err(err) if err.is_instance_of::<pyo3::exceptions::PyAttributeError>(decoder.py()) => {
            Ok(None)
        }
        Err(err) => Err(err),
    }
}

fn validate_external_factory_descriptor(
    descriptor: NonNull<FaultScopeNativeDecoderFactoryV2>,
) -> PyResult<()> {
    let abi_version = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).abi_version).read() };
    if abi_version != NATIVE_DECODER_PLUGIN_ABI_VERSION {
        return Err(PyValueError::new_err(format!(
            "native decoder factory ABI version {} is unsupported; expected ABI v{}",
            abi_version, NATIVE_DECODER_PLUGIN_ABI_VERSION
        )));
    }
    let struct_size = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).struct_size).read() };
    let minimum_descriptor_size = mem::size_of::<FaultScopeNativeDecoderFactoryV2>();
    if struct_size < minimum_descriptor_size {
        return Err(PyValueError::new_err(format!(
            "ABI v2 native decoder factory descriptor has size {}; expected at least {}",
            struct_size, minimum_descriptor_size
        )));
    }
    let flags = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).flags).read() };
    if flags & NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE == 0 {
        return Err(PyValueError::new_err(
            "ABI v2 native decoder factory must declare the thread-safe factory flag",
        ));
    }
    let state = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).factory_state).read() };
    if state.is_null() {
        return Err(PyValueError::new_err(
            "ABI v2 native decoder factory has a null factory state pointer",
        ));
    }
    let drop_factory_state =
        unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).drop_factory_state).read() };
    let name = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).name).read() };
    let detector_ids = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).detector_ids).read() };
    let observable_ids =
        unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).observable_ids).read() };
    let create_worker = unsafe { std::ptr::addr_of!((*descriptor.as_ptr()).create_worker).read() };
    if drop_factory_state.is_none()
        || name.is_none()
        || detector_ids.is_none()
        || observable_ids.is_none()
        || create_worker.is_none()
    {
        return Err(PyValueError::new_err(
            "ABI v2 native decoder factory is missing required callbacks",
        ));
    }
    Ok(())
}

fn external_worker_required_field_end() -> usize {
    mem::offset_of!(FaultScopeNativeDecoderWorkerV2, decode_packed_batch)
}

fn external_worker_packed_field_end() -> usize {
    mem::offset_of!(FaultScopeNativeDecoderWorkerV2, decode_detector_event_batch)
}

fn external_worker_event_field_end() -> usize {
    mem::size_of::<FaultScopeNativeDecoderWorkerV2>()
}

fn validate_external_worker_descriptor(
    worker: &FaultScopeNativeDecoderWorkerV2,
    factory_name: &str,
) -> faultscope_core::NpResult<()> {
    let capacity = mem::size_of::<FaultScopeNativeDecoderWorkerV2>();
    if worker.struct_size < external_worker_required_field_end() || worker.struct_size > capacity {
        return Err(faultscope_core::NpError::new(format!(
            "ABI v2 factory {factory_name} returned worker descriptor size {}; expected {}..={capacity}",
            worker.struct_size,
            external_worker_required_field_end(),
        )));
    }
    if worker.worker_state.is_null() {
        return Err(faultscope_core::NpError::new(format!(
            "ABI v2 factory {factory_name} returned a null worker state pointer"
        )));
    }
    if worker.drop_worker_state.is_none() || worker.decode_batch.is_none() {
        return Err(faultscope_core::NpError::new(format!(
            "ABI v2 factory {factory_name} returned a worker missing required callbacks"
        )));
    }
    Ok(())
}

unsafe fn call_decoder_name(
    state: NonNull<std::ffi::c_void>,
    callback: Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut FaultScopeNativeDecoderStringViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
) -> Result<String, String> {
    let mut out = FaultScopeNativeDecoderStringViewV1::empty();
    let callback = callback.expect("external factory descriptor was validated");
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
    decoder: Arc<dyn CoreNativeDecoderFactory>,
    batch: &Bound<'_, PyAny>,
) -> PyResult<PyObject> {
    let shots = batch.getattr("shots")?.extract::<usize>()?;
    let detectors = batch.getattr("detectors")?;
    let detectors = detectors
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("batch.detectors must be a dict"))?;
    let mut worker = decoder
        .create_worker()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let detector_ids = decoder.detector_ids().to_vec();
    let detector_masks = py_detector_masks_to_vec(detectors, &detector_ids, shots)?;
    let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, shots)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let corrections = worker
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
