use crate::*;
#[cfg(feature = "decoder-fusion-blossom")]
use npsim_core::NativeFusionBlossomDecoder as CoreNativeFusionBlossomDecoder;
use npsim_core::{
    BinaryLinearDecodingProblem, CorrectionMaskBatch, DetectorMaskBatchView,
    GraphlikeDecodingProblem, IndexedDem, NativeBatchDecoder as CoreNativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder as CoreNativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder as CoreNativeNoCorrectionDecoder,
    NpsimNativeCorrectionMaskBatchMutViewV1, NpsimNativeDecoderI64SliceV1,
    NpsimNativeDecoderMaskMutViewV1, NpsimNativeDecoderMaskViewV1, NpsimNativeDecoderStatusV1,
    NpsimNativeDecoderStringViewV1, NpsimNativeDecoderV1, NpsimNativeDetectorMaskBatchViewV1,
    SparseBinaryMatrix, NATIVE_DECODER_PLUGIN_ABI_VERSION, NATIVE_DECODER_PLUGIN_CAPSULE_METHOD,
    NATIVE_DECODER_PLUGIN_CAPSULE_NAME, NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
    NATIVE_DECODER_PLUGIN_STATUS_OK,
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

#[pyclass(name = "NativeBatchDecoder", module = "npsim._npsim_native")]
pub(crate) struct PyNativeBatchDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[pyclass(name = "NativeNoCorrectionDecoder", module = "npsim._npsim_native")]
pub(crate) struct PyNativeNoCorrectionDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[pyclass(
    name = "NativeGraphlikeDetectorCopyDecoder",
    module = "npsim._npsim_native"
)]
pub(crate) struct PyNativeGraphlikeDetectorCopyDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
}

#[cfg(feature = "decoder-fusion-blossom")]
#[pyclass(name = "NativeFusionBlossomDecoder", module = "npsim._npsim_native")]
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
    fn from_core_dem(core_dem: npsim_core::DetectorErrorModel) -> PyResult<Self> {
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
    fn from_core_dem(core_dem: npsim_core::DetectorErrorModel) -> PyResult<Self> {
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
        Python::with_gil(|_| unsafe {
            pyo3::ffi::Py_DECREF(self.ptr);
        });
    }
}

unsafe impl Send for OwnedPyObjectPtr {}
unsafe impl Sync for OwnedPyObjectPtr {}

struct ExternalNativeBatchDecoder {
    descriptor: NonNull<NpsimNativeDecoderV1>,
    _capsule: OwnedPyObjectPtr,
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
        let descriptor = NonNull::new(pointer.cast::<NpsimNativeDecoderV1>()).ok_or_else(|| {
            PyValueError::new_err("native decoder capsule contained a null descriptor pointer")
        })?;
        let descriptor_ref = unsafe { descriptor.as_ref() };
        validate_external_decoder_descriptor(descriptor_ref)?;
        let name = unsafe { call_decoder_name(descriptor_ref)? };
        let detector_ids =
            unsafe { call_decoder_ids(descriptor_ref, descriptor_ref.detector_ids)? };
        let observable_ids =
            unsafe { call_decoder_ids(descriptor_ref, descriptor_ref.observable_ids)? };
        Ok(Self {
            descriptor,
            _capsule: OwnedPyObjectPtr::new(capsule),
            name,
            detector_ids,
            observable_ids,
        })
    }

    fn descriptor(&self) -> &NpsimNativeDecoderV1 {
        unsafe { self.descriptor.as_ref() }
    }
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

    fn decode_batch(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> npsim_core::NpResult<CorrectionMaskBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(npsim_core::NpError::new(format!(
                "{} received detector masks in an unexpected order",
                self.name
            )));
        }
        let word_count = npsim_core::word_count(detectors.shots);
        let input_masks = detectors
            .masks
            .iter()
            .map(|mask| NpsimNativeDecoderMaskViewV1 {
                words: mask.words.as_ptr(),
                word_count: mask.words.len(),
            })
            .collect::<Vec<_>>();
        let input = NpsimNativeDetectorMaskBatchViewV1 {
            detector_ids: self.detector_ids.as_ptr(),
            detector_count: self.detector_ids.len(),
            masks: input_masks.as_ptr(),
            shots: detectors.shots,
            word_count,
        };
        let mut output_masks = vec![Mask::zero(word_count); self.observable_ids.len()];
        let mut output_views = output_masks
            .iter_mut()
            .map(|mask| NpsimNativeDecoderMaskMutViewV1 {
                words: mask.words.as_mut_ptr(),
                word_count: mask.words.len(),
            })
            .collect::<Vec<_>>();
        let mut output = NpsimNativeCorrectionMaskBatchMutViewV1 {
            observable_ids: self.observable_ids.as_ptr(),
            observable_count: self.observable_ids.len(),
            masks: output_views.as_mut_ptr(),
            shots: detectors.shots,
            word_count,
        };
        let descriptor = self.descriptor();
        let decode_batch = descriptor
            .decode_batch
            .expect("external decoder descriptor was validated");
        let status = unsafe { decode_batch(descriptor.state, &input, &mut output) };
        status_to_np_result(status)?;
        CorrectionMaskBatch::new(self.observable_ids.clone(), output_masks, detectors.shots)
    }
}

pub(crate) fn native_decoder_from_py(
    decoder: &Bound<'_, PyAny>,
) -> PyResult<Option<Arc<dyn CoreNativeBatchDecoder>>> {
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeBatchDecoder>>() {
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

fn validate_external_decoder_descriptor(descriptor: &NpsimNativeDecoderV1) -> PyResult<()> {
    if descriptor.abi_version != NATIVE_DECODER_PLUGIN_ABI_VERSION {
        return Err(PyValueError::new_err(format!(
            "native decoder plugin ABI version {} is unsupported; expected {}",
            descriptor.abi_version, NATIVE_DECODER_PLUGIN_ABI_VERSION
        )));
    }
    if descriptor.struct_size < mem::size_of::<NpsimNativeDecoderV1>() {
        return Err(PyValueError::new_err(format!(
            "native decoder descriptor has size {}; expected at least {}",
            descriptor.struct_size,
            mem::size_of::<NpsimNativeDecoderV1>()
        )));
    }
    if descriptor.flags & NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE == 0 {
        return Err(PyValueError::new_err(
            "native decoder descriptor must declare thread-safe decode callbacks",
        ));
    }
    if descriptor.state.is_null() {
        return Err(PyValueError::new_err(
            "native decoder descriptor has a null state pointer",
        ));
    }
    if descriptor.name.is_none()
        || descriptor.detector_ids.is_none()
        || descriptor.observable_ids.is_none()
        || descriptor.decode_batch.is_none()
    {
        return Err(PyValueError::new_err(
            "native decoder descriptor is missing required callbacks",
        ));
    }
    Ok(())
}

unsafe fn call_decoder_name(descriptor: &NpsimNativeDecoderV1) -> PyResult<String> {
    let mut out = NpsimNativeDecoderStringViewV1::empty();
    let callback = descriptor
        .name
        .expect("external decoder descriptor was validated");
    let status = callback(descriptor.state.cast_const(), &mut out);
    status_to_py_result(status)?;
    string_view_to_string(out)
}

unsafe fn call_decoder_ids(
    descriptor: &NpsimNativeDecoderV1,
    callback: Option<
        unsafe extern "C" fn(
            *const std::ffi::c_void,
            *mut NpsimNativeDecoderI64SliceV1,
        ) -> NpsimNativeDecoderStatusV1,
    >,
) -> PyResult<Vec<i64>> {
    let mut out = NpsimNativeDecoderI64SliceV1::empty();
    let callback = callback.expect("external decoder descriptor was validated");
    let status = callback(descriptor.state.cast_const(), &mut out);
    status_to_py_result(status)?;
    if out.len == 0 {
        return Ok(Vec::new());
    }
    if out.ptr.is_null() {
        return Err(PyValueError::new_err(
            "native decoder callback returned a null id pointer",
        ));
    }
    Ok(slice::from_raw_parts(out.ptr, out.len).to_vec())
}

fn status_to_py_result(status: NpsimNativeDecoderStatusV1) -> PyResult<()> {
    if status.code == NATIVE_DECODER_PLUGIN_STATUS_OK {
        return Ok(());
    }
    Err(PyValueError::new_err(format!(
        "native decoder plugin error: {}",
        status_message(status)
    )))
}

fn status_to_np_result(status: NpsimNativeDecoderStatusV1) -> npsim_core::NpResult<()> {
    if status.code == NATIVE_DECODER_PLUGIN_STATUS_OK {
        return Ok(());
    }
    Err(npsim_core::NpError::new(format!(
        "native decoder plugin error: {}",
        status_message(status)
    )))
}

fn status_message(status: NpsimNativeDecoderStatusV1) -> String {
    string_view_to_string(status.message).unwrap_or_else(|_| format!("status code {}", status.code))
}

fn string_view_to_string(view: NpsimNativeDecoderStringViewV1) -> PyResult<String> {
    if view.len == 0 {
        return Ok(String::new());
    }
    if view.ptr.is_null() {
        return Err(PyValueError::new_err(
            "native decoder callback returned a null string pointer",
        ));
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
    let words = npsim_core::word_count(shots);
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

#[pyclass(name = "IndexedDemEdge", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyIndexedDemEdge {
    edge: npsim_core::IndexedDemEdge,
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

#[pyclass(name = "IndexedDem", module = "npsim._npsim_native", frozen)]
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

#[pyclass(name = "GraphlikeEdge", module = "npsim._npsim_native", frozen)]
pub(crate) struct PyGraphlikeEdge {
    edge: npsim_core::GraphlikeEdge,
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

#[pyclass(
    name = "GraphlikeDecodingProblem",
    module = "npsim._npsim_native",
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

#[pyclass(name = "SparseBinaryMatrix", module = "npsim._npsim_native", frozen)]
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

#[pyclass(
    name = "BinaryLinearDecodingProblem",
    module = "npsim._npsim_native",
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

fn tuple_usize_local(py: Python<'_>, values: &[usize]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}

fn indexed_edge_summary_to_py(
    py: Python<'_>,
    edges: &[npsim_core::IndexedDemEdge],
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
    edges: &[npsim_core::GraphlikeEdge],
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
