use crate::*;
#[cfg(feature = "decoder-fusion-blossom")]
use npsim_core::NativeFusionBlossomDecoder as CoreNativeFusionBlossomDecoder;
use npsim_core::{
    BinaryLinearDecodingProblem, CorrectionMaskBatch, DetectorMaskBatchView,
    GraphlikeDecodingProblem, IndexedDem, NativeBatchDecoder as CoreNativeBatchDecoder,
    NativeGraphlikeDetectorCopyDecoder as CoreNativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder as CoreNativeNoCorrectionDecoder, SparseBinaryMatrix,
};
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

pub(crate) fn native_decoder_from_py(
    decoder: &Bound<'_, PyAny>,
) -> Option<Arc<dyn CoreNativeBatchDecoder>> {
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeBatchDecoder>>() {
        return Some(decoder.inner.clone());
    }
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeNoCorrectionDecoder>>() {
        return Some(decoder.inner.clone());
    }
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeGraphlikeDetectorCopyDecoder>>() {
        return Some(decoder.inner.clone());
    }
    #[cfg(feature = "decoder-fusion-blossom")]
    if let Ok(decoder) = decoder.extract::<PyRef<'_, PyNativeFusionBlossomDecoder>>() {
        return Some(decoder.inner.clone());
    }
    None
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
