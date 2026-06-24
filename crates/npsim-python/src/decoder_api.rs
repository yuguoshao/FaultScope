use crate::*;
use npsim_core::{
    BinaryLinearDecodingProblem, CorrectionMaskBatch, DetectorMaskBatchView,
    GraphlikeDecodingProblem, IndexedDem, NativeBatchDecoder as CoreNativeBatchDecoder,
    NativeNoCorrectionDecoder as CoreNativeNoCorrectionDecoder, SparseBinaryMatrix,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[pyclass(name = "NativeBatchDecoder", module = "npsim._npsim_native")]
pub(crate) struct PyNativeBatchDecoder {
    pub(crate) inner: Arc<dyn CoreNativeBatchDecoder>,
    python_decode_calls: Arc<AtomicUsize>,
    name: String,
}

#[pyclass(name = "NativeNoCorrectionDecoder", module = "npsim._npsim_native")]
pub(crate) struct PyNativeNoCorrectionDecoder {
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
            name: "NativeNoCorrectionDecoder".to_string(),
        }
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
            "{}(detector_ids={:?}, observable_ids={:?})",
            self.name,
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
}

#[pymethods]
impl PyNativeNoCorrectionDecoder {
    #[new]
    #[pyo3(signature = (observable_ids=None, detector_ids=None))]
    pub(crate) fn new(
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
            "NativeNoCorrectionDecoder(detector_ids={:?}, observable_ids={:?})",
            self.inner.detector_ids(),
            self.inner.observable_ids(),
        )
    }
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
        .decode_batch(view)
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
            let value = detectors
                .get_item(*detector_id)?
                .ok_or_else(|| PyValueError::new_err(format!("missing detector id {detector_id}")))?;
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
}

#[pyclass(name = "GraphlikeDecodingProblem", module = "npsim._npsim_native", frozen)]
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
    pub(crate) fn entries(&self, py: Python<'_>) -> PyResult<PyObject> {
        let entries = self
            .matrix
            .entries
            .iter()
            .map(|(row, col)| PyTuple::new(py, [*row, *col]))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, entries.iter())?.into())
    }
}

#[pyclass(name = "BinaryLinearDecodingProblem", module = "npsim._npsim_native", frozen)]
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
        .into_any()
        )
    }

    #[getter]
    pub(crate) fn f(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(Py::new(
            py,
            PySparseBinaryMatrix {
                matrix: self.problem.f.clone(),
            },
        )?
        .into_any()
        )
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
}

fn tuple_i64(py: Python<'_>, values: &[i64]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}

fn tuple_usize_local(py: Python<'_>, values: &[usize]) -> PyResult<PyObject> {
    Ok(PyTuple::new(py, values.iter().copied())?.into())
}
