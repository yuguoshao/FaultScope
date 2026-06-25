use npsim_core::{
    NpsimNativeCorrectionMaskBatchMutViewV1, NpsimNativeDecoderI64SliceV1,
    NpsimNativeDecoderStatusV1, NpsimNativeDecoderStringViewV1, NpsimNativeDecoderV1,
    NpsimNativeDetectorMaskBatchViewV1, NATIVE_DECODER_PLUGIN_ABI_VERSION,
    NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE, NATIVE_DECODER_PLUGIN_STATUS_ERROR,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyTuple};
use std::ffi::{c_void, CString};
use std::mem;
use std::os::raw::c_char;
use std::ptr;
use std::slice;
use std::sync::Mutex;

const BACKEND_NAME: &str = "fusion-blossom";
const CAPSULE_NAME: &[u8] = b"npsim.native_decoder_plugin.v1\0";

#[pyclass(name = "NativeFusionBlossomNativeDecoder")]
struct PyNativeFusionBlossomNativeDecoder {
    capsule: Py<PyAny>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_detector_indices: Vec<Option<usize>>,
}

#[pymethods]
impl PyNativeFusionBlossomNativeDecoder {
    #[staticmethod]
    fn from_graphlike_problem(py: Python<'_>, problem: &Bound<'_, PyAny>) -> PyResult<Self> {
        let detector_ids = problem.getattr("detector_ids")?.extract::<Vec<i64>>()?;
        let observable_ids = problem.getattr("observable_ids")?.extract::<Vec<i64>>()?;
        let mut observable_detector_indices = vec![None; observable_ids.len()];

        let edges = problem.getattr("edges")?;
        for edge in PyIterator::from_object(&edges)? {
            let edge = edge?;
            let detectors = edge.getattr("detectors")?.extract::<Vec<usize>>()?;
            let fault_observables = edge.getattr("fault_observables")?.extract::<Vec<usize>>()?;
            if detectors.len() != 1 || fault_observables.len() != 1 {
                continue;
            }
            let detector_index = detectors[0];
            let observable_index = fault_observables[0];
            if detector_index >= detector_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom scaffold edge references detector index {detector_index} but only {} detectors exist",
                    detector_ids.len()
                )));
            }
            if observable_index >= observable_ids.len() {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom scaffold edge references observable index {observable_index} but only {} observables exist",
                    observable_ids.len()
                )));
            }
            if let Some(existing_detector_index) = observable_detector_indices[observable_index] {
                return Err(PyValueError::new_err(format!(
                    "fusion-blossom scaffold found multiple single-detector candidate edges for observable id {}; detector indices {} and {}",
                    observable_ids[observable_index],
                    existing_detector_index,
                    detector_index
                )));
            }
            observable_detector_indices[observable_index] = Some(detector_index);
        }

        let state = Box::new(DecoderState {
            detector_ids: detector_ids.clone(),
            observable_ids: observable_ids.clone(),
            observable_detector_indices: observable_detector_indices.clone(),
            last_error: Mutex::new(CString::new("").expect("empty CString")),
        });
        let descriptor = Box::new(NpsimNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size: mem::size_of::<NpsimNativeDecoderV1>(),
            flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
            state: Box::into_raw(state).cast::<c_void>(),
            drop_state: Some(drop_state),
            name: Some(decoder_name),
            detector_ids: Some(decoder_detector_ids),
            observable_ids: Some(decoder_observable_ids),
            decode_batch: Some(decoder_decode_batch),
        });
        let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(descriptor))? };

        Ok(Self {
            capsule,
            detector_ids,
            observable_ids,
            observable_detector_indices,
        })
    }

    #[getter]
    fn name(&self) -> &'static str {
        BACKEND_NAME
    }

    #[getter]
    fn detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.detector_ids.iter().copied())?.into())
    }

    #[getter]
    fn observable_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyTuple::new(py, self.observable_ids.iter().copied())?.into())
    }

    #[getter]
    fn observable_detector_ids(&self, py: Python<'_>) -> PyResult<PyObject> {
        let ids = self
            .observable_detector_indices
            .iter()
            .map(|index| index.map(|index| self.detector_ids[index]))
            .collect::<Vec<_>>();
        Ok(PyTuple::new(py, ids)?.into())
    }

    fn __npsim_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "NativeFusionBlossomNativeDecoder(name={:?}, detector_ids={:?}, observable_ids={:?})",
            BACKEND_NAME, self.detector_ids, self.observable_ids
        )
    }
}

#[pyclass(name = "InvalidNativeDecoderCapsule")]
struct PyInvalidNativeDecoderCapsule {
    capsule: Py<PyAny>,
}

#[pymethods]
impl PyInvalidNativeDecoderCapsule {
    #[new]
    fn new(py: Python<'_>, kind: &str) -> PyResult<Self> {
        let state = Box::new(DecoderState {
            detector_ids: vec![0],
            observable_ids: vec![0],
            observable_detector_indices: vec![None],
            last_error: Mutex::new(CString::new("").expect("empty CString")),
        });
        let mut descriptor = NpsimNativeDecoderV1 {
            abi_version: NATIVE_DECODER_PLUGIN_ABI_VERSION,
            struct_size: mem::size_of::<NpsimNativeDecoderV1>(),
            flags: NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE,
            state: Box::into_raw(state).cast::<c_void>(),
            drop_state: Some(drop_state),
            name: Some(decoder_name),
            detector_ids: Some(decoder_detector_ids),
            observable_ids: Some(decoder_observable_ids),
            decode_batch: Some(decoder_decode_batch),
        };
        match kind {
            "abi-mismatch" => descriptor.abi_version = NATIVE_DECODER_PLUGIN_ABI_VERSION + 1,
            "missing-callback" => descriptor.decode_batch = None,
            "not-thread-safe" => descriptor.flags = 0,
            "decode-error" => descriptor.decode_batch = Some(forced_error_decode_batch),
            _ => {
                unsafe {
                    drop_descriptor(Box::into_raw(Box::new(descriptor)));
                }
                return Err(PyValueError::new_err(format!(
                    "unknown invalid native decoder kind {kind:?}"
                )));
            }
        }
        let capsule = unsafe { create_decoder_capsule(py, Box::into_raw(Box::new(descriptor)))? };
        Ok(Self { capsule })
    }

    fn __npsim_native_decoder_capsule__(&self, py: Python<'_>) -> Py<PyAny> {
        self.capsule.clone_ref(py)
    }
}

struct DecoderState {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_detector_indices: Vec<Option<usize>>,
    last_error: Mutex<CString>,
}

unsafe extern "C" fn forced_error_decode_batch(
    state: *mut c_void,
    _input: *const NpsimNativeDetectorMaskBatchViewV1,
    _output: *mut NpsimNativeCorrectionMaskBatchMutViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("forced error decoder received null state pointer");
    }
    state_error(
        &*state.cast::<DecoderState>(),
        "forced native decoder decode failure",
    )
}

unsafe extern "C" fn drop_state(state: *mut c_void) {
    if !state.is_null() {
        drop(Box::from_raw(state.cast::<DecoderState>()));
    }
}

unsafe extern "C" fn decoder_name(
    state: *const c_void,
    out: *mut NpsimNativeDecoderStringViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("fusion-blossom name callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom name callback received null state pointer");
    }
    *out = NpsimNativeDecoderStringViewV1 {
        ptr: BACKEND_NAME.as_ptr().cast::<c_char>(),
        len: BACKEND_NAME.len(),
    };
    NpsimNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_detector_ids(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
) -> NpsimNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.detector_ids)
}

unsafe extern "C" fn decoder_observable_ids(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
) -> NpsimNativeDecoderStatusV1 {
    ids_callback(state, out, |state| &state.observable_ids)
}

unsafe fn ids_callback(
    state: *const c_void,
    out: *mut NpsimNativeDecoderI64SliceV1,
    ids: impl FnOnce(&DecoderState) -> &[i64],
) -> NpsimNativeDecoderStatusV1 {
    if out.is_null() {
        return static_error("fusion-blossom ids callback received null output pointer");
    }
    if state.is_null() {
        return static_error("fusion-blossom ids callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    let ids = ids(state);
    *out = NpsimNativeDecoderI64SliceV1 {
        ptr: ids.as_ptr(),
        len: ids.len(),
    };
    NpsimNativeDecoderStatusV1::ok()
}

unsafe extern "C" fn decoder_decode_batch(
    state: *mut c_void,
    input: *const NpsimNativeDetectorMaskBatchViewV1,
    output: *mut NpsimNativeCorrectionMaskBatchMutViewV1,
) -> NpsimNativeDecoderStatusV1 {
    if state.is_null() {
        return static_error("fusion-blossom decode callback received null state pointer");
    }
    let state = &*state.cast::<DecoderState>();
    if input.is_null() {
        return state_error(
            state,
            "fusion-blossom decode callback received null input pointer",
        );
    }
    if output.is_null() {
        return state_error(
            state,
            "fusion-blossom decode callback received null output pointer",
        );
    }
    let input = &*input;
    let output = &mut *output;
    if input.detector_count != state.detector_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} detector masks but received {}",
                state.detector_ids.len(),
                input.detector_count
            ),
        );
    }
    if output.observable_count != state.observable_ids.len() {
        return state_error(
            state,
            format!(
                "fusion-blossom expected {} correction masks but received {}",
                state.observable_ids.len(),
                output.observable_count
            ),
        );
    }
    if input.word_count != output.word_count {
        return state_error(
            state,
            "fusion-blossom input and output word counts do not match",
        );
    }
    if input.detector_count > 0 && input.masks.is_null() {
        return state_error(state, "fusion-blossom input masks pointer is null");
    }
    if output.observable_count > 0 && output.masks.is_null() {
        return state_error(state, "fusion-blossom output masks pointer is null");
    }
    let input_masks = slice::from_raw_parts(input.masks, input.detector_count);
    let output_masks = slice::from_raw_parts_mut(output.masks, output.observable_count);
    for (observable_index, detector_index) in state.observable_detector_indices.iter().enumerate() {
        let Some(detector_index) = detector_index else {
            continue;
        };
        let Some(input_mask) = input_masks.get(*detector_index) else {
            return state_error(
                state,
                format!("fusion-blossom missing detector mask at index {detector_index}"),
            );
        };
        let output_mask = &mut output_masks[observable_index];
        if input_mask.word_count != input.word_count || output_mask.word_count != output.word_count
        {
            return state_error(
                state,
                "fusion-blossom mask word count does not match batch word count",
            );
        }
        if input.word_count > 0 && input_mask.words.is_null() {
            return state_error(state, "fusion-blossom input mask words pointer is null");
        }
        if output.word_count > 0 && output_mask.words.is_null() {
            return state_error(state, "fusion-blossom output mask words pointer is null");
        }
        ptr::copy_nonoverlapping(input_mask.words, output_mask.words, input.word_count);
    }
    NpsimNativeDecoderStatusV1::ok()
}

unsafe fn create_decoder_capsule(
    py: Python<'_>,
    descriptor: *mut NpsimNativeDecoderV1,
) -> PyResult<Py<PyAny>> {
    let ptr = pyo3::ffi::PyCapsule_New(
        descriptor.cast::<c_void>(),
        CAPSULE_NAME.as_ptr().cast::<c_char>(),
        Some(capsule_destructor),
    );
    if ptr.is_null() {
        drop_descriptor(descriptor);
        return Err(PyErr::fetch(py));
    }
    Ok(Py::from_owned_ptr(py, ptr))
}

unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, CAPSULE_NAME.as_ptr().cast::<c_char>());
    if !pointer.is_null() {
        drop_descriptor(pointer.cast::<NpsimNativeDecoderV1>());
    }
}

unsafe fn drop_descriptor(descriptor: *mut NpsimNativeDecoderV1) {
    if descriptor.is_null() {
        return;
    }
    let descriptor = Box::from_raw(descriptor);
    if let Some(drop_state) = descriptor.drop_state {
        drop_state(descriptor.state);
    }
}

fn static_error(message: &'static str) -> NpsimNativeDecoderStatusV1 {
    NpsimNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: NpsimNativeDecoderStringViewV1 {
            ptr: message.as_ptr().cast::<c_char>(),
            len: message.len(),
        },
    }
}

fn state_error(state: &DecoderState, message: impl Into<String>) -> NpsimNativeDecoderStatusV1 {
    let sanitized = message.into().replace('\0', "\\0");
    let mut last_error = state
        .last_error
        .lock()
        .expect("fusion-blossom error mutex poisoned");
    *last_error = CString::new(sanitized).expect("NUL was sanitized");
    NpsimNativeDecoderStatusV1 {
        code: NATIVE_DECODER_PLUGIN_STATUS_ERROR,
        message: NpsimNativeDecoderStringViewV1 {
            ptr: last_error.as_ptr(),
            len: last_error.as_bytes().len(),
        },
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeFusionBlossomNativeDecoder>()?;
    module.add_class::<PyInvalidNativeDecoderCapsule>()?;
    Ok(())
}
