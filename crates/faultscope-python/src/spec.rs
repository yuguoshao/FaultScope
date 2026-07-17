use crate::*;
use faultscope_core::{
    Detector as CoreDetector, LogicalObservable as CoreLogicalObservable,
    NoiseLocation as CoreNoiseLocation,
};

pub(crate) use faultscope_core::{NoiseModel, TagValue};

pub(crate) type NoiseLocationSpec = CoreNoiseLocation;
pub(crate) type DemDetectorSpec = CoreDetector;
pub(crate) type DemObservableSpec = CoreLogicalObservable;
pub(crate) type Op = faultscope_core::Operation;
pub(crate) type DemEdgeSpec = faultscope_core::DetectorErrorEdge;

pub(crate) fn parse_core_circuit_object(
    value: &Bound<'_, PyAny>,
) -> PyResult<faultscope_core::Circuit> {
    if let Ok(circuit) = value.extract::<PyRef<'_, PyCircuit>>() {
        if let Some(core_circuit) = &circuit.core_circuit {
            return Ok(core_circuit.circuit().clone());
        }
        let py = value.py();
        let operations = circuit
            .operations
            .iter()
            .map(|operation| parse_operation_object(operation.bind(py)))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(faultscope_core::Circuit {
            n_qubits: circuit.n_qubits,
            operations,
        });
    }
    let (n_qubits, operations) = parse_circuit_object(value)?;
    Ok(faultscope_core::Circuit {
        n_qubits,
        operations,
    })
}

pub(crate) fn parse_circuit_object(value: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<Op>)> {
    let n_qubits = required_attr(value, "n_qubits", "Circuit")?.extract::<usize>()?;
    let operations = parse_operation_sequence(&required_attr(value, "operations", "Circuit")?)?;
    Ok((n_qubits, operations))
}

pub(crate) fn parse_py_noise_locations(
    circuit: &Bound<'_, PyAny>,
    program: &faultscope_core::SamplerProgram,
) -> PyResult<Vec<Py<PyAny>>> {
    let locations = circuit.call_method0("noise_locations")?;
    let locations = locations
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Circuit.noise_locations() must return a dict"))?;
    let mut out = Vec::with_capacity(program.noise_locations().len());
    for location in program.noise_locations() {
        let label = program.location_catalog().label(location.location_id);
        let value = locations.get_item(label)?.ok_or_else(|| {
            PyValueError::new_err(format!(
                "Circuit.noise_locations() did not return compiled location {label:?}"
            ))
        })?;
        out.push(value.unbind());
    }
    Ok(out)
}

pub(crate) fn parse_operation_sequence(value: &Bound<'_, PyAny>) -> PyResult<Vec<Op>> {
    let py = value.py();
    let items = value
        .extract::<Vec<Py<PyAny>>>()
        .map_err(|_| PyValueError::new_err("Circuit.operations must be a sequence"))?;
    items
        .iter()
        .map(|item| parse_operation_object(item.bind(py)))
        .collect()
}

pub(crate) fn parse_operation_object(value: &Bound<'_, PyAny>) -> PyResult<Op> {
    if let Ok(operation) = value.extract::<PyRef<'_, PyOperation>>() {
        return parse_native_operation_object(value.py(), &operation);
    }
    let kind = required_attr(value, "kind", "Operation")?.extract::<String>()?;
    let qubits = required_attr(value, "qubits", "Operation")?.extract::<Vec<usize>>()?;
    match kind.as_str() {
        "tick" => Ok(Op::Tick),
        "shift_coords" => {
            let metadata = required_attr(value, "metadata", "Operation")?;
            let offsets = required_mapping_item(&metadata, "offsets", "shift_coords")?
                .extract::<Vec<f64>>()?;
            Ok(Op::ShiftCoords(offsets))
        }
        "repeat" => {
            let count = required_attr(value, "repeat_count", "Operation")?.extract::<usize>()?;
            if count == 0 {
                return Err(PyValueError::new_err("repeat count must be positive"));
            }
            let body = parse_operation_sequence(&required_attr(value, "body", "Operation")?)?;
            Ok(Op::Repeat { count, body })
        }
        "h" => Ok(Op::H(one_qubit(&qubits, "h")?)),
        "s" => Ok(Op::S(one_qubit(&qubits, "s")?)),
        "s_dag" => Ok(Op::SDag(one_qubit(&qubits, "s_dag")?)),
        "cx" => {
            let (a, b) = two_qubits(&qubits, "cx")?;
            Ok(Op::Cx(a, b))
        }
        "cz" => {
            let (a, b) = two_qubits(&qubits, "cz")?;
            Ok(Op::Cz(a, b))
        }
        "swap" => {
            let (a, b) = two_qubits(&qubits, "swap")?;
            Ok(Op::Swap(a, b))
        }
        "pauli" => Ok(Op::Pauli {
            qubits,
            pauli: required_attr(value, "pauli", "Operation")?.extract::<String>()?,
        }),
        "noise" => {
            let location = required_attr(value, "noise_location", "Operation")?;
            if location.is_none() {
                return Err(PyValueError::new_err(
                    "noise operation requires noise_location",
                ));
            }
            Ok(Op::Noise(parse_noise_location_object(&location)?))
        }
        "measure" => Ok(Op::Measure {
            qubit: one_qubit(&qubits, "measure")?,
            key: optional_string_attr(value, "key")?,
            basis: required_attr(value, "basis", "Operation")?
                .extract::<String>()?
                .to_uppercase(),
            noise: optional_noise_location_attr(value, "noise_location")?,
        }),
        "measure_pauli" => Ok(Op::MeasurePauli {
            qubits,
            pauli: required_attr(value, "pauli", "Operation")?.extract::<String>()?,
            key: optional_string_attr(value, "key")?,
            noise: optional_noise_location_attr(value, "noise_location")?,
        }),
        "measure_reset" => Ok(Op::MeasureReset {
            qubit: one_qubit(&qubits, "measure_reset")?,
            basis: required_attr(value, "basis", "Operation")?
                .extract::<String>()?
                .to_uppercase(),
        }),
        "reset" => Ok(Op::Reset {
            qubit: one_qubit(&qubits, "reset")?,
            key: optional_string_attr(value, "key")?,
            basis: required_attr(value, "basis", "Operation")?
                .extract::<String>()?
                .to_uppercase(),
        }),
        "detector" => {
            let metadata = required_attr(value, "metadata", "Operation")?;
            let metadata = metadata
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(item) if !item.is_none() => Some(item.extract::<i64>()?),
                _ => None,
            };
            let coords = match metadata.get_item("coords")? {
                Some(item) if !item.is_none() => item.extract::<Vec<f64>>()?,
                _ => Vec::new(),
            };
            Ok(Op::Detector {
                detector_id,
                measurement_keys: required_attr(value, "measurement_keys", "Operation")?
                    .extract::<Vec<String>>()?,
                coords,
            })
        }
        "detector_rec" => {
            let metadata = required_attr(value, "metadata", "Operation")?;
            let metadata = metadata
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(item) if !item.is_none() => Some(item.extract::<i64>()?),
                _ => None,
            };
            let coords = match metadata.get_item("coords")? {
                Some(item) if !item.is_none() => item.extract::<Vec<f64>>()?,
                _ => Vec::new(),
            };
            let lookbacks =
                required_attr(value, "record_lookbacks", "Operation")?.extract::<Vec<usize>>()?;
            validate_record_lookbacks(&lookbacks)?;
            Ok(Op::DetectorRec {
                detector_id,
                lookbacks,
                coords,
            })
        }
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: required_attr(value, "observable_id", "Operation")?.extract::<i64>()?,
            measurement_keys: required_attr(value, "measurement_keys", "Operation")?
                .extract::<Vec<String>>()?,
        }),
        "observable_include_rec" => {
            let lookbacks =
                required_attr(value, "record_lookbacks", "Operation")?.extract::<Vec<usize>>()?;
            validate_record_lookbacks(&lookbacks)?;
            Ok(Op::ObservableIncludeRec {
                observable_id: required_attr(value, "observable_id", "Operation")?
                    .extract::<i64>()?,
                lookbacks,
            })
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
}

fn parse_native_operation_object(py: Python<'_>, operation: &PyOperation) -> PyResult<Op> {
    if let Some(core_op) = &operation.core_op {
        return Ok(core_op.clone());
    }
    match operation.kind.as_str() {
        "tick" => Ok(Op::Tick),
        "shift_coords" => {
            let metadata = operation
                .metadata
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
            let offsets = metadata
                .get_item("offsets")?
                .ok_or_else(|| PyValueError::new_err("shift_coords requires offsets"))?
                .extract::<Vec<f64>>()?;
            Ok(Op::ShiftCoords(offsets))
        }
        "repeat" => {
            let count = operation
                .repeat_count
                .ok_or_else(|| PyValueError::new_err("repeat requires repeat_count"))?;
            if count == 0 {
                return Err(PyValueError::new_err("repeat count must be positive"));
            }
            let body = operation
                .body
                .iter()
                .map(|item| parse_operation_object(item.bind(py)))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(Op::Repeat { count, body })
        }
        "h" => Ok(Op::H(one_qubit(&operation.qubits, "h")?)),
        "s" => Ok(Op::S(one_qubit(&operation.qubits, "s")?)),
        "s_dag" => Ok(Op::SDag(one_qubit(&operation.qubits, "s_dag")?)),
        "cx" => {
            let (a, b) = two_qubits(&operation.qubits, "cx")?;
            Ok(Op::Cx(a, b))
        }
        "cz" => {
            let (a, b) = two_qubits(&operation.qubits, "cz")?;
            Ok(Op::Cz(a, b))
        }
        "swap" => {
            let (a, b) = two_qubits(&operation.qubits, "swap")?;
            Ok(Op::Swap(a, b))
        }
        "pauli" => Ok(Op::Pauli {
            qubits: operation.qubits.clone(),
            pauli: operation
                .pauli
                .clone()
                .ok_or_else(|| PyValueError::new_err("pauli operation requires pauli"))?,
        }),
        "noise" => {
            let location = operation
                .noise_location
                .as_ref()
                .ok_or_else(|| PyValueError::new_err("noise operation requires noise_location"))?;
            Ok(Op::Noise(parse_noise_location_object(location.bind(py))?))
        }
        "measure" => Ok(Op::Measure {
            qubit: one_qubit(&operation.qubits, "measure")?,
            key: operation.key.clone(),
            basis: operation.basis.to_uppercase(),
            noise: optional_native_noise_location(py, &operation.noise_location)?,
        }),
        "measure_pauli" => Ok(Op::MeasurePauli {
            qubits: operation.qubits.clone(),
            pauli: operation
                .pauli
                .clone()
                .ok_or_else(|| PyValueError::new_err("measure_pauli operation requires pauli"))?,
            key: operation.key.clone(),
            noise: optional_native_noise_location(py, &operation.noise_location)?,
        }),
        "measure_reset" => Ok(Op::MeasureReset {
            qubit: one_qubit(&operation.qubits, "measure_reset")?,
            basis: operation.basis.to_uppercase(),
        }),
        "reset" => Ok(Op::Reset {
            qubit: one_qubit(&operation.qubits, "reset")?,
            key: operation.key.clone(),
            basis: operation.basis.to_uppercase(),
        }),
        "detector" => {
            let metadata = operation
                .metadata
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(item) if !item.is_none() => Some(item.extract::<i64>()?),
                _ => None,
            };
            let coords = match metadata.get_item("coords")? {
                Some(item) if !item.is_none() => item.extract::<Vec<f64>>()?,
                _ => Vec::new(),
            };
            Ok(Op::Detector {
                detector_id,
                measurement_keys: operation.measurement_keys.clone(),
                coords,
            })
        }
        "detector_rec" => {
            validate_record_lookbacks(&operation.record_lookbacks)?;
            let metadata = operation
                .metadata
                .bind(py)
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(item) if !item.is_none() => Some(item.extract::<i64>()?),
                _ => None,
            };
            let coords = match metadata.get_item("coords")? {
                Some(item) if !item.is_none() => item.extract::<Vec<f64>>()?,
                _ => Vec::new(),
            };
            Ok(Op::DetectorRec {
                detector_id,
                lookbacks: operation.record_lookbacks.clone(),
                coords,
            })
        }
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: operation.observable_id.ok_or_else(|| {
                PyValueError::new_err("observable_include requires observable_id")
            })?,
            measurement_keys: operation.measurement_keys.clone(),
        }),
        "observable_include_rec" => {
            validate_record_lookbacks(&operation.record_lookbacks)?;
            Ok(Op::ObservableIncludeRec {
                observable_id: operation.observable_id.ok_or_else(|| {
                    PyValueError::new_err("observable_include_rec requires observable_id")
                })?,
                lookbacks: operation.record_lookbacks.clone(),
            })
        }
        kind => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
}

fn validate_record_lookbacks(lookbacks: &[usize]) -> PyResult<()> {
    if lookbacks.contains(&0) {
        return Err(PyValueError::new_err(
            "measurement record lookbacks must be positive",
        ));
    }
    Ok(())
}

fn required_mapping_item<'py>(
    mapping: &'py Bound<'py, PyAny>,
    key: &str,
    context: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let mapping = mapping
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Operation.metadata must be a dict"))?;
    mapping
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{context} requires {key}")))
}

pub(crate) fn optional_native_noise_location(
    py: Python<'_>,
    location: &Option<Py<PyAny>>,
) -> PyResult<Option<NoiseLocationSpec>> {
    location
        .as_ref()
        .map(|location| parse_noise_location_object(location.bind(py)))
        .transpose()
}

pub(crate) fn cache_safe_optional_noise_location(
    py: Python<'_>,
    location: &Option<Py<PyAny>>,
) -> Option<Option<NoiseLocationSpec>> {
    match location {
        Some(location) => cache_safe_native_noise_location(py, location).map(Some),
        None => Some(None),
    }
}

pub(crate) fn cache_safe_native_noise_location(
    py: Python<'_>,
    value: &Py<PyAny>,
) -> Option<NoiseLocationSpec> {
    let location = value
        .bind(py)
        .extract::<PyRef<'_, PyNoiseLocation>>()
        .ok()?;
    let model = cache_safe_native_noise_model(location.model.bind(py))?;
    let tags = location.core_tags.clone()?;
    Some(NoiseLocationSpec {
        id: location.id.clone(),
        model,
        rate: location.rate,
        qubits: location.qubits.clone(),
        tags,
    })
}

pub(crate) fn optional_noise_location_attr(
    value: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Option<NoiseLocationSpec>> {
    match value.getattr(attr) {
        Ok(location) if !location.is_none() => Ok(Some(parse_noise_location_object(&location)?)),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

pub(crate) fn parse_noise_location_object(value: &Bound<'_, PyAny>) -> PyResult<NoiseLocationSpec> {
    if let Ok(location) = value.extract::<PyRef<'_, PyNoiseLocation>>() {
        return Ok(NoiseLocationSpec {
            id: location.id.clone(),
            model: parse_noise_model_object(location.model.bind(value.py()))?,
            rate: location.rate,
            qubits: location.qubits.clone(),
            tags: match &location.core_tags {
                Some(tags) => tags.clone(),
                None => parse_tags_mapping(location.tags.bind(value.py()))?,
            },
        });
    }
    let rate = required_attr(value, "rate", "NoiseLocation")?.extract::<f64>()?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    let model = parse_noise_model_object(&required_attr(value, "model", "NoiseLocation")?)?;
    Ok(NoiseLocationSpec {
        id: required_attr(value, "id", "NoiseLocation")?.extract::<String>()?,
        model,
        rate,
        qubits: required_attr(value, "qubits", "NoiseLocation")?.extract::<Vec<usize>>()?,
        tags: parse_optional_tags_attr(value, "tags")?,
    })
}

pub(crate) fn parse_noise_model_object(value: &Bound<'_, PyAny>) -> PyResult<NoiseModel> {
    if let Some(model) = cache_safe_native_noise_model(value) {
        return Ok(model);
    }

    let type_name = value.get_type().getattr("__name__")?.extract::<String>()?;
    let model = match type_name.as_str() {
        "BernoulliPauliNoise" => Ok(NoiseModel::BernoulliPauli(
            required_attr(value, "pauli", "BernoulliPauliNoise")?.extract::<String>()?,
        )),
        "MeasurementBitFlip" => Ok(NoiseModel::MeasurementBitFlip),
        "SingleQubitDepolarizing" => Ok(NoiseModel::SingleQubitDepolarizing),
        "TwoQubitDepolarizing" => {
            let events = required_attr(value, "_events", "TwoQubitDepolarizing")?
                .extract::<Vec<String>>()?;
            validate_two_qubit_events(&events)?;
            Ok(NoiseModel::TwoQubitDepolarizing)
        }
        "PauliChannel" => {
            let weights = required_attr(value, "weights", "PauliChannel")?;
            let weights = weights
                .downcast::<PyDict>()
                .map_err(|_| PyValueError::new_err("PauliChannel.weights must be a dict"))?;
            let mut out = Vec::with_capacity(weights.len());
            for (pauli, weight) in weights.iter() {
                out.push((pauli.extract::<String>()?, weight.extract::<f64>()?));
            }
            Ok(NoiseModel::PauliChannel(out))
        }
        _ => Err(PyValueError::new_err(format!(
            "unsupported native noise model {type_name:?}"
        ))),
    }?;
    model
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(model)
}

fn cache_safe_native_noise_model(value: &Bound<'_, PyAny>) -> Option<NoiseModel> {
    if let Ok(model) = value.extract::<PyRef<'_, PyBernoulliPauliNoise>>() {
        return Some(NoiseModel::BernoulliPauli(model.pauli.clone()));
    }
    if value.extract::<PyRef<'_, PyMeasurementBitFlip>>().is_ok() {
        return Some(NoiseModel::MeasurementBitFlip);
    }
    if value
        .extract::<PyRef<'_, PySingleQubitDepolarizing>>()
        .is_ok()
    {
        return Some(NoiseModel::SingleQubitDepolarizing);
    }
    if value.extract::<PyRef<'_, PyTwoQubitDepolarizing>>().is_ok() {
        return Some(NoiseModel::TwoQubitDepolarizing);
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyPauliChannel>>() {
        return Some(NoiseModel::PauliChannel(model.weights.clone()));
    }
    None
}

pub(crate) fn parse_optional_tags_attr(
    value: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<HashMap<String, TagValue>> {
    match value.getattr(attr) {
        Ok(tags) if !tags.is_none() => parse_tags_mapping(&tags),
        _ => Ok(HashMap::new()),
    }
}

pub(crate) fn parse_tags_mapping(value: &Bound<'_, PyAny>) -> PyResult<HashMap<String, TagValue>> {
    let tags = value
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("native tags must be a dict"))?;
    let mut out = HashMap::new();
    for (key, value) in tags.iter() {
        out.insert(key.extract::<String>()?, parse_tag_value(value)?);
    }
    Ok(out)
}

pub(crate) fn parse_tag_value(value: Bound<'_, PyAny>) -> PyResult<TagValue> {
    if value.is_none() {
        return Ok(TagValue::None);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(TagValue::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(TagValue::String(value.extract::<String>()?));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(TagValue::Int(value.extract::<i64>()?));
    }
    if value.is_instance_of::<PyFloat>() {
        let value = value.extract::<f64>()?;
        if value.is_nan() {
            return Err(PyValueError::new_err("native tags do not support NaN"));
        }
        return Ok(TagValue::Float(value.to_bits()));
    }
    Err(PyValueError::new_err(
        "native tags support only str, int, float, bool, or None",
    ))
}

pub(crate) fn parse_dem_detector_sequence(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<DemDetectorSpec>> {
    let py = value.py();
    let items = value
        .extract::<Vec<Py<PyAny>>>()
        .map_err(|_| PyValueError::new_err("detectors must be a sequence"))?;
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items {
        let detector = parse_dem_detector_object(item.bind(py))?;
        if !seen.insert(detector.id) {
            return Err(PyValueError::new_err("detector ids must be unique"));
        }
        out.push(detector);
    }
    Ok(out)
}

pub(crate) fn parse_dem_detector_object(value: &Bound<'_, PyAny>) -> PyResult<DemDetectorSpec> {
    Ok(DemDetectorSpec {
        id: required_attr(value, "id", "Detector")?.extract::<i64>()?,
        measurement_keys: required_attr(value, "measurement_keys", "Detector")?
            .extract::<Vec<String>>()?,
        coords: required_attr(value, "coords", "Detector")?.extract::<Vec<f64>>()?,
    })
}

pub(crate) fn parse_dem_observable_sequence(
    value: &Bound<'_, PyAny>,
) -> PyResult<Vec<DemObservableSpec>> {
    let py = value.py();
    let items = value
        .extract::<Vec<Py<PyAny>>>()
        .map_err(|_| PyValueError::new_err("observables must be a sequence"))?;
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items {
        let observable = parse_dem_observable_object(item.bind(py))?;
        if !seen.insert(observable.id) {
            return Err(PyValueError::new_err(
                "logical observable ids must be unique",
            ));
        }
        out.push(observable);
    }
    Ok(out)
}

pub(crate) fn parse_dem_observable_object(value: &Bound<'_, PyAny>) -> PyResult<DemObservableSpec> {
    let observable = DemObservableSpec {
        id: required_attr(value, "id", "LogicalObservable")?.extract::<i64>()?,
        measurement_keys: required_attr(value, "measurement_keys", "LogicalObservable")?
            .extract::<Vec<String>>()?,
        pauli_qubits: required_attr(value, "pauli_qubits", "LogicalObservable")?
            .extract::<Vec<usize>>()?,
        pauli: required_attr(value, "pauli", "LogicalObservable")?.extract::<String>()?,
    };
    observable
        .validate()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(observable)
}

pub(crate) fn parse_dem_edge_sequence(value: &Bound<'_, PyAny>) -> PyResult<Vec<DemEdgeSpec>> {
    let py = value.py();
    let items = value
        .extract::<Vec<Py<PyAny>>>()
        .map_err(|_| PyValueError::new_err("DEM edges must be a sequence"))?;
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        out.push(parse_dem_sampler_edge_object(item.bind(py))?);
    }
    Ok(out)
}

pub(crate) fn parse_dem_sampler_edge_object(value: &Bound<'_, PyAny>) -> PyResult<DemEdgeSpec> {
    let edge = DemEdgeSpec {
        probability: required_attr(value, "probability", "DetectorErrorEdge")?.extract::<f64>()?,
        detectors: required_attr(value, "detectors", "DetectorErrorEdge")?.extract::<Vec<i64>>()?,
        observables: required_attr(value, "observables", "DetectorErrorEdge")?
            .extract::<Vec<i64>>()?,
        location_id: required_attr(value, "location_id", "DetectorErrorEdge")?
            .extract::<String>()?,
        event: parse_dem_event(required_attr(value, "event", "DetectorErrorEdge")?)?,
        tags: parse_optional_tags_attr(value, "tags")?,
    };
    edge.validate()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(edge)
}

fn parse_dem_event(value: Bound<'_, PyAny>) -> PyResult<faultscope_core::DemEvent> {
    if value.is_instance_of::<PyBool>() {
        return Ok(faultscope_core::DemEvent::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(faultscope_core::DemEvent::Pauli(value.extract::<String>()?));
    }
    Err(PyValueError::new_err(
        "DetectorErrorEdge.event must be a Pauli/event string or bool",
    ))
}

pub(crate) fn required_attr<'py>(
    value: &Bound<'py, PyAny>,
    attr: &str,
    type_name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    value.getattr(attr).map_err(|_| {
        PyValueError::new_err(format!(
            "expected {type_name}-like object with attribute {attr:?}"
        ))
    })
}

pub(crate) fn optional_string_attr(
    value: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Option<String>> {
    match value.getattr(attr) {
        Ok(item) if !item.is_none() => Ok(Some(item.extract::<String>()?)),
        Ok(_) => Ok(None),
        Err(_) => Ok(None),
    }
}

pub(crate) fn one_qubit(qubits: &[usize], kind: &str) -> PyResult<usize> {
    if qubits.len() == 1 {
        Ok(qubits[0])
    } else {
        Err(PyValueError::new_err(format!(
            "{kind} requires exactly one qubit"
        )))
    }
}

pub(crate) fn two_qubits(qubits: &[usize], kind: &str) -> PyResult<(usize, usize)> {
    if qubits.len() == 2 {
        Ok((qubits[0], qubits[1]))
    } else {
        Err(PyValueError::new_err(format!("{kind} requires two qubits")))
    }
}
