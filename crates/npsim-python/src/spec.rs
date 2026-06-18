use crate::*;
use npsim_core::{
    Detector as CoreDetector, LogicalObservable as CoreLogicalObservable,
    NoiseLocation as CoreNoiseLocation,
};

pub(crate) use npsim_core::{NoiseModel, TagValue};

pub(crate) type NoiseLocationSpec = CoreNoiseLocation;
pub(crate) type DemDetectorSpec = CoreDetector;
pub(crate) type DemObservableSpec = CoreLogicalObservable;
pub(crate) type Op = npsim_core::Operation;
pub(crate) type DemEdgeSpec = npsim_core::DemSamplerEdge;

pub(crate) fn parse_core_circuit_object(value: &Bound<'_, PyAny>) -> PyResult<npsim_core::Circuit> {
    let (n_qubits, operations) = parse_circuit_object(value)?;
    Ok(npsim_core::Circuit {
        n_qubits,
        operations,
    })
}

pub(crate) fn parse_circuit_object(value: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<Op>)> {
    let n_qubits = required_attr(value, "n_qubits", "Circuit")?.extract::<usize>()?;
    let operations = parse_operation_sequence(&required_attr(value, "operations", "Circuit")?)?;
    Ok((n_qubits, operations))
}

pub(crate) fn parse_py_noise_location_map(
    circuit: &Bound<'_, PyAny>,
) -> PyResult<HashMap<String, Py<PyAny>>> {
    let locations = circuit.call_method0("noise_locations")?;
    let locations = locations
        .downcast::<PyDict>()
        .map_err(|_| PyValueError::new_err("Circuit.noise_locations() must return a dict"))?;
    let mut out = HashMap::new();
    for (location_id, location) in locations.iter() {
        out.insert(location_id.extract::<String>()?, location.unbind());
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
    let kind = required_attr(value, "kind", "Operation")?.extract::<String>()?;
    let qubits = required_attr(value, "qubits", "Operation")?.extract::<Vec<usize>>()?;
    match kind.as_str() {
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
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: required_attr(value, "observable_id", "Operation")?.extract::<i64>()?,
            measurement_keys: required_attr(value, "measurement_keys", "Operation")?
                .extract::<Vec<String>>()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
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
    if let Ok(model) = value.extract::<PyRef<'_, PyBernoulliPauliNoise>>() {
        return Ok(NoiseModel::BernoulliPauli(model.pauli.clone()));
    }
    if value.extract::<PyRef<'_, PyMeasurementBitFlip>>().is_ok() {
        return Ok(NoiseModel::MeasurementBitFlip);
    }
    if value
        .extract::<PyRef<'_, PySingleQubitDepolarizing>>()
        .is_ok()
    {
        return Ok(NoiseModel::SingleQubitDepolarizing);
    }
    if value.extract::<PyRef<'_, PyTwoQubitDepolarizing>>().is_ok() {
        return Ok(NoiseModel::TwoQubitDepolarizing);
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyPauliChannel>>() {
        return Ok(NoiseModel::PauliChannel(model.weights.clone()));
    }

    let type_name = value.get_type().getattr("__name__")?.extract::<String>()?;
    match type_name.as_str() {
        "BernoulliPauliNoise" => Ok(NoiseModel::BernoulliPauli(
            required_attr(value, "pauli", "BernoulliPauliNoise")?.extract::<String>()?,
        )),
        "MeasurementBitFlip" => Ok(NoiseModel::MeasurementBitFlip),
        "SingleQubitDepolarizing" => Ok(NoiseModel::SingleQubitDepolarizing),
        "TwoQubitDepolarizing" => Ok(NoiseModel::TwoQubitDepolarizing),
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
    }
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

fn parse_dem_event(value: Bound<'_, PyAny>) -> PyResult<npsim_core::DemEvent> {
    if value.is_instance_of::<PyBool>() {
        return Ok(npsim_core::DemEvent::Bool(value.extract::<bool>()?));
    }
    if value.is_instance_of::<PyString>() {
        return Ok(npsim_core::DemEvent::Pauli(value.extract::<String>()?));
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
