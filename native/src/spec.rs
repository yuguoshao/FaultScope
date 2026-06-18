use crate::*;

#[derive(Clone)]
pub(crate) enum NoiseModel {
    BernoulliPauli(String),
    MeasurementBitFlip,
    SingleQubitDepolarizing,
    TwoQubitDepolarizing,
    PauliChannel(Vec<(String, f64)>),
}

#[derive(Clone)]
pub(crate) struct NoiseLocationSpec {
    pub(crate) id: String,
    pub(crate) model: NoiseModel,
    pub(crate) rate: f64,
    pub(crate) qubits: Vec<usize>,
    pub(crate) tags: HashMap<String, TagValue>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) enum TagValue {
    None,
    Bool(bool),
    Int(i64),
    Float(u64),
    String(String),
}

#[derive(Clone)]
pub(crate) enum DemEvent {
    Pauli(String),
    Bool(bool),
}

#[derive(Clone)]
pub(crate) struct DemDetectorSpec {
    pub(crate) id: i64,
    pub(crate) measurement_keys: Vec<String>,
}

#[derive(Clone)]
pub(crate) struct DemObservableSpec {
    pub(crate) id: i64,
    pub(crate) measurement_keys: Vec<String>,
    pub(crate) pauli_qubits: Vec<usize>,
    pub(crate) pauli: String,
}

#[derive(Clone)]
pub(crate) struct IndexedDemDetectorSpec {
    pub(crate) id: i64,
    pub(crate) measurement_indices: Vec<usize>,
}

#[derive(Clone)]
pub(crate) struct IndexedDemObservableSpec {
    pub(crate) id: i64,
    pub(crate) measurement_indices: Vec<usize>,
    pub(crate) pauli_qubits: Vec<usize>,
    pub(crate) pauli: String,
}

#[derive(Clone)]
pub(crate) struct DemMeasurementPlan {
    pub(crate) measurement_count: usize,
    pub(crate) measurement_indices_by_op: Vec<Option<usize>>,
    pub(crate) detectors: Vec<IndexedDemDetectorSpec>,
    pub(crate) observables: Vec<IndexedDemObservableSpec>,
}

#[derive(Clone)]
pub(crate) struct DemEdgeSpec {
    pub(crate) probability: f64,
    pub(crate) detectors: Vec<i64>,
    pub(crate) observables: Vec<i64>,
    pub(crate) location_id: String,
    pub(crate) tags: HashMap<String, TagValue>,
}

#[derive(Clone)]
pub(crate) struct DemLocationGroup {
    pub(crate) location_id: String,
    pub(crate) edge_indices: Vec<usize>,
    pub(crate) total_probability: f64,
}

pub(crate) struct GeneratedDemEdge {
    pub(crate) probability: f64,
    pub(crate) detectors: Vec<i64>,
    pub(crate) observables: Vec<i64>,
    pub(crate) location_id: String,
    pub(crate) event: DemEvent,
}

#[derive(Clone)]
pub(crate) struct NoiseOccurrence {
    pub(crate) op_index: usize,
    pub(crate) location: NoiseLocationSpec,
}

#[derive(Clone)]
pub(crate) enum Op {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Pauli {
        qubits: Vec<usize>,
        pauli: String,
    },
    Noise(NoiseLocationSpec),
    Measure {
        qubit: usize,
        key: Option<String>,
        basis: String,
        noise: Option<NoiseLocationSpec>,
    },
    MeasurePauli {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        noise: Option<NoiseLocationSpec>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
    },
    Detector {
        detector_id: i64,
        measurement_keys: Vec<String>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

#[derive(Clone)]
pub(crate) enum RunOp {
    H(usize),
    S(usize),
    SDag(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    Swap(usize, usize),
    Noise(NoiseLocationSpec),
    Measure {
        qubits: Vec<usize>,
        pauli: String,
        key: Option<String>,
        ideal: Expr,
        noise: Option<NoiseLocationSpec>,
    },
    Reset {
        qubit: usize,
        key: Option<String>,
        basis: String,
        ideal: Expr,
    },
    Detector {
        detector_id: i64,
        measurement_keys: Vec<String>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_keys: Vec<String>,
    },
}

impl Op {
    pub(crate) fn noise_locations(&self) -> Vec<&NoiseLocationSpec> {
        match self {
            Op::Noise(location) => vec![location],
            Op::Measure {
                noise: Some(location),
                ..
            } => vec![location],
            Op::MeasurePauli {
                noise: Some(location),
                ..
            } => vec![location],
            _ => Vec::new(),
        }
    }
}

pub(crate) fn parse_circuit_spec(spec: &Bound<'_, PyDict>) -> PyResult<(usize, Vec<Op>)> {
    let n_qubits = required(spec, "n_qubits")?.extract::<usize>()?;
    let operations_any = required(spec, "operations")?;
    let operations_seq = operations_any.downcast::<PyList>()?;
    let compact = match spec.get_item("format")? {
        Some(value) if !value.is_none() => value.extract::<String>()? == "compact_v1",
        _ => false,
    };
    let mut operations = Vec::with_capacity(operations_seq.len());
    for item in operations_seq.iter() {
        if compact {
            operations.push(parse_compact_operation(item.downcast::<PyTuple>()?)?);
        } else {
            operations.push(parse_operation(item.downcast::<PyDict>()?)?);
        }
    }
    Ok((n_qubits, operations))
}

pub(crate) fn parse_circuit_object(
    circuit: &Bound<'_, PyAny>,
    include_tags: bool,
) -> PyResult<(usize, Vec<Op>)> {
    let n_qubits = circuit.getattr("n_qubits")?.extract::<usize>()?;
    let operations_any = circuit.getattr("operations")?;
    let mut operations = Vec::new();
    if let Ok(items) = operations_any.downcast::<PyList>() {
        operations.reserve(items.len());
        for item in items.iter() {
            operations.push(parse_operation_object(&item, include_tags)?);
        }
    } else if let Ok(items) = operations_any.downcast::<PyTuple>() {
        operations.reserve(items.len());
        for item in items.iter() {
            operations.push(parse_operation_object(&item, include_tags)?);
        }
    } else {
        return Err(PyValueError::new_err(
            "Circuit.operations must be a list or tuple for native parsing",
        ));
    }
    Ok((n_qubits, operations))
}

pub(crate) fn parse_operation(dict: &Bound<'_, PyDict>) -> PyResult<Op> {
    let kind = required(dict, "kind")?.extract::<String>()?;
    let qubits = optional_vec_usize(dict, "qubits")?.unwrap_or_default();
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
            pauli: required(dict, "pauli")?.extract::<String>()?,
        }),
        "noise" => {
            let location = parse_noise_location(required(dict, "noise_location")?)?;
            Ok(Op::Noise(location))
        }
        "measure" => Ok(Op::Measure {
            qubit: one_qubit(&qubits, "measure")?,
            key: optional_string(dict, "key")?,
            basis: required(dict, "basis")?.extract::<String>()?.to_uppercase(),
            noise: optional_noise_location(dict)?,
        }),
        "measure_pauli" => Ok(Op::MeasurePauli {
            qubits,
            pauli: required(dict, "pauli")?.extract::<String>()?,
            key: optional_string(dict, "key")?,
            noise: optional_noise_location(dict)?,
        }),
        "reset" => Ok(Op::Reset {
            qubit: one_qubit(&qubits, "reset")?,
            key: optional_string(dict, "key")?,
            basis: required(dict, "basis")?.extract::<String>()?.to_uppercase(),
        }),
        "detector" => {
            let metadata_any = required(dict, "metadata")?;
            let metadata = metadata_any.downcast::<PyDict>()?;
            let detector_id = match metadata.get_item("detector_id")? {
                Some(value) if !value.is_none() => value.extract::<i64>()?,
                _ => -1,
            };
            Ok(Op::Detector {
                detector_id,
                measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
            })
        }
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: required(dict, "observable_id")?.extract::<i64>()?,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
}

pub(crate) fn parse_operation_object(
    operation: &Bound<'_, PyAny>,
    include_tags: bool,
) -> PyResult<Op> {
    let kind = operation.getattr("kind")?.extract::<String>()?;
    match kind.as_str() {
        "h" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            Ok(Op::H(one_qubit(&qubits, "h")?))
        }
        "s" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            Ok(Op::S(one_qubit(&qubits, "s")?))
        }
        "s_dag" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            Ok(Op::SDag(one_qubit(&qubits, "s_dag")?))
        }
        "cx" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "cx")?;
            Ok(Op::Cx(a, b))
        }
        "cz" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "cz")?;
            Ok(Op::Cz(a, b))
        }
        "swap" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "swap")?;
            Ok(Op::Swap(a, b))
        }
        "pauli" => Ok(Op::Pauli {
            qubits: operation.getattr("qubits")?.extract::<Vec<usize>>()?,
            pauli: optional_attr_string(operation, "pauli")?
                .ok_or_else(|| PyValueError::new_err("pauli operation is missing pauli"))?,
        }),
        "noise" => {
            let location = optional_attr_noise_location(operation, "noise_location", include_tags)?
                .ok_or_else(|| {
                    PyValueError::new_err("noise operation is missing noise_location")
                })?;
            Ok(Op::Noise(location))
        }
        "measure" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            Ok(Op::Measure {
                qubit: one_qubit(&qubits, "measure")?,
                key: optional_attr_string(operation, "key")?,
                basis: operation
                    .getattr("basis")?
                    .extract::<String>()?
                    .to_uppercase(),
                noise: optional_attr_noise_location(operation, "noise_location", include_tags)?,
            })
        }
        "measure_pauli" => Ok(Op::MeasurePauli {
            qubits: operation.getattr("qubits")?.extract::<Vec<usize>>()?,
            pauli: optional_attr_string(operation, "pauli")?
                .ok_or_else(|| PyValueError::new_err("measure_pauli operation is missing pauli"))?,
            key: optional_attr_string(operation, "key")?,
            noise: optional_attr_noise_location(operation, "noise_location", include_tags)?,
        }),
        "reset" => {
            let qubits = operation.getattr("qubits")?.extract::<Vec<usize>>()?;
            Ok(Op::Reset {
                qubit: one_qubit(&qubits, "reset")?,
                key: optional_attr_string(operation, "key")?,
                basis: operation
                    .getattr("basis")?
                    .extract::<String>()?
                    .to_uppercase(),
            })
        }
        "detector" => {
            let metadata = operation.getattr("metadata")?;
            let detector_id = if let Ok(metadata) = metadata.downcast::<PyDict>() {
                match metadata.get_item("detector_id")? {
                    Some(value) if !value.is_none() => value.extract::<i64>()?,
                    _ => -1,
                }
            } else {
                -1
            };
            Ok(Op::Detector {
                detector_id,
                measurement_keys: operation
                    .getattr("measurement_keys")?
                    .extract::<Vec<String>>()?,
            })
        }
        "observable_include" => Ok(Op::ObservableInclude {
            observable_id: optional_attr_i64(operation, "observable_id")?.ok_or_else(|| {
                PyValueError::new_err("observable_include operation is missing observable_id")
            })?,
            measurement_keys: operation
                .getattr("measurement_keys")?
                .extract::<Vec<String>>()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported native operation kind {kind:?}"
        ))),
    }
}

pub(crate) fn parse_compact_operation(item: &Bound<'_, PyTuple>) -> PyResult<Op> {
    if item.len() != 9 {
        return Err(PyValueError::new_err(format!(
            "compact native operation requires 9 fields, got {}",
            item.len()
        )));
    }
    let kind = item.get_item(0)?.extract::<i64>()?;

    match kind {
        0 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            Ok(Op::H(one_qubit(&qubits, "h")?))
        }
        1 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            Ok(Op::S(one_qubit(&qubits, "s")?))
        }
        2 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            Ok(Op::SDag(one_qubit(&qubits, "s_dag")?))
        }
        3 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "cx")?;
            Ok(Op::Cx(a, b))
        }
        4 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "cz")?;
            Ok(Op::Cz(a, b))
        }
        5 => {
            let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
            let (a, b) = two_qubits(&qubits, "swap")?;
            Ok(Op::Swap(a, b))
        }
        6 => Ok(Op::Pauli {
            qubits: item.get_item(1)?.extract::<Vec<usize>>()?,
            pauli: optional_string_value(item.get_item(4)?)?
                .ok_or_else(|| PyValueError::new_err("compact pauli operation is missing pauli"))?,
        }),
        7 => Ok(Op::Noise(
            optional_compact_noise_location(item.get_item(7)?)?.ok_or_else(|| {
                PyValueError::new_err("compact noise operation is missing noise location")
            })?,
        )),
        8 => Ok(Op::Measure {
            qubit: {
                let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
                one_qubit(&qubits, "measure")?
            },
            key: optional_string_value(item.get_item(2)?)?,
            basis: item.get_item(3)?.extract::<String>()?.to_uppercase(),
            noise: optional_compact_noise_location(item.get_item(7)?)?,
        }),
        9 => Ok(Op::MeasurePauli {
            qubits: item.get_item(1)?.extract::<Vec<usize>>()?,
            pauli: optional_string_value(item.get_item(4)?)?.ok_or_else(|| {
                PyValueError::new_err("compact measure_pauli operation is missing pauli")
            })?,
            key: optional_string_value(item.get_item(2)?)?,
            noise: optional_compact_noise_location(item.get_item(7)?)?,
        }),
        10 => Ok(Op::Reset {
            qubit: {
                let qubits = item.get_item(1)?.extract::<Vec<usize>>()?;
                one_qubit(&qubits, "reset")?
            },
            key: optional_string_value(item.get_item(2)?)?,
            basis: item.get_item(3)?.extract::<String>()?.to_uppercase(),
        }),
        11 => Ok(Op::Detector {
            detector_id: optional_i64_value(item.get_item(8)?)?.unwrap_or(-1),
            measurement_keys: item.get_item(5)?.extract::<Vec<String>>()?,
        }),
        12 => Ok(Op::ObservableInclude {
            observable_id: optional_i64_value(item.get_item(6)?)?.ok_or_else(|| {
                PyValueError::new_err(
                    "compact observable_include operation is missing observable_id",
                )
            })?,
            measurement_keys: item.get_item(5)?.extract::<Vec<String>>()?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unsupported compact native operation kind {kind}"
        ))),
    }
}

pub(crate) fn optional_attr_string(
    object: &Bound<'_, PyAny>,
    attr: &str,
) -> PyResult<Option<String>> {
    let value = object.getattr(attr)?;
    optional_string_value(value)
}

pub(crate) fn optional_attr_i64(object: &Bound<'_, PyAny>, attr: &str) -> PyResult<Option<i64>> {
    let value = object.getattr(attr)?;
    optional_i64_value(value)
}

pub(crate) fn optional_attr_noise_location(
    object: &Bound<'_, PyAny>,
    attr: &str,
    include_tags: bool,
) -> PyResult<Option<NoiseLocationSpec>> {
    let value = object.getattr(attr)?;
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(parse_noise_location_object(&value, include_tags)?))
}

pub(crate) fn optional_noise_location(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<NoiseLocationSpec>> {
    match dict.get_item("noise_location")? {
        Some(value) if !value.is_none() => Ok(Some(parse_noise_location(value)?)),
        _ => Ok(None),
    }
}

pub(crate) fn optional_compact_noise_location(
    value: Bound<'_, PyAny>,
) -> PyResult<Option<NoiseLocationSpec>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(parse_compact_noise_location(value)?))
}

pub(crate) fn parse_noise_location(value: Bound<'_, PyAny>) -> PyResult<NoiseLocationSpec> {
    let dict = value.downcast::<PyDict>()?;
    let model_any = required(dict, "model")?;
    let model_dict = model_any.downcast::<PyDict>()?;
    let model_type = required(model_dict, "type")?.extract::<String>()?;
    let model = match model_type.as_str() {
        "bernoulli_pauli" => {
            NoiseModel::BernoulliPauli(required(model_dict, "pauli")?.extract::<String>()?)
        }
        "measurement_bit_flip" => NoiseModel::MeasurementBitFlip,
        "single_qubit_depolarizing" => NoiseModel::SingleQubitDepolarizing,
        "two_qubit_depolarizing" => NoiseModel::TwoQubitDepolarizing,
        "pauli_channel" => {
            let weights = required(model_dict, "weights")?.extract::<Vec<(String, f64)>>()?;
            NoiseModel::PauliChannel(weights)
        }
        _ => {
            return Err(PyValueError::new_err(format!(
                "unsupported native noise model {model_type:?}"
            )))
        }
    };
    let rate = required(dict, "rate")?.extract::<f64>()?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    Ok(NoiseLocationSpec {
        id: required(dict, "id")?.extract::<String>()?,
        model,
        rate,
        qubits: required(dict, "qubits")?.extract::<Vec<usize>>()?,
        tags: parse_optional_tags(dict)?,
    })
}

pub(crate) fn parse_noise_location_object(
    value: &Bound<'_, PyAny>,
    include_tags: bool,
) -> PyResult<NoiseLocationSpec> {
    let rate = value.getattr("rate")?.extract::<f64>()?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    let tags = if include_tags {
        parse_tags_value(value.getattr("tags")?)?
    } else {
        HashMap::new()
    };
    Ok(NoiseLocationSpec {
        id: value.getattr("id")?.extract::<String>()?,
        model: parse_noise_model_object(&value.getattr("model")?)?,
        rate,
        qubits: value.getattr("qubits")?.extract::<Vec<usize>>()?,
        tags,
    })
}

pub(crate) fn parse_compact_noise_location(value: Bound<'_, PyAny>) -> PyResult<NoiseLocationSpec> {
    let tuple = value.downcast::<PyTuple>()?;
    if tuple.len() != 5 {
        return Err(PyValueError::new_err(format!(
            "compact noise location requires 5 fields, got {}",
            tuple.len()
        )));
    }
    let rate = tuple.get_item(1)?.extract::<f64>()?;
    if !(0.0..=1.0).contains(&rate) {
        return Err(PyValueError::new_err(format!(
            "noise rate must be in [0, 1], got {rate}"
        )));
    }
    Ok(NoiseLocationSpec {
        id: tuple.get_item(0)?.extract::<String>()?,
        rate,
        qubits: tuple.get_item(2)?.extract::<Vec<usize>>()?,
        tags: parse_tags_value(tuple.get_item(3)?)?,
        model: parse_compact_noise_model(tuple.get_item(4)?)?,
    })
}

pub(crate) fn parse_noise_model_object(value: &Bound<'_, PyAny>) -> PyResult<NoiseModel> {
    let type_name = value.get_type().name()?.to_string();
    match type_name.as_str() {
        "BernoulliPauliNoise" => Ok(NoiseModel::BernoulliPauli(
            value.getattr("pauli")?.extract::<String>()?,
        )),
        "MeasurementBitFlip" => Ok(NoiseModel::MeasurementBitFlip),
        "SingleQubitDepolarizing" => Ok(NoiseModel::SingleQubitDepolarizing),
        "TwoQubitDepolarizing" => Ok(NoiseModel::TwoQubitDepolarizing),
        "PauliChannel" => {
            let weights_any = value.getattr("weights")?;
            let weights = weights_any.downcast::<PyDict>()?;
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

pub(crate) fn parse_compact_noise_model(value: Bound<'_, PyAny>) -> PyResult<NoiseModel> {
    let tuple = value.downcast::<PyTuple>()?;
    if tuple.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "compact noise model requires 3 fields, got {}",
            tuple.len()
        )));
    }
    let code = tuple.get_item(0)?.extract::<i64>()?;
    match code {
        0 => Ok(NoiseModel::BernoulliPauli(
            tuple.get_item(1)?.extract::<String>()?,
        )),
        1 => Ok(NoiseModel::MeasurementBitFlip),
        2 => Ok(NoiseModel::SingleQubitDepolarizing),
        3 => Ok(NoiseModel::TwoQubitDepolarizing),
        4 => Ok(NoiseModel::PauliChannel(
            tuple.get_item(2)?.extract::<Vec<(String, f64)>>()?,
        )),
        _ => Err(PyValueError::new_err(format!(
            "unsupported compact native noise model {code}"
        ))),
    }
}

pub(crate) fn parse_optional_tags(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, TagValue>> {
    match dict.get_item("tags")? {
        Some(value) if !value.is_none() => parse_tags_value(value),
        _ => Ok(HashMap::new()),
    }
}

pub(crate) fn parse_tags_value(value: Bound<'_, PyAny>) -> PyResult<HashMap<String, TagValue>> {
    let tags = value.downcast::<PyDict>()?;
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

pub(crate) fn required<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("native sampler spec missing {key}")))
}

pub(crate) fn optional_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<String>()?)),
        _ => Ok(None),
    }
}

pub(crate) fn optional_string_value(value: Bound<'_, PyAny>) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<String>()?))
}

pub(crate) fn optional_i64_value(value: Bound<'_, PyAny>) -> PyResult<Option<i64>> {
    if value.is_none() {
        return Ok(None);
    }
    Ok(Some(value.extract::<i64>()?))
}

pub(crate) fn optional_vec_usize(
    dict: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<Vec<usize>>> {
    match dict.get_item(key)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<Vec<usize>>()?)),
        _ => Ok(None),
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

pub(crate) fn parse_dem_detectors(items: &Bound<'_, PyList>) -> PyResult<Vec<DemDetectorSpec>> {
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items.iter() {
        let dict = item.downcast::<PyDict>()?;
        let id = required(dict, "id")?.extract::<i64>()?;
        if !seen.insert(id) {
            return Err(PyValueError::new_err("detector ids must be unique"));
        }
        out.push(DemDetectorSpec {
            id,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
        });
    }
    Ok(out)
}

pub(crate) fn parse_dem_observables(items: &Bound<'_, PyList>) -> PyResult<Vec<DemObservableSpec>> {
    let mut out = Vec::with_capacity(items.len());
    let mut seen = HashSet::new();
    for item in items.iter() {
        let dict = item.downcast::<PyDict>()?;
        let id = required(dict, "id")?.extract::<i64>()?;
        if !seen.insert(id) {
            return Err(PyValueError::new_err(
                "logical observable ids must be unique",
            ));
        }
        let pauli_qubits = required(dict, "pauli_qubits")?.extract::<Vec<usize>>()?;
        let pauli = required(dict, "pauli")?.extract::<String>()?;
        if pauli_qubits.len() != pauli.len() {
            return Err(PyValueError::new_err(
                "pauli_qubits and pauli must have the same length",
            ));
        }
        out.push(DemObservableSpec {
            id,
            measurement_keys: required(dict, "measurement_keys")?.extract::<Vec<String>>()?,
            pauli_qubits,
            pauli,
        });
    }
    Ok(out)
}
