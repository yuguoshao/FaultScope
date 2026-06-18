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
    let mut operations = Vec::with_capacity(operations_seq.len());
    for item in operations_seq.iter() {
        let dict = item.downcast::<PyDict>()?;
        operations.push(parse_operation(dict)?);
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

pub(crate) fn optional_noise_location(
    dict: &Bound<'_, PyDict>,
) -> PyResult<Option<NoiseLocationSpec>> {
    match dict.get_item("noise_location")? {
        Some(value) if !value.is_none() => Ok(Some(parse_noise_location(value)?)),
        _ => Ok(None),
    }
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

pub(crate) fn parse_optional_tags(dict: &Bound<'_, PyDict>) -> PyResult<HashMap<String, TagValue>> {
    match dict.get_item("tags")? {
        Some(value) if !value.is_none() => {
            let tags = value.downcast::<PyDict>()?;
            let mut out = HashMap::new();
            for (key, value) in tags.iter() {
                out.insert(key.extract::<String>()?, parse_tag_value(value)?);
            }
            Ok(out)
        }
        _ => Ok(HashMap::new()),
    }
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
