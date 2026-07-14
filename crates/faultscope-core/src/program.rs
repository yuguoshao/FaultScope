use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::labels::LocationCatalogBuilder;
use crate::{
    Circuit, IndexedNoiseLocation, LocationCatalog, NoiseLocation, NpError, NpResult, Operation,
    PauliBasis,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopRange {
    pub(crate) iterations: Vec<Range<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpansionMode {
    Sampler,
    Dem,
}

impl ExpansionMode {
    const fn retains_detector_coords(self) -> bool {
        matches!(self, Self::Dem)
    }
}

/// Canonical expansion shared by forward-sampler and DEM compilation.
///
/// Operations refer to measurements and noise locations by dense integer ID.
/// String labels live only in the boundary tables used for input resolution
/// and public result materialization.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpandedProgram {
    pub(crate) operations: Vec<ExpandedOperation>,
    pub(crate) measurement_keys: Vec<String>,
    pub(crate) noise_locations: Vec<IndexedNoiseLocation>,
    pub(crate) location_catalog: LocationCatalog,
    pub(crate) detector_coords: Vec<Vec<f64>>,
    pub(crate) stored_operation_count: usize,
    pub(crate) logical_operation_count: usize,
    pub(crate) ideal_count: usize,
    pub(crate) loop_count: usize,
    pub(crate) loop_ranges: Vec<LoopRange>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ExpandedOperation {
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
    Noise(usize),
    MeasureSingle {
        qubit: usize,
        basis: PauliBasis,
        measurement_id: usize,
        noise: Option<usize>,
    },
    MeasurePauli {
        qubits: Vec<usize>,
        pauli: String,
        measurement_id: usize,
        noise: Option<usize>,
    },
    Reset {
        qubit: usize,
        measurement_id: Option<usize>,
        basis: PauliBasis,
    },
    Detector {
        detector_id: i64,
        measurement_ids: Vec<usize>,
    },
    ObservableInclude {
        observable_id: i64,
        measurement_ids: Vec<usize>,
    },
}

/// Expand a structured operation tree into the single integer-indexed IR used
/// by both forward-sampler compilation and DEM generation.
pub(crate) fn expand_operations(
    operations: &[Operation],
    mode: ExpansionMode,
) -> NpResult<ExpandedProgram> {
    let counts = operation_counts(operations)?;
    let mut state = ExpansionState::with_capacity(&counts, mode);
    let mut site_path = Vec::new();
    state.expand_sequence(operations, &mut site_path)?;
    let logical_operation_count = state.output.len();
    let measurement_keys = state
        .measurement_keys
        .into_iter()
        .enumerate()
        .map(|(measurement_id, key)| {
            key.ok_or_else(|| {
                NpError::new(format!(
                    "internal error: missing measurement key for id {measurement_id}"
                ))
            })
        })
        .collect::<NpResult<Vec<_>>>()?;
    Ok(ExpandedProgram {
        operations: state.output,
        measurement_keys,
        noise_locations: state.noise_locations,
        location_catalog: state.location_catalog.finish(),
        detector_coords: state.detector_coords,
        stored_operation_count: counts.stored,
        logical_operation_count,
        ideal_count: counts.ideals,
        loop_count: state.loop_count,
        loop_ranges: state.loop_ranges,
    })
}

pub(crate) fn expand_circuit(circuit: &Circuit) -> NpResult<(Circuit, ExpandedProgram)> {
    let program = expand_operations(&circuit.operations, ExpansionMode::Dem)?;
    Ok((
        Circuit {
            n_qubits: circuit.n_qubits,
            operations: materialize_operations(&program),
        },
        program,
    ))
}

pub fn expand_circuit_operations(circuit: &Circuit) -> NpResult<Circuit> {
    Ok(expand_circuit(circuit)?.0)
}

pub(crate) fn materialize_operations(program: &ExpandedProgram) -> Vec<Operation> {
    let mut detector_index = 0usize;
    program
        .operations
        .iter()
        .map(|operation| match operation {
            ExpandedOperation::H(qubit) => Operation::H(*qubit),
            ExpandedOperation::S(qubit) => Operation::S(*qubit),
            ExpandedOperation::SDag(qubit) => Operation::SDag(*qubit),
            ExpandedOperation::Cx(control, target) => Operation::Cx(*control, *target),
            ExpandedOperation::Cz(left, right) => Operation::Cz(*left, *right),
            ExpandedOperation::Swap(left, right) => Operation::Swap(*left, *right),
            ExpandedOperation::Pauli { qubits, pauli } => Operation::Pauli {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
            },
            ExpandedOperation::Noise(noise_id) => Operation::Noise(
                program
                    .location_catalog
                    .materialize_noise_location(&program.noise_locations[*noise_id]),
            ),
            ExpandedOperation::MeasureSingle {
                qubit,
                basis,
                measurement_id,
                noise,
            } => Operation::Measure {
                qubit: *qubit,
                key: Some(program.measurement_keys[*measurement_id].clone()),
                basis: basis.as_str().to_string(),
                noise: noise.map(|noise_id| {
                    program
                        .location_catalog
                        .materialize_noise_location(&program.noise_locations[noise_id])
                }),
            },
            ExpandedOperation::MeasurePauli {
                qubits,
                pauli,
                measurement_id,
                noise,
            } => Operation::MeasurePauli {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
                key: Some(program.measurement_keys[*measurement_id].clone()),
                noise: noise.map(|noise_id| {
                    program
                        .location_catalog
                        .materialize_noise_location(&program.noise_locations[noise_id])
                }),
            },
            ExpandedOperation::Reset {
                qubit,
                measurement_id,
                basis,
            } => Operation::Reset {
                qubit: *qubit,
                key: measurement_id.map(|id| program.measurement_keys[id].clone()),
                basis: basis.as_str().to_string(),
            },
            ExpandedOperation::Detector {
                detector_id,
                measurement_ids,
            } => {
                let coords = program
                    .detector_coords
                    .get(detector_index)
                    .cloned()
                    .unwrap_or_default();
                detector_index += 1;
                Operation::Detector {
                    detector_id: Some(*detector_id),
                    measurement_keys: measurement_ids
                        .iter()
                        .map(|id| program.measurement_keys[*id].clone())
                        .collect(),
                    coords,
                }
            }
            ExpandedOperation::ObservableInclude {
                observable_id,
                measurement_ids,
            } => Operation::ObservableInclude {
                observable_id: *observable_id,
                measurement_keys: measurement_ids
                    .iter()
                    .map(|id| program.measurement_keys[*id].clone())
                    .collect(),
            },
        })
        .collect()
}

#[derive(Default)]
struct OperationCounts {
    stored: usize,
    logical: usize,
    expanded_loops: usize,
    measurements: usize,
    ideals: usize,
    needs_measurement_history: bool,
}

fn operation_counts(operations: &[Operation]) -> NpResult<OperationCounts> {
    let mut counts = OperationCounts::default();
    for operation in operations {
        counts.stored = counts
            .stored
            .checked_add(1)
            .ok_or_else(|| NpError::new("stored operation count overflow"))?;
        match operation {
            Operation::Repeat { count, body } => {
                let body_counts = operation_counts(body)?;
                counts.stored = counts
                    .stored
                    .checked_add(body_counts.stored)
                    .ok_or_else(|| NpError::new("stored operation count overflow"))?;
                counts.logical = counts
                    .logical
                    .checked_add(
                        body_counts
                            .logical
                            .checked_mul(*count)
                            .ok_or_else(|| NpError::new("logical operation count overflow"))?,
                    )
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
                counts.expanded_loops = counts
                    .expanded_loops
                    .checked_add(1)
                    .and_then(|value| {
                        body_counts
                            .expanded_loops
                            .checked_mul(*count)
                            .and_then(|nested| value.checked_add(nested))
                    })
                    .ok_or_else(|| NpError::new("loop count overflow"))?;
                counts.measurements = counts
                    .measurements
                    .checked_add(
                        body_counts
                            .measurements
                            .checked_mul(*count)
                            .ok_or_else(|| NpError::new("measurement count overflow"))?,
                    )
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
                counts.ideals = counts
                    .ideals
                    .checked_add(
                        body_counts
                            .ideals
                            .checked_mul(*count)
                            .ok_or_else(|| NpError::new("measurement count overflow"))?,
                    )
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
                counts.needs_measurement_history |= body_counts.needs_measurement_history;
            }
            Operation::Tick | Operation::ShiftCoords(_) => {}
            Operation::Measure { .. }
            | Operation::MeasurePauli { .. }
            | Operation::MeasureReset { .. } => {
                counts.logical = counts
                    .logical
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
                counts.measurements = counts
                    .measurements
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
                counts.ideals = counts
                    .ideals
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
            }
            Operation::Reset { key, .. } => {
                counts.logical = counts
                    .logical
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
                counts.measurements = counts
                    .measurements
                    .checked_add(usize::from(key.is_some()))
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
                counts.ideals = counts
                    .ideals
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("measurement count overflow"))?;
            }
            Operation::DetectorRec { .. } | Operation::ObservableIncludeRec { .. } => {
                counts.logical = counts
                    .logical
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
                counts.needs_measurement_history = true;
            }
            _ => {
                counts.logical = counts
                    .logical
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
            }
        }
    }
    Ok(counts)
}

struct ExpansionState {
    output: Vec<ExpandedOperation>,
    measurement_history: Vec<usize>,
    measurement_keys: Vec<Option<String>>,
    key_aliases: HashMap<String, usize>,
    explicit_key_ids: HashMap<String, usize>,
    unknown_key_ids: HashMap<String, usize>,
    coordinate_shift: Vec<f64>,
    mode: ExpansionMode,
    repeat_path: Vec<usize>,
    detector_site_instances: HashMap<Vec<usize>, i64>,
    seen_detector_ids: HashSet<i64>,
    noise_locations: Vec<IndexedNoiseLocation>,
    location_catalog: LocationCatalogBuilder,
    detector_coords: Vec<Vec<f64>>,
    measurement_count: usize,
    record_measurement_history: bool,
    detector_ordinal: i64,
    loop_count: usize,
    loop_ranges: Vec<LoopRange>,
}

impl ExpansionState {
    fn with_capacity(counts: &OperationCounts, mode: ExpansionMode) -> Self {
        Self {
            output: Vec::with_capacity(counts.logical),
            measurement_history: if counts.needs_measurement_history {
                Vec::with_capacity(counts.measurements)
            } else {
                Vec::new()
            },
            measurement_keys: vec![None; counts.measurements],
            key_aliases: HashMap::new(),
            explicit_key_ids: HashMap::new(),
            unknown_key_ids: HashMap::new(),
            coordinate_shift: Vec::new(),
            mode,
            repeat_path: Vec::new(),
            detector_site_instances: HashMap::new(),
            seen_detector_ids: HashSet::new(),
            noise_locations: Vec::new(),
            location_catalog: LocationCatalogBuilder::with_capacity(counts.logical),
            detector_coords: Vec::new(),
            measurement_count: 0,
            record_measurement_history: counts.needs_measurement_history,
            detector_ordinal: 0,
            loop_count: 0,
            loop_ranges: Vec::with_capacity(counts.expanded_loops),
        }
    }

    fn expand_sequence(
        &mut self,
        operations: &[Operation],
        site_path: &mut Vec<usize>,
    ) -> NpResult<()> {
        for (index, operation) in operations.iter().enumerate() {
            site_path.push(index);
            self.expand_operation(operation, site_path)?;
            site_path.pop();
        }
        Ok(())
    }

    fn expand_operation(
        &mut self,
        operation: &Operation,
        site_path: &mut Vec<usize>,
    ) -> NpResult<()> {
        match operation {
            Operation::Tick => {}
            Operation::ShiftCoords(offsets) => {
                if self.mode.retains_detector_coords() {
                    add_coords(&mut self.coordinate_shift, offsets);
                }
            }
            Operation::Repeat { count, body } => {
                if *count == 0 {
                    return Err(NpError::new("repeat count must be positive"));
                }
                self.loop_count = self
                    .loop_count
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("loop count overflow"))?;
                let outer_aliases = self.key_aliases.clone();
                let mut iterations = Vec::with_capacity(*count);
                for iteration in 0..*count {
                    self.key_aliases = outer_aliases.clone();
                    self.repeat_path.push(iteration);
                    let start = self.output.len();
                    self.expand_sequence(body, site_path)?;
                    iterations.push(start..self.output.len());
                    self.repeat_path.pop();
                }
                self.key_aliases = outer_aliases;
                self.loop_ranges.push(LoopRange { iterations });
            }
            Operation::H(qubit) => self.output.push(ExpandedOperation::H(*qubit)),
            Operation::S(qubit) => self.output.push(ExpandedOperation::S(*qubit)),
            Operation::SDag(qubit) => self.output.push(ExpandedOperation::SDag(*qubit)),
            Operation::Cx(control, target) => {
                self.output.push(ExpandedOperation::Cx(*control, *target))
            }
            Operation::Cz(left, right) => self.output.push(ExpandedOperation::Cz(*left, *right)),
            Operation::Swap(left, right) => {
                self.output.push(ExpandedOperation::Swap(*left, *right))
            }
            Operation::Pauli { qubits, pauli } => {
                self.output.push(ExpandedOperation::Pauli {
                    qubits: qubits.clone(),
                    pauli: pauli.clone(),
                });
            }
            Operation::Noise(location) => {
                let noise_id = self.register_noise(location)?;
                self.output.push(ExpandedOperation::Noise(noise_id));
            }
            Operation::Measure {
                qubit,
                key,
                basis,
                noise,
            } => {
                let measurement_id = self.measurement_key(key.as_deref())?;
                let noise = noise
                    .as_ref()
                    .map(|location| self.register_noise(location))
                    .transpose()?;
                self.output.push(ExpandedOperation::MeasureSingle {
                    qubit: *qubit,
                    basis: PauliBasis::parse(basis)?,
                    measurement_id,
                    noise,
                });
            }
            Operation::MeasurePauli {
                qubits,
                pauli,
                key,
                noise,
            } => {
                let measurement_id = self.measurement_key(key.as_deref())?;
                let noise = noise
                    .as_ref()
                    .map(|location| self.register_noise(location))
                    .transpose()?;
                self.output.push(ExpandedOperation::MeasurePauli {
                    qubits: qubits.clone(),
                    pauli: pauli.clone(),
                    measurement_id,
                    noise,
                });
            }
            Operation::MeasureReset { qubit, basis } => {
                let measurement_id = self.measurement_key(None)?;
                self.output.push(ExpandedOperation::Reset {
                    qubit: *qubit,
                    measurement_id: Some(measurement_id),
                    basis: PauliBasis::parse(basis)?,
                });
            }
            Operation::Reset { qubit, key, basis } => {
                let measurement_id = key
                    .as_deref()
                    .map(|key| self.measurement_key(Some(key)))
                    .transpose()?;
                self.output.push(ExpandedOperation::Reset {
                    qubit: *qubit,
                    measurement_id,
                    basis: PauliBasis::parse(basis)?,
                });
            }
            Operation::Detector {
                detector_id,
                measurement_keys,
                coords,
            } => {
                let detector_id = self.detector_id(*detector_id, site_path)?;
                let measurement_ids = self.resolve_reference_keys(measurement_keys);
                self.record_detector_coords(coords);
                self.output.push(ExpandedOperation::Detector {
                    detector_id,
                    measurement_ids,
                });
            }
            Operation::DetectorRec {
                detector_id,
                lookbacks,
                coords,
            } => {
                let detector_id = self.detector_id(*detector_id, site_path)?;
                let measurement_ids = self.resolve_lookbacks(lookbacks)?;
                self.record_detector_coords(coords);
                self.output.push(ExpandedOperation::Detector {
                    detector_id,
                    measurement_ids,
                });
            }
            Operation::ObservableInclude {
                observable_id,
                measurement_keys,
            } => {
                let measurement_ids = self.resolve_reference_keys(measurement_keys);
                self.output.push(ExpandedOperation::ObservableInclude {
                    observable_id: *observable_id,
                    measurement_ids,
                });
            }
            Operation::ObservableIncludeRec {
                observable_id,
                lookbacks,
            } => {
                let measurement_ids = self.resolve_lookbacks(lookbacks)?;
                self.output.push(ExpandedOperation::ObservableInclude {
                    observable_id: *observable_id,
                    measurement_ids,
                });
            }
        }
        Ok(())
    }

    fn measurement_key(&mut self, explicit: Option<&str>) -> NpResult<usize> {
        let measurement_id = self.measurement_count;
        let (key, automatic) = match explicit {
            Some(base) if !self.repeat_path.is_empty() => {
                let key = format!("{base}@r{}", format_repeat_path(&self.repeat_path));
                self.key_aliases.insert(base.to_string(), measurement_id);
                (key, false)
            }
            Some(base) => {
                self.key_aliases.insert(base.to_string(), measurement_id);
                (base.to_string(), false)
            }
            None => (automatic_measurement_key(measurement_id), true),
        };
        let duplicate = if automatic {
            self.explicit_key_ids.contains_key(&key)
        } else {
            automatic_key_ordinal(&key).is_some_and(|ordinal| {
                ordinal < self.measurement_count
                    && self.measurement_keys[ordinal].as_deref() == Some(key.as_str())
            }) || self.explicit_key_ids.contains_key(&key)
        };
        if duplicate {
            return Err(NpError::new(format!("duplicate measurement key {key:?}")));
        }
        if !automatic {
            self.explicit_key_ids.insert(key.clone(), measurement_id);
        }
        let slot = self
            .measurement_keys
            .get_mut(measurement_id)
            .ok_or_else(|| NpError::new("measurement count exceeds reserved capacity"))?;
        *slot = Some(key);
        self.measurement_count = self
            .measurement_count
            .checked_add(1)
            .ok_or_else(|| NpError::new("measurement count overflow"))?;
        if self.record_measurement_history {
            self.measurement_history.push(measurement_id);
        }
        Ok(measurement_id)
    }

    fn resolve_reference_keys(&mut self, keys: &[String]) -> Vec<usize> {
        keys.iter()
            .map(|key| self.resolve_reference_key(key))
            .collect()
    }

    fn resolve_reference_key(&mut self, key: &str) -> usize {
        if let Some(measurement_id) = self.key_aliases.get(key) {
            return *measurement_id;
        }
        if let Some(measurement_id) = automatic_key_ordinal(key) {
            if measurement_id < self.measurement_count
                && self.measurement_keys[measurement_id].as_deref() == Some(key)
            {
                return measurement_id;
            }
        }
        if let Some(measurement_id) = self.explicit_key_ids.get(key) {
            return *measurement_id;
        }
        if let Some(measurement_id) = self.unknown_key_ids.get(key) {
            return *measurement_id;
        }
        let measurement_id = self.measurement_keys.len();
        self.unknown_key_ids.insert(key.to_string(), measurement_id);
        self.measurement_keys.push(Some(key.to_string()));
        measurement_id
    }

    fn resolve_lookbacks(&self, lookbacks: &[usize]) -> NpResult<Vec<usize>> {
        lookbacks
            .iter()
            .map(|lookback| {
                if *lookback == 0 || *lookback > self.measurement_history.len() {
                    return Err(NpError::new(format!(
                        "measurement record lookback {lookback} is out of range"
                    )));
                }
                Ok(self.measurement_history[self.measurement_history.len() - lookback])
            })
            .collect()
    }

    fn register_noise(&mut self, location: &NoiseLocation) -> NpResult<usize> {
        let location = self.instantiate_location(location);
        let (location_id, inserted) = self
            .location_catalog
            .intern_with_status(location.id.clone(), location.tags.clone());
        if !inserted {
            return Err(NpError::new(format!(
                "native sampler requires unique noise location ids; duplicate {:?}",
                location.id
            )));
        }
        let noise_id = self.noise_locations.len();
        self.noise_locations
            .push(IndexedNoiseLocation::from_input(location_id, location));
        Ok(noise_id)
    }

    fn instantiate_location(&self, location: &NoiseLocation) -> NoiseLocation {
        if self.repeat_path.is_empty() {
            return location.clone();
        }
        let mut location = location.clone();
        location.id = format!("{}@r{}", location.id, format_repeat_path(&self.repeat_path));
        location
    }

    fn detector_id(&mut self, detector_id: Option<i64>, site_path: &[usize]) -> NpResult<i64> {
        let ordinal = self.detector_ordinal;
        self.detector_ordinal = self
            .detector_ordinal
            .checked_add(1)
            .ok_or_else(|| NpError::new("detector id overflow"))?;
        let id = match detector_id {
            None => ordinal,
            Some(base) if self.repeat_path.is_empty() => base,
            Some(base) => {
                let instance = self
                    .detector_site_instances
                    .entry(site_path.to_vec())
                    .or_insert(0);
                let id = base
                    .checked_add(*instance)
                    .ok_or_else(|| NpError::new("detector id overflow"))?;
                *instance = instance
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("detector id overflow"))?;
                id
            }
        };
        if !self.seen_detector_ids.insert(id) {
            return Err(NpError::new(format!("duplicate detector id {id}")));
        }
        Ok(id)
    }

    fn record_detector_coords(&mut self, coords: &[f64]) {
        if self.mode.retains_detector_coords() {
            self.detector_coords
                .push(shifted_coords(coords, &self.coordinate_shift));
        }
    }
}

fn format_repeat_path(path: &[usize]) -> String {
    path.iter()
        .map(|index| format!("[{index}]"))
        .collect::<String>()
}

fn automatic_measurement_key(ordinal: usize) -> String {
    let mut digits = [0u8; 20];
    let mut start = digits.len();
    let mut remaining = ordinal;
    loop {
        start -= 1;
        digits[start] = b'0' + (remaining % 10) as u8;
        remaining /= 10;
        if remaining == 0 {
            break;
        }
    }
    let decimal =
        std::str::from_utf8(&digits[start..]).expect("ASCII decimal digits are always valid UTF-8");
    let mut key = String::with_capacity(1 + decimal.len());
    key.push('m');
    key.push_str(decimal);
    key
}

pub(crate) fn automatic_key_ordinal(key: &str) -> Option<usize> {
    let bytes = key.as_bytes();
    if bytes.len() < 2 || bytes[0] != b'm' || (bytes.len() > 2 && bytes[1] == b'0') {
        return None;
    }
    let mut ordinal = 0usize;
    for byte in &bytes[1..] {
        if !byte.is_ascii_digit() {
            return None;
        }
        ordinal = ordinal
            .checked_mul(10)?
            .checked_add((byte - b'0') as usize)?;
    }
    Some(ordinal)
}

fn add_coords(target: &mut Vec<f64>, offsets: &[f64]) {
    if target.len() < offsets.len() {
        target.resize(offsets.len(), 0.0);
    }
    for (index, offset) in offsets.iter().enumerate() {
        target[index] += offset;
    }
}

fn shifted_coords(coords: &[f64], shift: &[f64]) -> Vec<f64> {
    let mut result = coords.to_vec();
    if result.len() < shift.len() {
        result.resize(shift.len(), 0.0);
    }
    for (index, offset) in shift.iter().enumerate() {
        result[index] += offset;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expands_nested_repeat_lookbacks_and_coordinates() {
        let program = expand_operations(
            &[
                Operation::Measure {
                    qubit: 0,
                    key: None,
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Repeat {
                    count: 2,
                    body: vec![
                        Operation::Measure {
                            qubit: 0,
                            key: None,
                            basis: "Z".to_string(),
                            noise: None,
                        },
                        Operation::DetectorRec {
                            detector_id: None,
                            lookbacks: vec![1, 2],
                            coords: vec![1.0],
                        },
                        Operation::ShiftCoords(vec![0.5]),
                    ],
                },
            ],
            ExpansionMode::Dem,
        )
        .unwrap();

        assert_eq!(program.loop_count, 1);
        assert_eq!(program.operations.len(), 5);
        assert!(matches!(
            &program.operations[2],
            ExpandedOperation::Detector { measurement_ids, .. }
                if measurement_ids == &vec![1, 0]
        ));
        assert!(matches!(
            &program.operations[4],
            ExpandedOperation::Detector { measurement_ids, .. }
                if measurement_ids == &vec![2, 1]
        ));
        assert_eq!(program.detector_coords, vec![vec![1.0], vec![1.5]]);
    }

    #[test]
    fn sampler_mode_uses_the_same_operations_without_coordinate_storage() {
        let operations = [
            Operation::ShiftCoords(vec![3.0, 4.0]),
            Operation::DetectorRec {
                detector_id: Some(7),
                lookbacks: Vec::new(),
                coords: vec![1.0],
            },
        ];
        let sampler = expand_operations(&operations, ExpansionMode::Sampler).unwrap();
        let dem = expand_operations(&operations, ExpansionMode::Dem).unwrap();

        assert_eq!(sampler.operations, dem.operations);
        assert!(sampler.detector_coords.is_empty());
        assert_eq!(dem.detector_coords, vec![vec![4.0, 4.0]]);
    }

    #[test]
    fn repeat_instances_explicit_keys_and_locations() {
        let location = NoiseLocation {
            id: "n".to_string(),
            model: crate::NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.1,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let program = expand_operations(
            &[Operation::Repeat {
                count: 2,
                body: vec![
                    Operation::Noise(location),
                    Operation::Measure {
                        qubit: 0,
                        key: Some("s".to_string()),
                        basis: "Z".to_string(),
                        noise: None,
                    },
                    Operation::Detector {
                        detector_id: Some(10),
                        measurement_keys: vec!["s".to_string()],
                        coords: Vec::new(),
                    },
                ],
            }],
            ExpansionMode::Dem,
        )
        .unwrap();
        assert!(matches!(
            &program.operations[0],
            ExpandedOperation::Noise(0)
        ));
        assert_eq!(
            program
                .location_catalog
                .label(program.noise_locations[0].location_id),
            "n@r[0]"
        );
        assert!(matches!(
            &program.operations[1],
            ExpandedOperation::MeasureSingle {
                measurement_id: 0,
                ..
            }
        ));
        assert_eq!(program.measurement_keys[0], "s@r[0]");
        assert!(matches!(
            &program.operations[5],
            ExpandedOperation::Detector {
                detector_id: 11,
                ..
            }
        ));
    }

    #[test]
    fn assigns_automatic_detector_ids_and_rejects_collisions() {
        let program = expand_operations(
            &[
                Operation::DetectorRec {
                    detector_id: None,
                    lookbacks: Vec::new(),
                    coords: Vec::new(),
                },
                Operation::DetectorRec {
                    detector_id: None,
                    lookbacks: Vec::new(),
                    coords: Vec::new(),
                },
            ],
            ExpansionMode::Dem,
        )
        .unwrap();
        assert!(matches!(
            &program.operations[0],
            ExpandedOperation::Detector { detector_id: 0, .. }
        ));
        assert!(matches!(
            &program.operations[1],
            ExpandedOperation::Detector { detector_id: 1, .. }
        ));

        let error = expand_operations(
            &[
                Operation::DetectorRec {
                    detector_id: Some(3),
                    lookbacks: Vec::new(),
                    coords: Vec::new(),
                },
                Operation::DetectorRec {
                    detector_id: Some(3),
                    lookbacks: Vec::new(),
                    coords: Vec::new(),
                },
            ],
            ExpansionMode::Dem,
        )
        .unwrap_err();
        assert!(error.to_string().contains("duplicate detector id 3"));
    }
}
