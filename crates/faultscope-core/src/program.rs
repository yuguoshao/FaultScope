use std::collections::{HashMap, HashSet};
use std::ops::Range;

use crate::{Circuit, NoiseLocation, NpError, NpResult, Operation};

/// A fully indexed reference expansion of a structured operation tree.
///
/// Runtime compilation and DEM generation use this as their semantic source of
/// truth. Loop-aware fast paths may avoid performing the expensive tableau work
/// for every item, but must produce the same result as this expansion.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ExpandedProgram {
    pub(crate) operations: Vec<Operation>,
    pub(crate) stored_operation_count: usize,
    pub(crate) logical_operation_count: usize,
    pub(crate) loop_count: usize,
    pub(crate) loop_ranges: Vec<LoopRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LoopRange {
    pub(crate) iterations: Vec<Range<usize>>,
}

pub(crate) fn expand_circuit(circuit: &Circuit) -> NpResult<(Circuit, ExpandedProgram)> {
    let program = expand_operations(&circuit.operations)?;
    Ok((
        Circuit {
            n_qubits: circuit.n_qubits,
            operations: program.operations.clone(),
        },
        program,
    ))
}

pub fn expand_circuit_operations(circuit: &Circuit) -> NpResult<Circuit> {
    Ok(expand_circuit(circuit)?.0)
}

pub(crate) fn expand_operations(operations: &[Operation]) -> NpResult<ExpandedProgram> {
    let stored_operation_count = stored_count(operations)?;
    let mut state = ExpansionState::default();
    let mut site_path = Vec::new();
    state.expand_sequence(operations, &mut site_path)?;
    let logical_operation_count = state.output.len();
    Ok(ExpandedProgram {
        operations: state.output,
        stored_operation_count,
        logical_operation_count,
        loop_count: state.loop_count,
        loop_ranges: state.loop_ranges,
    })
}

fn stored_count(operations: &[Operation]) -> NpResult<usize> {
    let mut count = 0usize;
    for operation in operations {
        count = count
            .checked_add(1)
            .ok_or_else(|| NpError::new("stored operation count overflow"))?;
        if let Operation::Repeat { body, .. } = operation {
            count = count
                .checked_add(stored_count(body)?)
                .ok_or_else(|| NpError::new("stored operation count overflow"))?;
        }
    }
    Ok(count)
}

#[derive(Default)]
struct ExpansionState {
    output: Vec<Operation>,
    measurement_history: Vec<String>,
    key_aliases: HashMap<String, String>,
    coordinate_shift: Vec<f64>,
    repeat_path: Vec<usize>,
    detector_site_instances: HashMap<Vec<usize>, i64>,
    seen_detector_ids: HashSet<i64>,
    detector_ordinal: i64,
    seen_keys: HashSet<String>,
    loop_count: usize,
    loop_ranges: Vec<LoopRange>,
}

impl ExpansionState {
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
            Operation::ShiftCoords(offsets) => add_coords(&mut self.coordinate_shift, offsets),
            Operation::Repeat { count, body } => {
                if *count == 0 {
                    return Err(NpError::new("repeat count must be positive"));
                }
                self.loop_count = self
                    .loop_count
                    .checked_add(1)
                    .ok_or_else(|| NpError::new("loop count overflow"))?;
                self.output
                    .len()
                    .checked_add(
                        body.len()
                            .checked_mul(*count)
                            .ok_or_else(|| NpError::new("logical operation count overflow"))?,
                    )
                    .ok_or_else(|| NpError::new("logical operation count overflow"))?;
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
            Operation::H(qubit) => self.output.push(Operation::H(*qubit)),
            Operation::S(qubit) => self.output.push(Operation::S(*qubit)),
            Operation::SDag(qubit) => self.output.push(Operation::SDag(*qubit)),
            Operation::Cx(control, target) => self.output.push(Operation::Cx(*control, *target)),
            Operation::Cz(left, right) => self.output.push(Operation::Cz(*left, *right)),
            Operation::Swap(left, right) => self.output.push(Operation::Swap(*left, *right)),
            Operation::Pauli { qubits, pauli } => self.output.push(Operation::Pauli {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
            }),
            Operation::Noise(location) => self
                .output
                .push(Operation::Noise(self.instantiate_location(location))),
            Operation::Measure {
                qubit,
                key,
                basis,
                noise,
            } => {
                let key = self.measurement_key(key.as_deref())?;
                self.output.push(Operation::Measure {
                    qubit: *qubit,
                    key: Some(key),
                    basis: basis.clone(),
                    noise: noise
                        .as_ref()
                        .map(|location| self.instantiate_location(location)),
                });
            }
            Operation::MeasurePauli {
                qubits,
                pauli,
                key,
                noise,
            } => {
                let key = self.measurement_key(key.as_deref())?;
                self.output.push(Operation::MeasurePauli {
                    qubits: qubits.clone(),
                    pauli: pauli.clone(),
                    key: Some(key),
                    noise: noise
                        .as_ref()
                        .map(|location| self.instantiate_location(location)),
                });
            }
            Operation::MeasureReset { qubit, basis } => {
                let key = self.measurement_key(None)?;
                self.output.push(Operation::Reset {
                    qubit: *qubit,
                    key: Some(key),
                    basis: basis.clone(),
                });
            }
            Operation::Reset { qubit, key, basis } => {
                let key = key
                    .as_deref()
                    .map(|key| self.measurement_key(Some(key)))
                    .transpose()?;
                self.output.push(Operation::Reset {
                    qubit: *qubit,
                    key,
                    basis: basis.clone(),
                });
            }
            Operation::Detector {
                detector_id,
                measurement_keys,
                coords,
            } => {
                let detector_id = self.detector_id(*detector_id, site_path)?;
                let measurement_keys = measurement_keys
                    .iter()
                    .map(|key| {
                        self.key_aliases
                            .get(key)
                            .cloned()
                            .unwrap_or_else(|| key.clone())
                    })
                    .collect();
                self.output.push(Operation::Detector {
                    detector_id,
                    measurement_keys,
                    coords: shifted_coords(coords, &self.coordinate_shift),
                });
            }
            Operation::DetectorRec {
                detector_id,
                lookbacks,
                coords,
            } => {
                let detector_id = self.detector_id(*detector_id, site_path)?;
                let measurement_keys = self.resolve_lookbacks(lookbacks)?;
                self.output.push(Operation::Detector {
                    detector_id,
                    measurement_keys,
                    coords: shifted_coords(coords, &self.coordinate_shift),
                });
            }
            Operation::ObservableInclude {
                observable_id,
                measurement_keys,
            } => self.output.push(Operation::ObservableInclude {
                observable_id: *observable_id,
                measurement_keys: measurement_keys
                    .iter()
                    .map(|key| {
                        self.key_aliases
                            .get(key)
                            .cloned()
                            .unwrap_or_else(|| key.clone())
                    })
                    .collect(),
            }),
            Operation::ObservableIncludeRec {
                observable_id,
                lookbacks,
            } => self.output.push(Operation::ObservableInclude {
                observable_id: *observable_id,
                measurement_keys: self.resolve_lookbacks(lookbacks)?,
            }),
        }
        Ok(())
    }

    fn measurement_key(&mut self, explicit: Option<&str>) -> NpResult<String> {
        let key = match explicit {
            Some(base) if !self.repeat_path.is_empty() => {
                let key = format!("{base}@r{}", format_repeat_path(&self.repeat_path));
                self.key_aliases.insert(base.to_string(), key.clone());
                key
            }
            Some(base) => {
                self.key_aliases.insert(base.to_string(), base.to_string());
                base.to_string()
            }
            None => format!("m{}", self.measurement_history.len()),
        };
        if !self.seen_keys.insert(key.clone()) {
            return Err(NpError::new(format!("duplicate measurement key {key:?}")));
        }
        self.measurement_history.push(key.clone());
        Ok(key)
    }

    fn instantiate_location(&self, location: &NoiseLocation) -> NoiseLocation {
        if self.repeat_path.is_empty() {
            return location.clone();
        }
        let mut location = location.clone();
        location.id = format!("{}@r{}", location.id, format_repeat_path(&self.repeat_path));
        location
    }

    fn detector_id(
        &mut self,
        detector_id: Option<i64>,
        site_path: &[usize],
    ) -> NpResult<Option<i64>> {
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
        Ok(Some(id))
    }

    fn resolve_lookbacks(&self, lookbacks: &[usize]) -> NpResult<Vec<String>> {
        lookbacks
            .iter()
            .map(|lookback| {
                if *lookback == 0 || *lookback > self.measurement_history.len() {
                    return Err(NpError::new(format!(
                        "measurement record lookback {lookback} is out of range"
                    )));
                }
                Ok(self.measurement_history[self.measurement_history.len() - lookback].clone())
            })
            .collect()
    }
}

fn format_repeat_path(path: &[usize]) -> String {
    path.iter()
        .map(|index| format!("[{index}]"))
        .collect::<String>()
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
        let program = expand_operations(&[
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
        ])
        .unwrap();

        assert_eq!(program.loop_count, 1);
        assert_eq!(program.operations.len(), 5);
        assert!(matches!(
            &program.operations[2],
            Operation::Detector { measurement_keys, coords, .. }
                if measurement_keys == &vec!["m1".to_string(), "m0".to_string()]
                    && coords == &vec![1.0]
        ));
        assert!(matches!(
            &program.operations[4],
            Operation::Detector { measurement_keys, coords, .. }
                if measurement_keys == &vec!["m2".to_string(), "m1".to_string()]
                    && coords == &vec![1.5]
        ));
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
        let program = expand_operations(&[Operation::Repeat {
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
        }])
        .unwrap();
        assert!(
            matches!(&program.operations[0], Operation::Noise(location) if location.id == "n@r[0]")
        );
        assert!(
            matches!(&program.operations[1], Operation::Measure { key: Some(key), .. } if key == "s@r[0]")
        );
        assert!(matches!(
            &program.operations[5],
            Operation::Detector {
                detector_id: Some(11),
                ..
            }
        ));
    }

    #[test]
    fn assigns_automatic_detector_ids_and_rejects_collisions() {
        let program = expand_operations(&[
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
        ])
        .unwrap();
        assert!(matches!(
            &program.operations[0],
            Operation::Detector {
                detector_id: Some(0),
                ..
            }
        ));
        assert!(matches!(
            &program.operations[1],
            Operation::Detector {
                detector_id: Some(1),
                ..
            }
        ));

        let error = expand_operations(&[
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
        ])
        .unwrap_err();
        assert!(error.to_string().contains("duplicate detector id 3"));
    }
}
