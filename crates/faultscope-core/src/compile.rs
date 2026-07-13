use std::collections::HashSet;

use crate::stabilizer::SymbolicStabilizer;
use crate::{Expr, NoiseLocation, NpError, NpResult, Operation, RunOperation};

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledCircuit {
    pub runtime_operations: Vec<RunOperation>,
    pub noise_location_ids: Vec<String>,
    pub noise_locations: Vec<NoiseLocation>,
    pub random_source_count: usize,
}

pub fn compile_runtime_operations(
    n_qubits: usize,
    operations: Vec<Operation>,
) -> NpResult<CompiledCircuit> {
    let mut symbolic = SymbolicStabilizer::zero(n_qubits);
    let operation_count = operations.len();
    let noise_location_count = operations
        .iter()
        .map(|operation| operation.noise_locations().len())
        .sum();
    let mut runtime_operations = Vec::with_capacity(operation_count);
    let mut seen_noise_ids = HashSet::with_capacity(noise_location_count);
    let mut noise_location_ids = Vec::with_capacity(noise_location_count);
    let mut noise_locations = Vec::with_capacity(noise_location_count);
    let reset_reuse = find_measure_reset_reuse(&operations);
    let mut measurement_reused = vec![false; operation_count];
    for measurement_index in reset_reuse.iter().flatten() {
        measurement_reused[*measurement_index] = true;
    }
    let mut cached_measurement_ideals = vec![None; operation_count];

    for (operation_index, operation) in operations.into_iter().enumerate() {
        for location in operation.noise_locations() {
            if !seen_noise_ids.insert(location.id.clone()) {
                return Err(NpError::new(format!(
                    "native sampler requires unique noise location ids; duplicate {:?}",
                    location.id
                )));
            }
            noise_location_ids.push(location.id.clone());
            noise_locations.push(location.clone());
        }

        match operation {
            Operation::H(q) => {
                symbolic.apply_h(q);
                runtime_operations.push(RunOperation::H(q));
            }
            Operation::S(q) => {
                symbolic.apply_s(q);
                runtime_operations.push(RunOperation::S(q));
            }
            Operation::SDag(q) => {
                symbolic.apply_s_dag(q);
                runtime_operations.push(RunOperation::SDag(q));
            }
            Operation::Cx(c, t) => {
                symbolic.apply_cx(c, t);
                runtime_operations.push(RunOperation::Cx(c, t));
            }
            Operation::Cz(l, r) => {
                symbolic.apply_cz(l, r);
                runtime_operations.push(RunOperation::Cz(l, r));
            }
            Operation::Swap(l, r) => {
                symbolic.apply_swap(l, r);
                runtime_operations.push(RunOperation::Swap(l, r));
            }
            Operation::Pauli { qubits, pauli } => {
                symbolic.apply_sparse_pauli_string(&qubits, &pauli)?;
            }
            Operation::Noise(location) => runtime_operations.push(RunOperation::Noise(location)),
            Operation::Measure {
                qubit,
                key,
                basis,
                noise,
            } => {
                let qubits = vec![qubit];
                let ideal = symbolic.measure_sparse_pauli_expr(&qubits, &basis)?;
                if measurement_reused[operation_index] {
                    cached_measurement_ideals[operation_index] = Some(ideal.clone());
                }
                runtime_operations.push(RunOperation::Measure {
                    qubits,
                    pauli: basis,
                    key,
                    ideal,
                    noise,
                });
            }
            Operation::MeasurePauli {
                qubits,
                pauli,
                key,
                noise,
            } => {
                let ideal = symbolic.measure_sparse_pauli_expr(&qubits, &pauli)?;
                runtime_operations.push(RunOperation::Measure {
                    qubits,
                    pauli,
                    key,
                    ideal,
                    noise,
                });
            }
            Operation::Reset { qubit, key, basis } => {
                let qubits = vec![qubit];
                let reused_ideal = reset_reuse[operation_index].and_then(|measurement_index| {
                    cached_measurement_ideals[measurement_index].clone()
                });
                let ideal = match reused_ideal {
                    Some(ideal) => ideal,
                    None => symbolic.measure_sparse_pauli_expr(&qubits, &basis)?,
                };
                let correction = reset_correction(&basis)?;
                symbolic.apply_sparse_pauli_string_expr(&qubits, correction, &ideal)?;
                runtime_operations.push(RunOperation::Reset {
                    qubit,
                    key,
                    basis,
                    ideal: if reset_reuse[operation_index].is_some() {
                        Expr::constant(false)
                    } else {
                        ideal
                    },
                });
            }
            Operation::Detector {
                detector_id,
                measurement_keys,
                ..
            } => runtime_operations.push(RunOperation::Detector {
                detector_id: detector_id.unwrap_or(-1),
                measurement_keys,
            }),
            Operation::ObservableInclude {
                observable_id,
                measurement_keys,
            } => runtime_operations.push(RunOperation::ObservableInclude {
                observable_id,
                measurement_keys,
            }),
        }
    }
    Ok(CompiledCircuit {
        runtime_operations,
        noise_location_ids,
        noise_locations,
        random_source_count: symbolic.random_source_count(),
    })
}

fn find_measure_reset_reuse(operations: &[Operation]) -> Vec<Option<usize>> {
    let mut reuse = vec![None; operations.len()];
    let mut index = 0;
    while index < operations.len() {
        let measurement_start = index;
        while matches!(operations.get(index), Some(Operation::Measure { .. })) {
            index += 1;
        }
        if index == measurement_start {
            index += 1;
            continue;
        }

        let reset_start = index;
        while matches!(
            operations.get(index),
            Some(Operation::Reset { key: None, .. })
        ) {
            index += 1;
        }
        let measurement_count = reset_start - measurement_start;
        let reset_count = index - reset_start;
        if measurement_count != reset_count {
            continue;
        }

        let matching = (0..measurement_count).all(|offset| {
            matches!(
                (&operations[measurement_start + offset], &operations[reset_start + offset]),
                (
                    Operation::Measure {
                        qubit: measurement_qubit,
                        basis: measurement_basis,
                        ..
                    },
                    Operation::Reset {
                        qubit: reset_qubit,
                        basis: reset_basis,
                        key: None,
                    }
                ) if measurement_qubit == reset_qubit && measurement_basis == reset_basis
            )
        });
        if matching {
            for offset in 0..measurement_count {
                reuse[reset_start + offset] = Some(measurement_start + offset);
            }
        }
    }
    reuse
}

fn reset_correction(basis: &str) -> NpResult<&'static str> {
    match basis {
        "Z" => Ok("X"),
        "X" => Ok("Z"),
        "Y" => Ok("X"),
        _ => Err(NpError::new(format!("unsupported reset basis {basis:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NoiseLocation, NoiseModel};
    use std::collections::HashMap;

    #[test]
    fn compiles_random_measurement_to_runtime_op() {
        let compiled = compile_runtime_operations(
            1,
            vec![Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "X".to_string(),
                noise: None,
            }],
        )
        .unwrap();

        assert_eq!(compiled.random_source_count, 1);
        assert!(matches!(
            compiled.runtime_operations.as_slice(),
            [RunOperation::Measure { .. }]
        ));
    }

    #[test]
    fn rejects_duplicate_noise_location_ids() {
        let location = NoiseLocation {
            id: "n".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.1,
            qubits: vec![0],
            tags: HashMap::new(),
        };

        let err = compile_runtime_operations(
            1,
            vec![
                Operation::Noise(location.clone()),
                Operation::Noise(location.clone()),
            ],
        )
        .unwrap_err();

        assert!(err.message().contains("unique noise location ids"));
    }

    #[test]
    fn fuses_adjacent_measure_then_reset_without_changing_runtime_shape() {
        let compiled = compile_runtime_operations(
            1,
            vec![
                Operation::H(0),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Reset {
                    qubit: 0,
                    key: None,
                    basis: "Z".to_string(),
                },
            ],
        )
        .unwrap();

        assert_eq!(compiled.runtime_operations.len(), 3);
        assert_eq!(compiled.random_source_count, 1);
        assert!(matches!(
            compiled.runtime_operations.as_slice(),
            [
                RunOperation::H(0),
                RunOperation::Measure { ideal, .. },
                RunOperation::Reset {
                    ideal: reset_ideal,
                    ..
                }
            ] if ideal.terms() == [0] && reset_ideal.terms().is_empty()
        ));
    }

    #[test]
    fn reuses_measurement_expressions_for_batched_measure_then_reset() {
        let compiled = compile_runtime_operations(
            2,
            vec![
                Operation::H(0),
                Operation::H(1),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Measure {
                    qubit: 1,
                    key: Some("m1".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Reset {
                    qubit: 0,
                    key: None,
                    basis: "Z".to_string(),
                },
                Operation::Reset {
                    qubit: 1,
                    key: None,
                    basis: "Z".to_string(),
                },
            ],
        )
        .unwrap();

        assert_eq!(compiled.runtime_operations.len(), 6);
        assert_eq!(compiled.random_source_count, 2);
        assert!(compiled.runtime_operations[4..].iter().all(|operation| {
            matches!(operation, RunOperation::Reset { ideal, .. } if ideal.terms().is_empty())
        }));
    }

    #[test]
    fn fused_measure_reset_preserves_seeded_masks_and_followup_measurements() {
        let suffix = vec![
            Operation::H(0),
            Operation::Measure {
                qubit: 0,
                key: Some("after".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
        ];
        let mut expanded = vec![
            Operation::H(0),
            Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "Z".to_string(),
                noise: None,
            },
            Operation::Reset {
                qubit: 0,
                key: None,
                basis: "Z".to_string(),
            },
        ];
        expanded.extend(suffix.clone());
        let mut combined = vec![
            Operation::H(0),
            Operation::Reset {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "Z".to_string(),
            },
        ];
        combined.extend(suffix);

        let expanded = compile_runtime_operations(1, expanded).unwrap();
        let combined = compile_runtime_operations(1, combined).unwrap();
        assert_eq!(expanded.random_source_count, combined.random_source_count);
        let expanded_state = crate::run_packed_sample(
            1,
            &expanded.runtime_operations,
            &[],
            &expanded.noise_location_ids,
            130,
            Some(12345),
            false,
        )
        .unwrap();
        let combined_state = crate::run_packed_sample(
            1,
            &combined.runtime_operations,
            &[],
            &combined.noise_location_ids,
            130,
            Some(12345),
            false,
        )
        .unwrap();
        assert_eq!(expanded_state.measurements, combined_state.measurements);
    }

    #[test]
    fn does_not_fuse_measure_reset_across_a_gate_or_basis_change() {
        let compiled = compile_runtime_operations(
            1,
            vec![
                Operation::H(0),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::H(0),
                Operation::Reset {
                    qubit: 0,
                    key: None,
                    basis: "X".to_string(),
                },
            ],
        )
        .unwrap();

        assert_eq!(compiled.runtime_operations.len(), 4);
        assert!(matches!(
            compiled.runtime_operations.last(),
            Some(RunOperation::Reset { ideal, .. }) if !ideal.terms().is_empty()
        ));
    }

    #[test]
    fn does_not_reuse_measurement_expression_across_noise() {
        let location = NoiseLocation {
            id: "between".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.1,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        let compiled = compile_runtime_operations(
            1,
            vec![
                Operation::H(0),
                Operation::Measure {
                    qubit: 0,
                    key: Some("m".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Noise(location),
                Operation::Reset {
                    qubit: 0,
                    key: None,
                    basis: "Z".to_string(),
                },
            ],
        )
        .unwrap();

        assert!(matches!(
            compiled.runtime_operations.last(),
            Some(RunOperation::Reset { ideal, .. }) if !ideal.terms().is_empty()
        ));
    }
}
