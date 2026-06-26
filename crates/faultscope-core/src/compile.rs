use std::collections::HashSet;

use crate::{
    sparse_pauli_to_xz, NoiseLocation, NpError, NpResult, Operation, RunOperation,
    SymbolicStabilizer,
};

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
    let mut runtime_operations = Vec::new();
    let mut seen_noise_ids = HashSet::new();
    let mut noise_location_ids = Vec::new();
    let mut noise_locations = Vec::new();

    for operation in operations {
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
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &pauli)?;
                symbolic.apply_pauli_string(&x, &z);
            }
            Operation::Noise(location) => runtime_operations.push(RunOperation::Noise(location)),
            Operation::Measure {
                qubit,
                key,
                basis,
                noise,
            } => {
                let qubits = vec![qubit];
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &basis)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
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
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &pauli)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
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
                let (x, z) = sparse_pauli_to_xz(n_qubits, &qubits, &basis)?;
                let ideal = symbolic.measure_pauli_expr(&x, &z)?;
                let correction = match basis.as_str() {
                    "Z" => "X",
                    "X" => "Z",
                    "Y" => "X",
                    _ => return Err(NpError::new(format!("unsupported reset basis {basis:?}"))),
                };
                let (cx, cz) = sparse_pauli_to_xz(n_qubits, &qubits, correction)?;
                symbolic.apply_pauli_string_expr(&cx, &cz, &ideal);
                runtime_operations.push(RunOperation::Reset {
                    qubit,
                    key,
                    basis,
                    ideal,
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
}
