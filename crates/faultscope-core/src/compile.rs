use std::collections::HashSet;

use crate::stabilizer::SymbolicStabilizer;
use crate::{Expr, NoiseLocation, NpError, NpResult, Operation, RunOperation};

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledCircuit {
    pub runtime_operations: Vec<RunOperation>,
    pub noise_location_ids: Vec<String>,
    pub noise_locations: Vec<NoiseLocation>,
    pub random_source_count: usize,
    pub stored_operation_count: usize,
    pub logical_operation_count: usize,
    pub loop_kernel_count: usize,
}

pub fn compile_runtime_operations(
    n_qubits: usize,
    operations: Vec<Operation>,
) -> NpResult<CompiledCircuit> {
    let expanded = crate::program::expand_operations(&operations)?;
    let stored_operation_count = expanded.stored_operation_count;
    let logical_operation_count = expanded.logical_operation_count;
    let operations = expanded.operations;
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
    let mut loop_ranges = expanded.loop_ranges;
    loop_ranges.retain(|range| range.iterations.len() >= 2);
    loop_ranges.sort_by(|left, right| {
        let left_start = left.iterations[0].start;
        let right_start = right.iterations[0].start;
        left_start.cmp(&right_start).then_with(|| {
            right
                .iterations
                .last()
                .unwrap()
                .end
                .cmp(&left.iterations.last().unwrap().end)
        })
    });

    let mut cursor = 0usize;
    let mut loop_kernel_count = 0usize;
    for loop_range in loop_ranges {
        let start = loop_range.iterations[0].start;
        let end = loop_range.iterations.last().unwrap().end;
        if start < cursor || end > operations.len() {
            continue;
        }
        compile_regular_segment(
            &mut symbolic,
            &operations[cursor..start],
            &mut runtime_operations,
            &mut seen_noise_ids,
            &mut noise_location_ids,
            &mut noise_locations,
        )?;
        let used_affine_cycle = compile_affine_loop(
            &mut symbolic,
            &operations,
            &loop_range.iterations,
            &mut runtime_operations,
            &mut seen_noise_ids,
            &mut noise_location_ids,
            &mut noise_locations,
        )?;
        loop_kernel_count += usize::from(used_affine_cycle);
        cursor = end;
    }
    compile_regular_segment(
        &mut symbolic,
        &operations[cursor..],
        &mut runtime_operations,
        &mut seen_noise_ids,
        &mut noise_location_ids,
        &mut noise_locations,
    )?;
    Ok(CompiledCircuit {
        runtime_operations,
        noise_location_ids,
        noise_locations,
        random_source_count: symbolic.random_source_count(),
        stored_operation_count,
        logical_operation_count,
        loop_kernel_count,
    })
}

#[allow(clippy::too_many_arguments)]
fn compile_regular_segment(
    symbolic: &mut SymbolicStabilizer,
    operations: &[Operation],
    runtime_operations: &mut Vec<RunOperation>,
    seen_noise_ids: &mut HashSet<String>,
    noise_location_ids: &mut Vec<String>,
    noise_locations: &mut Vec<NoiseLocation>,
) -> NpResult<()> {
    if operations.is_empty() {
        return Ok(());
    }
    register_noise_locations(
        operations,
        seen_noise_ids,
        noise_location_ids,
        noise_locations,
    )?;
    let ideals = simulate_symbolic_segment(symbolic, operations)?;
    emit_runtime_segment(operations, &ideals, runtime_operations)
}

#[allow(clippy::too_many_arguments)]
fn compile_affine_loop(
    symbolic: &mut SymbolicStabilizer,
    operations: &[Operation],
    iterations: &[std::ops::Range<usize>],
    runtime_operations: &mut Vec<RunOperation>,
    seen_noise_ids: &mut HashSet<String>,
    noise_location_ids: &mut Vec<String>,
    noise_locations: &mut Vec<NoiseLocation>,
) -> NpResult<bool> {
    let iteration_len = iterations[0].end - iterations[0].start;
    if iterations
        .iter()
        .any(|iteration| iteration.end - iteration.start != iteration_len)
    {
        for iteration in iterations {
            compile_regular_segment(
                symbolic,
                &operations[iteration.clone()],
                runtime_operations,
                seen_noise_ids,
                noise_location_ids,
                noise_locations,
            )?;
        }
        return Ok(false);
    }
    let mut iteration_index = 0usize;
    while iteration_index < iterations.len() {
        let iteration = &iterations[iteration_index];
        let body = &operations[iteration.clone()];
        let mut template = symbolic.affine_template();
        let template_ideals = simulate_symbolic_segment(&mut template, body)?;
        if !symbolic.support_equals(&template) {
            compile_regular_segment(
                symbolic,
                body,
                runtime_operations,
                seen_noise_ids,
                noise_location_ids,
                noise_locations,
            )?;
            iteration_index += 1;
            continue;
        }
        let basis_count = symbolic.signs().len();
        let local_random_count = template
            .random_source_count()
            .checked_sub(basis_count)
            .ok_or_else(|| NpError::new("invalid affine loop random-source count"))?;
        for remaining in &iterations[iteration_index..] {
            let body = &operations[remaining.clone()];
            register_noise_locations(body, seen_noise_ids, noise_location_ids, noise_locations)?;
            let input_signs = symbolic.signs().to_vec();
            let source_base = symbolic.random_source_count();
            let ideals = template_ideals
                .iter()
                .map(|ideal| {
                    ideal
                        .as_ref()
                        .map(|ideal| substitute_affine_expr(ideal, &input_signs, source_base))
                })
                .collect::<Vec<_>>();
            emit_runtime_segment(body, &ideals, runtime_operations)?;
            let signs = template
                .signs()
                .iter()
                .map(|sign| substitute_affine_expr(sign, &input_signs, source_base))
                .collect();
            symbolic.replace_affine_state(
                signs,
                source_base
                    .checked_add(local_random_count)
                    .ok_or_else(|| NpError::new("random-source count overflow"))?,
            );
        }
        return Ok(true);
    }
    Ok(false)
}

fn substitute_affine_expr(template: &Expr, input_signs: &[Expr], source_base: usize) -> Expr {
    let mut result = Expr::constant(template.constant_value());
    let basis_count = input_signs.len();
    for term in template.terms() {
        if *term < basis_count {
            result.xor_assign(&input_signs[*term]);
        } else {
            result.xor_assign(&Expr::random(source_base + term - basis_count));
        }
    }
    result
}

fn simulate_symbolic_segment(
    symbolic: &mut SymbolicStabilizer,
    operations: &[Operation],
) -> NpResult<Vec<Option<Expr>>> {
    let reset_reuse = find_measure_reset_reuse(operations);
    let mut measurement_reused = vec![false; operations.len()];
    for measurement_index in reset_reuse.iter().flatten() {
        measurement_reused[*measurement_index] = true;
    }
    let mut cached_measurement_ideals = vec![None; operations.len()];
    let mut ideals = vec![None; operations.len()];
    for (operation_index, operation) in operations.iter().enumerate() {
        match operation {
            Operation::H(qubit) => symbolic.apply_h(*qubit),
            Operation::S(qubit) => symbolic.apply_s(*qubit),
            Operation::SDag(qubit) => symbolic.apply_s_dag(*qubit),
            Operation::Cx(control, target) => symbolic.apply_cx(*control, *target),
            Operation::Cz(left, right) => symbolic.apply_cz(*left, *right),
            Operation::Swap(left, right) => symbolic.apply_swap(*left, *right),
            Operation::Pauli { qubits, pauli } => {
                symbolic.apply_sparse_pauli_string(qubits, pauli)?;
            }
            Operation::Measure { qubit, basis, .. } => {
                let ideal = symbolic.measure_sparse_pauli_expr(&[*qubit], basis)?;
                if measurement_reused[operation_index] {
                    cached_measurement_ideals[operation_index] = Some(ideal.clone());
                }
                ideals[operation_index] = Some(ideal);
            }
            Operation::MeasurePauli { qubits, pauli, .. } => {
                ideals[operation_index] = Some(symbolic.measure_sparse_pauli_expr(qubits, pauli)?);
            }
            Operation::Reset { qubit, basis, .. } => {
                let reused_ideal = reset_reuse[operation_index].and_then(|measurement_index| {
                    cached_measurement_ideals[measurement_index].clone()
                });
                let ideal = match reused_ideal {
                    Some(ideal) => ideal,
                    None => symbolic.measure_sparse_pauli_expr(&[*qubit], basis)?,
                };
                symbolic.apply_sparse_pauli_string_expr(
                    &[*qubit],
                    reset_correction(basis)?,
                    &ideal,
                )?;
                ideals[operation_index] = Some(if reset_reuse[operation_index].is_some() {
                    Expr::constant(false)
                } else {
                    ideal
                });
            }
            Operation::Noise(_)
            | Operation::Detector { .. }
            | Operation::ObservableInclude { .. } => {}
            Operation::Tick
            | Operation::ShiftCoords(_)
            | Operation::Repeat { .. }
            | Operation::MeasureReset { .. }
            | Operation::DetectorRec { .. }
            | Operation::ObservableIncludeRec { .. } => {
                return Err(NpError::new(
                    "internal error: structured operation survived expansion",
                ));
            }
        }
    }
    Ok(ideals)
}

fn emit_runtime_segment(
    operations: &[Operation],
    ideals: &[Option<Expr>],
    runtime_operations: &mut Vec<RunOperation>,
) -> NpResult<()> {
    for (operation_index, operation) in operations.iter().enumerate() {
        match operation {
            Operation::H(qubit) => runtime_operations.push(RunOperation::H(*qubit)),
            Operation::S(qubit) => runtime_operations.push(RunOperation::S(*qubit)),
            Operation::SDag(qubit) => runtime_operations.push(RunOperation::SDag(*qubit)),
            Operation::Cx(control, target) => {
                runtime_operations.push(RunOperation::Cx(*control, *target));
            }
            Operation::Cz(left, right) => {
                runtime_operations.push(RunOperation::Cz(*left, *right));
            }
            Operation::Swap(left, right) => {
                runtime_operations.push(RunOperation::Swap(*left, *right));
            }
            Operation::Pauli { .. } => {}
            Operation::Noise(location) => {
                runtime_operations.push(RunOperation::Noise(location.clone()));
            }
            Operation::Measure {
                qubit,
                key,
                basis,
                noise,
            } => runtime_operations.push(RunOperation::Measure {
                qubits: vec![*qubit],
                pauli: basis.clone(),
                key: key.clone(),
                ideal: ideals[operation_index]
                    .clone()
                    .ok_or_else(|| NpError::new("missing compiled measurement expression"))?,
                noise: noise.clone(),
            }),
            Operation::MeasurePauli {
                qubits,
                pauli,
                key,
                noise,
            } => runtime_operations.push(RunOperation::Measure {
                qubits: qubits.clone(),
                pauli: pauli.clone(),
                key: key.clone(),
                ideal: ideals[operation_index]
                    .clone()
                    .ok_or_else(|| NpError::new("missing compiled measurement expression"))?,
                noise: noise.clone(),
            }),
            Operation::Reset { qubit, key, basis } => {
                runtime_operations.push(RunOperation::Reset {
                    qubit: *qubit,
                    key: key.clone(),
                    basis: basis.clone(),
                    ideal: ideals[operation_index]
                        .clone()
                        .ok_or_else(|| NpError::new("missing compiled reset expression"))?,
                });
            }
            Operation::Detector {
                detector_id,
                measurement_keys,
                ..
            } => runtime_operations.push(RunOperation::Detector {
                detector_id: detector_id.unwrap_or(-1),
                measurement_keys: measurement_keys.clone(),
            }),
            Operation::ObservableInclude {
                observable_id,
                measurement_keys,
            } => runtime_operations.push(RunOperation::ObservableInclude {
                observable_id: *observable_id,
                measurement_keys: measurement_keys.clone(),
            }),
            Operation::Tick
            | Operation::ShiftCoords(_)
            | Operation::Repeat { .. }
            | Operation::MeasureReset { .. }
            | Operation::DetectorRec { .. }
            | Operation::ObservableIncludeRec { .. } => {
                return Err(NpError::new(
                    "internal error: structured operation survived expansion",
                ));
            }
        }
    }
    Ok(())
}

fn register_noise_locations(
    operations: &[Operation],
    seen_noise_ids: &mut HashSet<String>,
    noise_location_ids: &mut Vec<String>,
    noise_locations: &mut Vec<NoiseLocation>,
) -> NpResult<()> {
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
    }
    Ok(())
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
    fn repeat_compilation_matches_explicit_expansion_with_same_seed() {
        let repeated = vec![Operation::Repeat {
            count: 3,
            body: vec![
                Operation::H(0),
                Operation::MeasureReset {
                    qubit: 0,
                    basis: "Z".to_string(),
                },
                Operation::DetectorRec {
                    detector_id: None,
                    lookbacks: vec![1],
                    coords: Vec::new(),
                },
            ],
        }];
        let mut expanded = Vec::new();
        for index in 0..3 {
            expanded.push(Operation::H(0));
            expanded.push(Operation::Reset {
                qubit: 0,
                key: Some(format!("m{index}")),
                basis: "Z".to_string(),
            });
            expanded.push(Operation::Detector {
                detector_id: None,
                measurement_keys: vec![format!("m{index}")],
                coords: Vec::new(),
            });
        }
        let repeated = compile_runtime_operations(1, repeated).unwrap();
        let expanded = compile_runtime_operations(1, expanded).unwrap();
        assert_eq!(repeated.random_source_count, expanded.random_source_count);
        let repeated_batch = crate::run_packed_sample(
            1,
            &repeated.runtime_operations,
            &[],
            &[],
            129,
            Some(123),
            false,
        )
        .unwrap();
        let expanded_batch = crate::run_packed_sample(
            1,
            &expanded.runtime_operations,
            &[],
            &[],
            129,
            Some(123),
            false,
        )
        .unwrap();
        assert_eq!(repeated_batch.measurements, expanded_batch.measurements);
        assert_eq!(repeated_batch.detectors, expanded_batch.detectors);
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
