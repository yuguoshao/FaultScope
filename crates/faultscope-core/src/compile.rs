use crate::program::{ExpandedOperation, ExpansionMode};
use crate::stabilizer::SymbolicStabilizer;
use crate::{
    Expr, IndexedNoiseLocation, LocationCatalog, LogicalObservable, NpError, NpResult, Operation,
    RuntimeCapacities, SamplerObservable, SamplerOperation, SamplerProgram,
};

#[cfg(test)]
thread_local! {
    static SYMBOLIC_OPERATION_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compile a borrowed operation tree into the canonical compact sampler program.
pub fn compile_sampler_program_ref(
    n_qubits: usize,
    operations: &[Operation],
    observables: Vec<LogicalObservable>,
) -> NpResult<SamplerProgram> {
    crate::model::validate_operations(n_qubits, operations)?;
    for observable in &observables {
        observable.validate_for_n_qubits(n_qubits)?;
    }
    let expanded = crate::program::expand_operations(operations, ExpansionMode::Sampler)?;
    compile_expanded_program(n_qubits, expanded, observables)
}

struct SymbolicCompilation {
    operations: Vec<ExpandedOperation>,
    ideals: Vec<Expr>,
    measurement_keys: Vec<String>,
    noise_locations: Vec<IndexedNoiseLocation>,
    location_catalog: LocationCatalog,
    random_source_count: usize,
    stored_operation_count: usize,
    logical_operation_count: usize,
    loop_kernel_count: usize,
}

fn compile_expanded_program(
    n_qubits: usize,
    expanded: crate::program::ExpandedProgram,
    observables: Vec<LogicalObservable>,
) -> NpResult<SamplerProgram> {
    let crate::program::ExpandedProgram {
        operations,
        measurement_keys,
        noise_locations,
        location_catalog,
        stored_operation_count,
        logical_operation_count,
        ideal_count,
        mut loop_ranges,
        ..
    } = expanded;
    let mut symbolic = SymbolicStabilizer::zero(n_qubits);
    let mut ideals = Vec::with_capacity(ideal_count);
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
        compile_regular_segment(&mut symbolic, &operations[cursor..start], &mut ideals)?;
        let used_affine_cycle = compile_affine_loop(
            &mut symbolic,
            &operations,
            &loop_range.iterations,
            &mut ideals,
        )?;
        loop_kernel_count += usize::from(used_affine_cycle);
        cursor = end;
    }
    compile_regular_segment(&mut symbolic, &operations[cursor..], &mut ideals)?;
    emit_sampler_program(
        n_qubits,
        SymbolicCompilation {
            operations,
            ideals,
            measurement_keys,
            noise_locations,
            location_catalog,
            random_source_count: symbolic.random_source_count(),
            stored_operation_count,
            logical_operation_count,
            loop_kernel_count,
        },
        observables,
    )
}

fn compile_regular_segment(
    symbolic: &mut SymbolicStabilizer,
    operations: &[ExpandedOperation],
    ideals: &mut Vec<Expr>,
) -> NpResult<()> {
    if operations.is_empty() {
        return Ok(());
    }
    simulate_symbolic_segment(symbolic, operations, ideals)
}

fn compile_affine_loop(
    symbolic: &mut SymbolicStabilizer,
    operations: &[ExpandedOperation],
    iterations: &[std::ops::Range<usize>],
    ideals: &mut Vec<Expr>,
) -> NpResult<bool> {
    let iteration_len = iterations[0].end - iterations[0].start;
    let has_clifford = operations[iterations[0].clone()].iter().any(|operation| {
        matches!(
            operation,
            ExpandedOperation::H(_)
                | ExpandedOperation::S(_)
                | ExpandedOperation::SDag(_)
                | ExpandedOperation::Cx(_, _)
                | ExpandedOperation::Cz(_, _)
                | ExpandedOperation::Swap(_, _)
        )
    });
    let minimum_iterations = if has_clifford { 8 } else { 4 };
    if iterations.len() < minimum_iterations
        || iteration_len < symbolic.n_qubits().saturating_mul(2)
        || iterations
            .iter()
            .any(|iteration| iteration.end - iteration.start != iteration_len)
    {
        for iteration in iterations {
            compile_regular_segment(symbolic, &operations[iteration.clone()], ideals)?;
        }
        return Ok(false);
    }

    let mut template = symbolic.affine_template();
    let mut template_ideals = Vec::new();
    simulate_symbolic_segment(
        &mut template,
        &operations[iterations[0].clone()],
        &mut template_ideals,
    )?;
    let mut period = usize::from(symbolic.support_equals(&template));
    let mut probed_iterations = 1usize;
    if period == 0 && has_clifford {
        simulate_symbolic_segment(
            &mut template,
            &operations[iterations[1].clone()],
            &mut template_ideals,
        )?;
        probed_iterations = 2;
        if symbolic.support_equals(&template) {
            period = 2;
        }
    }

    let basis_count = symbolic.signs().len();
    let local_random_count = template
        .random_source_count()
        .checked_sub(basis_count)
        .ok_or_else(|| NpError::new("invalid affine loop random-source count"))?;
    let input_signs = symbolic.signs().to_vec();
    let source_base = symbolic.random_source_count();
    instantiate_affine_ideals_into(&template_ideals, &input_signs, source_base, ideals);
    let probed_signs = instantiate_affine_signs(&template, &input_signs, source_base);
    let probed_random_source_count = source_base
        .checked_add(local_random_count)
        .ok_or_else(|| NpError::new("random-source count overflow"))?;

    let reusable_periods = (iterations.len() - probed_iterations)
        .checked_div(period)
        .unwrap_or(0);
    let use_affine_cycle = period != 0
        && reusable_periods >= 2
        && affine_template_is_compact(template.signs(), &template_ideals);
    if !use_affine_cycle {
        symbolic.replace_with_affine_template(template, probed_signs, probed_random_source_count);
        let remaining = &iterations[probed_iterations..];
        for iteration in remaining {
            compile_regular_segment(symbolic, &operations[iteration.clone()], ideals)?;
        }
        return Ok(false);
    }

    symbolic.replace_affine_state(probed_signs, probed_random_source_count);
    let mut next_iteration = probed_iterations;
    while next_iteration + period <= iterations.len() {
        let input_signs = symbolic.signs().to_vec();
        let source_base = symbolic.random_source_count();
        instantiate_affine_ideals_into(&template_ideals, &input_signs, source_base, ideals);
        let signs = instantiate_affine_signs(&template, &input_signs, source_base);
        symbolic.replace_affine_state(
            signs,
            source_base
                .checked_add(local_random_count)
                .ok_or_else(|| NpError::new("random-source count overflow"))?,
        );
        next_iteration += period;
    }
    for remaining in &iterations[next_iteration..] {
        compile_regular_segment(symbolic, &operations[remaining.clone()], ideals)?;
    }
    Ok(true)
}

fn affine_template_is_compact(signs: &[Expr], ideals: &[Expr]) -> bool {
    let expression_count = signs.len() + ideals.len();
    let expression_term_count = signs
        .iter()
        .chain(ideals)
        .map(|expr| expr.terms().len())
        .sum::<usize>();
    expression_term_count <= expression_count.saturating_mul(16)
}

fn instantiate_affine_ideals_into(
    template_ideals: &[Expr],
    input_signs: &[Expr],
    source_base: usize,
    ideals: &mut Vec<Expr>,
) {
    ideals.extend(
        template_ideals
            .iter()
            .map(|ideal| substitute_affine_expr(ideal, input_signs, source_base)),
    );
}

fn instantiate_affine_signs(
    template: &SymbolicStabilizer,
    input_signs: &[Expr],
    source_base: usize,
) -> Vec<Expr> {
    template
        .signs()
        .iter()
        .map(|sign| substitute_affine_expr(sign, input_signs, source_base))
        .collect()
}

fn substitute_affine_expr(template: &Expr, input_signs: &[Expr], source_base: usize) -> Expr {
    let mut result = Expr::constant(template.constant_value());
    let basis_count = input_signs.len();
    for term in template.terms() {
        if *term < basis_count {
            result.xor_assign(&input_signs[*term]);
        } else {
            result.toggle_term(source_base + term - basis_count);
        }
    }
    result
}

fn simulate_symbolic_segment(
    symbolic: &mut SymbolicStabilizer,
    operations: &[ExpandedOperation],
    ideals: &mut Vec<Expr>,
) -> NpResult<()> {
    #[cfg(test)]
    SYMBOLIC_OPERATION_COUNT.with(|count| count.set(count.get() + operations.len()));

    let reset_reuse = find_measure_reset_reuse(operations);
    let mut reuse_index = 0usize;
    let mut operation_index = 0usize;
    while operation_index < operations.len() {
        if let Some(reuse) = reset_reuse.get(reuse_index) {
            if reuse.measurement_start == operation_index {
                let mut cached_ideals = Vec::with_capacity(reuse.count);
                for operation in
                    &operations[reuse.measurement_start..reuse.measurement_start + reuse.count]
                {
                    let ExpandedOperation::MeasureSingle { qubit, basis, .. } = operation else {
                        return Err(NpError::new(
                            "internal error: invalid fused measurement block",
                        ));
                    };
                    let ideal = symbolic.measure_single_pauli_expr(*qubit, basis.as_str())?;
                    ideals.push(ideal.clone());
                    cached_ideals.push(ideal);
                }
                for (offset, ideal) in cached_ideals.iter().enumerate() {
                    let reset_index = reuse.measurement_start + reuse.count + offset;
                    let ExpandedOperation::Reset { qubit, basis, .. } = &operations[reset_index]
                    else {
                        return Err(NpError::new("internal error: invalid fused reset block"));
                    };
                    symbolic.apply_single_pauli_string_expr(
                        *qubit,
                        reset_correction(basis.as_str())?,
                        ideal,
                    )?;
                    ideals.push(Expr::constant(false));
                }
                operation_index += reuse.count * 2;
                reuse_index += 1;
                continue;
            }
        }

        match &operations[operation_index] {
            ExpandedOperation::H(qubit) => symbolic.apply_h(*qubit),
            ExpandedOperation::S(qubit) => symbolic.apply_s(*qubit),
            ExpandedOperation::SDag(qubit) => symbolic.apply_s_dag(*qubit),
            ExpandedOperation::Cx(control, target) => symbolic.apply_cx(*control, *target),
            ExpandedOperation::Cz(left, right) => symbolic.apply_cz(*left, *right),
            ExpandedOperation::Swap(left, right) => symbolic.apply_swap(*left, *right),
            ExpandedOperation::Pauli { qubits, pauli } => {
                symbolic.apply_sparse_pauli_string(qubits, pauli)?;
            }
            ExpandedOperation::MeasureSingle { qubit, basis, .. } => {
                ideals.push(symbolic.measure_single_pauli_expr(*qubit, basis.as_str())?);
            }
            ExpandedOperation::MeasurePauli { qubits, pauli, .. } => {
                ideals.push(symbolic.measure_sparse_pauli_expr(qubits, pauli)?);
            }
            ExpandedOperation::Reset { qubit, basis, .. } => {
                let basis = basis.as_str();
                let ideal = symbolic.measure_single_pauli_expr(*qubit, basis)?;
                symbolic.apply_single_pauli_string_expr(
                    *qubit,
                    reset_correction(basis)?,
                    &ideal,
                )?;
                ideals.push(ideal);
            }
            ExpandedOperation::Noise(_)
            | ExpandedOperation::Detector { .. }
            | ExpandedOperation::ObservableInclude { .. } => {}
        }
        operation_index += 1;
    }
    Ok(())
}

fn emit_sampler_program(
    n_qubits: usize,
    compiled: SymbolicCompilation,
    observables: Vec<LogicalObservable>,
) -> NpResult<SamplerProgram> {
    let mut sampler_operations = Vec::with_capacity(compiled.operations.len());
    let mut ideals = compiled.ideals.into_iter();
    let mut capacities = RuntimeCapacities {
        random_sources: compiled.random_source_count,
        ..RuntimeCapacities::default()
    };
    for operation in compiled.operations {
        match operation {
            ExpandedOperation::H(qubit) => {
                sampler_operations.push(SamplerOperation::H(qubit));
            }
            ExpandedOperation::S(qubit) => {
                sampler_operations.push(SamplerOperation::S(qubit));
            }
            ExpandedOperation::SDag(qubit) => {
                sampler_operations.push(SamplerOperation::SDag(qubit));
            }
            ExpandedOperation::Cx(control, target) => {
                sampler_operations.push(SamplerOperation::Cx(control, target));
            }
            ExpandedOperation::Cz(left, right) => {
                sampler_operations.push(SamplerOperation::Cz(left, right));
            }
            ExpandedOperation::Swap(left, right) => {
                sampler_operations.push(SamplerOperation::Swap(left, right));
            }
            ExpandedOperation::Pauli { .. } => {}
            ExpandedOperation::Noise(noise_id) => {
                sampler_operations.push(SamplerOperation::Noise(noise_id));
            }
            ExpandedOperation::MeasureSingle {
                qubit,
                basis,
                measurement_id,
                noise,
            } => {
                sampler_operations.push(SamplerOperation::MeasureSingle {
                    qubit,
                    basis,
                    measurement_id,
                    ideal: ideals
                        .next()
                        .ok_or_else(|| NpError::new("missing compiled measurement expression"))?,
                    noise,
                });
                capacities.measurements += 1;
            }
            ExpandedOperation::MeasurePauli {
                qubits,
                pauli,
                measurement_id,
                noise,
            } => {
                sampler_operations.push(SamplerOperation::MeasurePauli {
                    qubits,
                    pauli,
                    measurement_id,
                    ideal: ideals
                        .next()
                        .ok_or_else(|| NpError::new("missing compiled measurement expression"))?,
                    noise,
                });
                capacities.measurements += 1;
            }
            ExpandedOperation::Reset {
                qubit,
                measurement_id,
                basis,
            } => {
                capacities.measurements += usize::from(measurement_id.is_some());
                sampler_operations.push(SamplerOperation::Reset {
                    qubit,
                    measurement_id,
                    basis,
                    ideal: ideals
                        .next()
                        .ok_or_else(|| NpError::new("missing compiled reset expression"))?,
                });
            }
            ExpandedOperation::Detector {
                detector_id,
                measurement_ids,
                ..
            } => {
                sampler_operations.push(SamplerOperation::Detector {
                    detector_id,
                    measurement_ids,
                });
                capacities.detectors += 1;
            }
            ExpandedOperation::ObservableInclude {
                observable_id,
                measurement_ids,
            } => {
                sampler_operations.push(SamplerOperation::ObservableInclude {
                    observable_id,
                    measurement_ids,
                });
                capacities.observables += 1;
            }
        }
    }
    if ideals.next().is_some() {
        return Err(NpError::new(
            "internal error: unused compiled measurement expression",
        ));
    }

    let mut measurement_keys = compiled.measurement_keys;
    let compiled_observables = observables
        .iter()
        .map(|observable| SamplerObservable {
            id: observable.id,
            measurement_ids: observable
                .measurement_keys
                .iter()
                .map(|key| resolve_declared_measurement_key(key, &mut measurement_keys))
                .collect(),
            pauli_qubits: observable.pauli_qubits.clone(),
            pauli: observable.pauli.clone(),
        })
        .collect();
    Ok(SamplerProgram::from_compiled_parts(
        n_qubits,
        sampler_operations,
        observables,
        compiled_observables,
        measurement_keys,
        compiled.noise_locations,
        compiled.location_catalog,
        capacities,
        compiled.stored_operation_count,
        compiled.logical_operation_count,
        compiled.loop_kernel_count,
    ))
}

fn resolve_declared_measurement_key(key: &str, measurement_keys: &mut Vec<String>) -> usize {
    if let Some(measurement_id) = crate::program::automatic_key_ordinal(key) {
        if measurement_keys.get(measurement_id).map(String::as_str) == Some(key) {
            return measurement_id;
        }
    }
    if let Some(measurement_id) = measurement_keys
        .iter()
        .position(|defined_key| defined_key == key)
    {
        return measurement_id;
    }
    let measurement_id = measurement_keys.len();
    measurement_keys.push(key.to_string());
    measurement_id
}

struct MeasureResetReuse {
    measurement_start: usize,
    count: usize,
}

fn find_measure_reset_reuse(operations: &[ExpandedOperation]) -> Vec<MeasureResetReuse> {
    let mut reuse = Vec::new();
    let mut index = 0;
    while index < operations.len() {
        let measurement_start = index;
        while matches!(
            operations.get(index),
            Some(ExpandedOperation::MeasureSingle { .. })
        ) {
            index += 1;
        }
        if index == measurement_start {
            index += 1;
            continue;
        }

        let reset_start = index;
        while matches!(
            operations.get(index),
            Some(ExpandedOperation::Reset {
                measurement_id: None,
                ..
            })
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
                (
                    &operations[measurement_start + offset],
                    &operations[reset_start + offset]
                ),
                (
                    ExpandedOperation::MeasureSingle {
                        qubit: measurement_qubit,
                        basis: measurement_basis,
                        ..
                    },
                    ExpandedOperation::Reset {
                        qubit: reset_qubit,
                        basis: reset_basis,
                        measurement_id: None,
                    }
                ) if measurement_qubit == reset_qubit && measurement_basis == reset_basis
            )
        });
        if matching {
            reuse.push(MeasureResetReuse {
                measurement_start,
                count: measurement_count,
            });
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

    fn compile_sampler(n_qubits: usize, operations: Vec<Operation>) -> NpResult<SamplerProgram> {
        compile_sampler_program_ref(n_qubits, &operations, Vec::new())
    }

    #[test]
    fn compiles_random_measurement_to_sampler_operation() {
        let compiled = compile_sampler(
            1,
            vec![Operation::Measure {
                qubit: 0,
                key: Some("m".to_string()),
                basis: "X".to_string(),
                noise: None,
            }],
        )
        .unwrap();

        assert_eq!(compiled.capacities().random_sources, 1);
        assert!(matches!(
            compiled.operations(),
            [SamplerOperation::MeasureSingle { .. }]
        ));
    }

    #[test]
    fn public_compile_errors_remain_stable_on_the_canonical_path() {
        let assert_error = |operations, expected: &str| {
            let error = compile_sampler(1, operations).unwrap_err();
            assert_eq!(error.message(), expected);
        };

        assert_error(
            vec![Operation::Repeat {
                count: 0,
                body: Vec::new(),
            }],
            "repeat count must be positive",
        );
        assert_error(
            vec![
                Operation::Measure {
                    qubit: 0,
                    key: None,
                    basis: "Z".to_string(),
                    noise: None,
                },
                Operation::Measure {
                    qubit: 0,
                    key: Some("m0".to_string()),
                    basis: "Z".to_string(),
                    noise: None,
                },
            ],
            "duplicate measurement key \"m0\"",
        );
        assert_error(
            vec![Operation::DetectorRec {
                detector_id: Some(0),
                lookbacks: vec![1],
                coords: Vec::new(),
            }],
            "measurement record lookback 1 is out of range",
        );
        assert_error(
            vec![
                Operation::Detector {
                    detector_id: Some(3),
                    measurement_keys: Vec::new(),
                    coords: Vec::new(),
                },
                Operation::Detector {
                    detector_id: Some(3),
                    measurement_keys: Vec::new(),
                    coords: Vec::new(),
                },
            ],
            "duplicate detector id 3",
        );
        let duplicate_noise = NoiseLocation {
            id: "duplicate".to_string(),
            model: NoiseModel::BernoulliPauli("X".to_string()),
            rate: 0.1,
            qubits: vec![0],
            tags: HashMap::new(),
        };
        assert_error(
            vec![
                Operation::Noise(duplicate_noise.clone()),
                Operation::Noise(duplicate_noise),
            ],
            "native sampler requires unique noise location ids; duplicate \"duplicate\"",
        );
        assert_error(
            vec![Operation::Measure {
                qubit: 0,
                key: None,
                basis: "Q".to_string(),
                noise: None,
            }],
            "unsupported measurement basis \"Q\"",
        );
    }

    #[test]
    fn compiler_rejects_invalid_circuit_and_observable_targets() {
        for operation in [Operation::H(1), Operation::Cx(0, 0), Operation::Swap(0, 0)] {
            let err = compile_sampler(1, vec![operation]).unwrap_err();
            assert!(
                err.message().contains("targets qubit 1")
                    || err.message().contains("duplicate qubit 0"),
                "{err}"
            );
        }

        let observable = LogicalObservable {
            id: 7,
            measurement_keys: Vec::new(),
            pauli_qubits: vec![1],
            pauli: "Z".to_string(),
        };
        let err = compile_sampler_program_ref(1, &[], vec![observable]).unwrap_err();
        assert!(err.message().contains("logical observable 7"));
        assert!(err.message().contains("targets qubit 1"));
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
        let repeated = compile_sampler(1, repeated).unwrap();
        let expanded = compile_sampler(1, expanded).unwrap();
        assert_eq!(
            repeated.capacities().random_sources,
            expanded.capacities().random_sources
        );
        let repeated_batch = crate::run_sampler_program(&repeated, 129, Some(123), false).unwrap();
        let expanded_batch = crate::run_sampler_program(&expanded, 129, Some(123), false).unwrap();
        assert_eq!(repeated_batch.measurements, expanded_batch.measurements);
        assert_eq!(repeated_batch.detectors, expanded_batch.detectors);
    }

    #[test]
    fn affine_measure_reset_loop_matches_expansion_and_counts_kernel() {
        let repeated = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 4,
                body: vec![
                    Operation::MeasureReset {
                        qubit: 0,
                        basis: "Z".to_string(),
                    },
                    Operation::MeasureReset {
                        qubit: 0,
                        basis: "Z".to_string(),
                    },
                ],
            }],
        )
        .unwrap();
        let expanded = compile_sampler(
            1,
            (0..8)
                .map(|index| Operation::Reset {
                    qubit: 0,
                    key: Some(format!("m{index}")),
                    basis: "Z".to_string(),
                })
                .collect(),
        )
        .unwrap();

        assert_eq!(repeated.loop_kernel_count(), 1);
        assert_eq!(repeated.operations(), expanded.operations());
    }

    #[test]
    fn affine_mismatch_commits_first_iteration_without_changing_results() {
        let body = vec![
            Operation::Measure {
                qubit: 0,
                key: None,
                basis: "X".to_string(),
                noise: None,
            },
            Operation::Reset {
                qubit: 0,
                key: None,
                basis: "X".to_string(),
            },
        ];
        let repeated = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 4,
                body: body.clone(),
            }],
        )
        .unwrap();
        let expanded = compile_sampler(
            1,
            (0..4)
                .flat_map(|_| body.clone())
                .collect::<Vec<Operation>>(),
        )
        .unwrap();
        let repeated_batch = crate::run_sampler_program(&repeated, 129, Some(19), false).unwrap();
        let expanded_batch = crate::run_sampler_program(&expanded, 129, Some(19), false).unwrap();

        assert_eq!(repeated.loop_kernel_count(), 0);
        assert_eq!(repeated_batch.measurements, expanded_batch.measurements);
    }

    #[test]
    fn short_clifford_repeats_skip_affine_probe() {
        let repeated_h = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 500,
                body: vec![Operation::H(0)],
            }],
        )
        .unwrap();
        let repeated_cx = compile_sampler(
            2,
            vec![Operation::Repeat {
                count: 500,
                body: vec![Operation::Cx(0, 1)],
            }],
        )
        .unwrap();

        assert_eq!(repeated_h.loop_kernel_count(), 0);
        assert_eq!(repeated_cx.loop_kernel_count(), 0);
        assert_eq!(repeated_h.operations().len(), 500);
        assert_eq!(repeated_cx.operations().len(), 500);
    }

    #[test]
    fn clifford_period_one_kernel_uses_one_real_iteration() {
        SYMBOLIC_OPERATION_COUNT.with(|count| count.set(0));
        let body = vec![Operation::H(0), Operation::H(0)];
        let repeated = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 8,
                body: body.clone(),
            }],
        )
        .unwrap();
        let simulated = SYMBOLIC_OPERATION_COUNT.with(std::cell::Cell::get);
        let expanded = compile_sampler(
            1,
            (0..8)
                .flat_map(|_| body.clone())
                .collect::<Vec<Operation>>(),
        )
        .unwrap();

        assert_eq!(repeated.loop_kernel_count(), 1);
        assert_eq!(simulated, body.len());
        assert_eq!(repeated.operations(), expanded.operations());
    }

    #[test]
    fn clifford_period_two_kernel_uses_two_real_iterations() {
        SYMBOLIC_OPERATION_COUNT.with(|count| count.set(0));
        let body = vec![
            Operation::H(0),
            Operation::Pauli {
                qubits: vec![0],
                pauli: "Z".to_string(),
            },
        ];
        let repeated = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 8,
                body: body.clone(),
            }],
        )
        .unwrap();
        let simulated = SYMBOLIC_OPERATION_COUNT.with(std::cell::Cell::get);
        let expanded = compile_sampler(
            1,
            (0..8)
                .flat_map(|_| body.clone())
                .collect::<Vec<Operation>>(),
        )
        .unwrap();

        assert_eq!(repeated.loop_kernel_count(), 1);
        assert_eq!(simulated, body.len() * 2);
        assert_eq!(repeated.operations(), expanded.operations());
    }

    #[test]
    fn failed_clifford_probe_continues_from_third_iteration() {
        SYMBOLIC_OPERATION_COUNT.with(|count| count.set(0));
        let body = vec![Operation::H(0), Operation::S(0)];
        let repeated = compile_sampler(
            1,
            vec![Operation::Repeat {
                count: 8,
                body: body.clone(),
            }],
        )
        .unwrap();
        let simulated = SYMBOLIC_OPERATION_COUNT.with(std::cell::Cell::get);
        let expanded = compile_sampler(
            1,
            (0..8)
                .flat_map(|_| body.clone())
                .collect::<Vec<Operation>>(),
        )
        .unwrap();

        assert_eq!(repeated.loop_kernel_count(), 0);
        assert_eq!(simulated, body.len() * 8);
        assert_eq!(repeated.operations(), expanded.operations());
    }

    #[test]
    fn rejects_dense_affine_templates() {
        let mut dense = Expr::constant(false);
        for source in 0..17 {
            dense.xor_assign(&Expr::random(source));
        }

        assert!(!affine_template_is_compact(&[dense], &[]));
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

        let err = compile_sampler(
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
        let compiled = compile_sampler(
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

        assert_eq!(compiled.operations().len(), 3);
        assert_eq!(compiled.capacities().random_sources, 1);
        assert!(matches!(
            compiled.operations(),
            [
                SamplerOperation::H(0),
                SamplerOperation::MeasureSingle { ideal, .. },
                SamplerOperation::Reset {
                    ideal: reset_ideal,
                    ..
                }
            ] if ideal.terms() == [0] && reset_ideal.terms().is_empty()
        ));
    }

    #[test]
    fn reuses_measurement_expressions_for_batched_measure_then_reset() {
        let compiled = compile_sampler(
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

        assert_eq!(compiled.operations().len(), 6);
        assert_eq!(compiled.capacities().random_sources, 2);
        assert!(compiled.operations()[4..].iter().all(|operation| {
            matches!(operation, SamplerOperation::Reset { ideal, .. } if ideal.terms().is_empty())
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

        let expanded = compile_sampler(1, expanded).unwrap();
        let combined = compile_sampler(1, combined).unwrap();
        assert_eq!(
            expanded.capacities().random_sources,
            combined.capacities().random_sources
        );
        let expanded_state =
            crate::run_sampler_program(&expanded, 130, Some(12345), false).unwrap();
        let combined_state =
            crate::run_sampler_program(&combined, 130, Some(12345), false).unwrap();
        assert_eq!(expanded_state.measurements, combined_state.measurements);
    }

    #[test]
    fn does_not_fuse_measure_reset_across_a_gate_or_basis_change() {
        let compiled = compile_sampler(
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

        assert_eq!(compiled.operations().len(), 4);
        assert!(matches!(
            compiled.operations().last(),
            Some(SamplerOperation::Reset { ideal, .. }) if !ideal.terms().is_empty()
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
        let compiled = compile_sampler(
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
            compiled.operations().last(),
            Some(SamplerOperation::Reset { ideal, .. }) if !ideal.terms().is_empty()
        ));
    }
}
