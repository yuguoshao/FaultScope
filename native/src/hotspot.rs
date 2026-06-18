use crate::*;

pub(crate) struct DemBatch {
    pub(crate) shots: usize,
    pub(crate) all_mask: Mask,
    pub(crate) detectors: HashMap<i64, Mask>,
    pub(crate) observables: HashMap<i64, Mask>,
    pub(crate) edge_event_masks: Vec<Mask>,
    pub(crate) loss_mask: Mask,
}

pub(crate) struct DemEstimate {
    pub(crate) shots: usize,
    pub(crate) mean_loss: f64,
    pub(crate) baseline: f64,
    pub(crate) edge_sensitivities: Vec<f64>,
    pub(crate) edge_hotspots: Vec<f64>,
    pub(crate) location_sensitivities: HashMap<String, f64>,
    pub(crate) location_hotspots: HashMap<String, f64>,
    pub(crate) by_detector: HashMap<i64, f64>,
    pub(crate) by_round: HashMap<TagValue, f64>,
    pub(crate) by_gate: HashMap<TagValue, f64>,
    pub(crate) by_operation: HashMap<TagValue, f64>,
    pub(crate) detector_graph: DetectorGraphEstimate,
    pub(crate) top_edges: Vec<usize>,
    pub(crate) top_locations: Vec<String>,
}

pub(crate) struct PackedEstimate {
    pub(crate) shots: usize,
    pub(crate) mean_loss: f64,
    pub(crate) baseline: f64,
    pub(crate) sensitivities: HashMap<String, f64>,
    pub(crate) hotspots: HashMap<String, f64>,
    pub(crate) by_qubit: HashMap<usize, f64>,
    pub(crate) by_round: HashMap<TagValue, f64>,
    pub(crate) by_gate: HashMap<TagValue, f64>,
    pub(crate) by_operation: HashMap<TagValue, f64>,
    pub(crate) top_locations: Vec<String>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct DetectorGraphKey {
    pub(crate) detectors: Vec<i64>,
    pub(crate) observables: Vec<i64>,
}

pub(crate) struct DetectorGraphEstimate {
    pub(crate) by_detector_edge: HashMap<DetectorGraphKey, f64>,
    pub(crate) signed_by_detector_edge: HashMap<DetectorGraphKey, f64>,
    pub(crate) by_detector: HashMap<i64, f64>,
    pub(crate) signed_by_detector: HashMap<i64, f64>,
    pub(crate) by_observable: HashMap<i64, f64>,
    pub(crate) signed_by_observable: HashMap<i64, f64>,
    pub(crate) by_location: HashMap<String, f64>,
    pub(crate) signed_by_location: HashMap<String, f64>,
}

pub(crate) fn run_dem_batch(
    sampler: &NativeDemSampler,
    shots: usize,
    rng: &mut SmallRng,
    return_edge_events: bool,
) -> DemBatch {
    let words = word_count(shots);
    let all_mask = Mask::all(shots);
    let mut detectors = HashMap::new();
    for detector_id in sampler.detectors.iter() {
        detectors.insert(*detector_id, Mask::zero(words));
    }
    let mut observables = HashMap::new();
    for observable_id in sampler.observables.iter() {
        observables.insert(*observable_id, Mask::zero(words));
    }
    let mut edge_event_masks = if return_edge_events {
        Vec::with_capacity(sampler.edges.len())
    } else {
        Vec::new()
    };

    for edge in sampler.edges.iter() {
        let event_mask = bernoulli_mask(rng, shots, edge.probability);
        if !event_mask.is_zero() {
            for detector_id in &edge.detectors {
                detectors
                    .entry(*detector_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
            for observable_id in &edge.observables {
                observables
                    .entry(*observable_id)
                    .or_insert_with(|| Mask::zero(words))
                    .xor_assign(&event_mask);
            }
        }
        if return_edge_events {
            edge_event_masks.push(event_mask);
        }
    }

    let mut loss_mask = Mask::zero(words);
    for observable in observables.values() {
        loss_mask.or_assign(observable);
    }
    loss_mask.and_assign(&all_mask);

    DemBatch {
        shots,
        all_mask,
        detectors,
        observables,
        edge_event_masks,
        loss_mask,
    }
}

pub(crate) fn dem_batch_to_py(py: Python<'_>, batch: &DemBatch) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", batch.shots)?;
    out.set_item("all_mask", mask_to_py(py, &batch.all_mask)?)?;
    out.set_item("detectors", int_map_to_py(py, &batch.detectors)?)?;
    out.set_item("observables", int_map_to_py(py, &batch.observables)?)?;
    let edge_masks = PyDict::new(py);
    for (edge_index, mask) in batch.edge_event_masks.iter().enumerate() {
        edge_masks.set_item(edge_index, mask_to_py(py, mask)?)?;
    }
    out.set_item("edge_event_masks", edge_masks)?;
    Ok(out.into())
}

pub(crate) fn compute_packed_estimate(
    sampler: &NativePackedSampler,
    state: &RuntimeState,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> PackedEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&state.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / state.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut sensitivities = HashMap::new();
    let mut hotspots = HashMap::new();
    let mut by_qubit = HashMap::new();

    for location in &sampler.noise_locations {
        let zero_mask;
        let event_mask = if let Some(mask) = state.event_masks.get(&location.id) {
            mask
        } else {
            zero_mask = Mask::zero(state.all_mask.words.len());
            &zero_mask
        };
        let event_count = event_mask.bit_count();
        let no_event_count = state.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(location.rate);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        let sensitivity = (sum_loss_score - baseline_value * sum_score) / state.shots as f64;
        let hotspot = sensitivity.abs();
        sensitivities.insert(location.id.clone(), sensitivity);
        hotspots.insert(location.id.clone(), hotspot);
        for qubit in &location.qubits {
            add_f64(&mut by_qubit, *qubit, hotspot);
        }
    }

    let by_round = aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "round");
    let by_gate = aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "gate");
    let by_operation =
        aggregate_location_tag_hotspots(&sampler.noise_locations, &hotspots, "operation");
    let top_locations = top_location_ids(&hotspots, top_k);

    PackedEstimate {
        shots: state.shots,
        mean_loss,
        baseline: baseline_value,
        sensitivities,
        hotspots,
        by_qubit,
        by_round,
        by_gate,
        by_operation,
        top_locations,
    }
}

pub(crate) fn compute_dem_estimate(
    sampler: &NativeDemSampler,
    batch: &DemBatch,
    loss_mask: &Mask,
    baseline: Option<f64>,
    top_k: usize,
) -> DemEstimate {
    let mut clipped_loss = loss_mask.clone();
    clipped_loss.and_assign(&batch.all_mask);
    let loss_count = clipped_loss.bit_count();
    let mean_loss = loss_count as f64 / batch.shots as f64;
    let baseline_value = baseline.unwrap_or(mean_loss);
    let mut edge_sensitivities = Vec::with_capacity(sampler.edges.len());
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let event_mask = &batch.edge_event_masks[edge_index];
        let event_count = event_mask.bit_count();
        let no_event_count = batch.shots - event_count;
        let loss_event_count = clipped_loss.and_count(event_mask);
        let loss_no_event_count = loss_count - loss_event_count;
        let (event_score, no_event_score) = score_pair(edge.probability);
        let sum_loss_score =
            loss_event_count as f64 * event_score + loss_no_event_count as f64 * no_event_score;
        let sum_score = event_count as f64 * event_score + no_event_count as f64 * no_event_score;
        edge_sensitivities.push((sum_loss_score - baseline_value * sum_score) / batch.shots as f64);
    }

    let location_sensitivities = aggregate_dem_location_sensitivities(sampler, &edge_sensitivities);
    let edge_hotspots = edge_sensitivities
        .iter()
        .map(|sensitivity| sensitivity.abs())
        .collect::<Vec<_>>();
    let location_hotspots = location_sensitivities
        .iter()
        .map(|(location_id, sensitivity)| (location_id.clone(), sensitivity.abs()))
        .collect::<HashMap<_, _>>();
    let by_detector = aggregate_dem_detector_hotspots(sampler, &edge_hotspots);
    let by_round = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "round");
    let by_gate = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "gate");
    let by_operation = aggregate_dem_tag_hotspots(sampler, &location_hotspots, "operation");
    let detector_graph = compute_detector_graph_estimate(sampler, &edge_sensitivities);
    let top_edges = top_edge_indices(&edge_hotspots, top_k);
    let top_locations = top_location_ids(&location_hotspots, top_k);

    DemEstimate {
        shots: batch.shots,
        mean_loss,
        baseline: baseline_value,
        edge_sensitivities,
        edge_hotspots,
        location_sensitivities,
        location_hotspots,
        by_detector,
        by_round,
        by_gate,
        by_operation,
        detector_graph,
        top_edges,
        top_locations,
    }
}

pub(crate) fn packed_estimate_to_py(
    py: Python<'_>,
    estimate: &PackedEstimate,
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", estimate.shots)?;
    out.set_item("mean_loss", estimate.mean_loss)?;
    out.set_item("baseline", estimate.baseline)?;
    out.set_item(
        "sensitivities",
        string_f64_map_to_py(py, &estimate.sensitivities)?,
    )?;
    out.set_item("hotspots", string_f64_map_to_py(py, &estimate.hotspots)?)?;
    out.set_item("by_qubit", usize_f64_map_to_py(py, &estimate.by_qubit)?)?;
    out.set_item("by_round", tag_f64_map_to_py(py, &estimate.by_round)?)?;
    out.set_item("by_gate", tag_f64_map_to_py(py, &estimate.by_gate)?)?;
    out.set_item(
        "by_operation",
        tag_f64_map_to_py(py, &estimate.by_operation)?,
    )?;
    out.set_item(
        "top_hotspots",
        packed_top_locations_to_py(
            py,
            &estimate.top_locations,
            &estimate.sensitivities,
            &estimate.hotspots,
        )?,
    )?;
    Ok(out.into())
}

pub(crate) fn dem_estimate_to_py(py: Python<'_>, estimate: &DemEstimate) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item("shots", estimate.shots)?;
    out.set_item("mean_loss", estimate.mean_loss)?;
    out.set_item("baseline", estimate.baseline)?;
    let edge_dict = PyDict::new(py);
    for (edge_index, sensitivity) in estimate.edge_sensitivities.iter().enumerate() {
        edge_dict.set_item(edge_index, *sensitivity)?;
    }
    out.set_item("edge_sensitivities", edge_dict)?;
    let edge_hotspots = PyDict::new(py);
    for (edge_index, hotspot) in estimate.edge_hotspots.iter().enumerate() {
        edge_hotspots.set_item(edge_index, *hotspot)?;
    }
    out.set_item("edge_hotspots", edge_hotspots)?;
    out.set_item(
        "sensitivities",
        string_f64_map_to_py(py, &estimate.location_sensitivities)?,
    )?;
    out.set_item(
        "hotspots",
        string_f64_map_to_py(py, &estimate.location_hotspots)?,
    )?;
    out.set_item("by_detector", i64_f64_map_to_py(py, &estimate.by_detector)?)?;
    out.set_item("by_round", tag_f64_map_to_py(py, &estimate.by_round)?)?;
    out.set_item("by_gate", tag_f64_map_to_py(py, &estimate.by_gate)?)?;
    out.set_item(
        "by_operation",
        tag_f64_map_to_py(py, &estimate.by_operation)?,
    )?;
    out.set_item(
        "detector_graph_hotspots",
        detector_graph_to_py(
            py,
            &estimate.detector_graph,
            &estimate.edge_sensitivities,
            &estimate.edge_hotspots,
        )?,
    )?;
    out.set_item(
        "top_edges",
        dem_top_edges_to_py(
            py,
            &estimate.top_edges,
            &estimate.edge_sensitivities,
            &estimate.edge_hotspots,
        )?,
    )?;
    out.set_item(
        "top_hotspots",
        packed_top_locations_to_py(
            py,
            &estimate.top_locations,
            &estimate.location_sensitivities,
            &estimate.location_hotspots,
        )?,
    )?;
    Ok(out.into())
}

pub(crate) fn string_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<String, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn usize_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<usize, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn i64_f64_map_to_py(py: Python<'_>, values: &HashMap<i64, f64>) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(*key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn tag_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<TagValue, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        dict.set_item(tag_value_to_py(py, key)?, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn tag_value_to_py(py: Python<'_>, value: &TagValue) -> PyResult<PyObject> {
    match value {
        TagValue::None => Ok(py.None()),
        TagValue::Bool(value) => Ok(value.into_py(py)),
        TagValue::Int(value) => Ok(value.into_py(py)),
        TagValue::Float(bits) => Ok(f64::from_bits(*bits).into_py(py)),
        TagValue::String(value) => Ok(value.into_py(py)),
    }
}

pub(crate) fn packed_top_locations_to_py(
    py: Python<'_>,
    location_ids: &[String],
    sensitivities: &HashMap<String, f64>,
    hotspots: &HashMap<String, f64>,
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for location_id in location_ids {
        let row = PyDict::new(py);
        row.set_item("location_id", location_id)?;
        row.set_item(
            "sensitivity",
            *sensitivities.get(location_id).unwrap_or(&0.0),
        )?;
        row.set_item("hotspot", *hotspots.get(location_id).unwrap_or(&0.0))?;
        list.append(row)?;
    }
    Ok(list.into())
}

pub(crate) fn dem_top_edges_to_py(
    py: Python<'_>,
    edge_indices: &[usize],
    sensitivities: &[f64],
    hotspots: &[f64],
) -> PyResult<PyObject> {
    let list = PyList::empty(py);
    for edge_index in edge_indices {
        let row = PyDict::new(py);
        row.set_item("edge_index", *edge_index)?;
        row.set_item(
            "sensitivity",
            sensitivities.get(*edge_index).copied().unwrap_or(0.0),
        )?;
        row.set_item("hotspot", hotspots.get(*edge_index).copied().unwrap_or(0.0))?;
        list.append(row)?;
    }
    Ok(list.into())
}

pub(crate) fn detector_graph_to_py(
    py: Python<'_>,
    graph: &DetectorGraphEstimate,
    sensitivities: &[f64],
    hotspots: &[f64],
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    let edge_rows = PyList::empty(py);
    for (edge_index, sensitivity) in sensitivities.iter().enumerate() {
        let row = PyDict::new(py);
        row.set_item("edge_index", edge_index)?;
        row.set_item("sensitivity", *sensitivity)?;
        row.set_item("hotspot", hotspots.get(edge_index).copied().unwrap_or(0.0))?;
        edge_rows.append(row)?;
    }
    out.set_item("edge_hotspots", edge_rows)?;
    out.set_item(
        "by_detector_edge",
        graph_key_f64_map_to_py(py, &graph.by_detector_edge)?,
    )?;
    out.set_item(
        "signed_by_detector_edge",
        graph_key_f64_map_to_py(py, &graph.signed_by_detector_edge)?,
    )?;
    out.set_item("by_detector", i64_f64_map_to_py(py, &graph.by_detector)?)?;
    out.set_item(
        "signed_by_detector",
        i64_f64_map_to_py(py, &graph.signed_by_detector)?,
    )?;
    out.set_item(
        "by_observable",
        i64_f64_map_to_py(py, &graph.by_observable)?,
    )?;
    out.set_item(
        "signed_by_observable",
        i64_f64_map_to_py(py, &graph.signed_by_observable)?,
    )?;
    out.set_item("by_location", string_f64_map_to_py(py, &graph.by_location)?)?;
    out.set_item(
        "signed_by_location",
        string_f64_map_to_py(py, &graph.signed_by_location)?,
    )?;
    Ok(out.into())
}

pub(crate) fn graph_key_f64_map_to_py(
    py: Python<'_>,
    values: &HashMap<DetectorGraphKey, f64>,
) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    for (key, value) in values {
        let detectors = PyTuple::new(py, key.detectors.iter().copied())?;
        let observables = PyTuple::new(py, key.observables.iter().copied())?;
        let detectors_obj: PyObject = detectors.into_py(py);
        let observables_obj: PyObject = observables.into_py(py);
        let graph_key = PyTuple::new(py, [detectors_obj, observables_obj])?;
        dict.set_item(graph_key, *value)?;
    }
    Ok(dict.into())
}

pub(crate) fn aggregate_dem_location_sensitivities(
    sampler: &NativeDemSampler,
    edge_sensitivities: &[f64],
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for group in sampler.location_groups.iter() {
        let mut value = 0.0;
        for edge_index in &group.edge_indices {
            let weight = if group.total_probability > 0.0 {
                sampler.edges[*edge_index].probability / group.total_probability
            } else {
                1.0 / group.edge_indices.len() as f64
            };
            value += edge_sensitivities[*edge_index] * weight;
        }
        out.insert(group.location_id.clone(), value);
    }
    out
}

pub(crate) fn build_dem_location_groups(edges: &[DemEdgeSpec]) -> Vec<DemLocationGroup> {
    let mut group_indices = HashMap::<String, usize>::new();
    let mut groups = Vec::<DemLocationGroup>::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let group_index = if let Some(group_index) = group_indices.get(&edge.location_id) {
            *group_index
        } else {
            let group_index = groups.len();
            group_indices.insert(edge.location_id.clone(), group_index);
            groups.push(DemLocationGroup {
                location_id: edge.location_id.clone(),
                edge_indices: Vec::new(),
                total_probability: 0.0,
            });
            group_index
        };
        let group = &mut groups[group_index];
        group.edge_indices.push(edge_index);
        group.total_probability += edge.probability;
    }
    groups
}

pub(crate) fn aggregate_location_tag_hotspots(
    locations: &[NoiseLocationSpec],
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut out = HashMap::new();
    for location in locations {
        let Some(tag_value) = location.tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location.id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

pub(crate) fn aggregate_dem_tag_hotspots(
    sampler: &NativeDemSampler,
    hotspots: &HashMap<String, f64>,
    tag: &str,
) -> HashMap<TagValue, f64> {
    let mut by_location: HashMap<String, &HashMap<String, TagValue>> = HashMap::new();
    for edge in sampler.edges.iter() {
        by_location
            .entry(edge.location_id.clone())
            .or_insert(&edge.tags);
    }
    let mut out = HashMap::new();
    for (location_id, tags) in by_location {
        let Some(tag_value) = tags.get(tag) else {
            continue;
        };
        let hotspot = *hotspots.get(&location_id).unwrap_or(&0.0);
        add_f64(&mut out, tag_value.clone(), hotspot);
    }
    out
}

pub(crate) fn aggregate_dem_detector_hotspots(
    sampler: &NativeDemSampler,
    edge_hotspots: &[f64],
) -> HashMap<i64, f64> {
    let mut out = HashMap::new();
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let hotspot = edge_hotspots.get(edge_index).copied().unwrap_or(0.0);
        if hotspot == 0.0 || edge.detectors.is_empty() {
            continue;
        }
        let share = hotspot / edge.detectors.len() as f64;
        for detector_id in &edge.detectors {
            add_f64(&mut out, *detector_id, share);
        }
    }
    out
}

pub(crate) fn compute_detector_graph_estimate(
    sampler: &NativeDemSampler,
    edge_sensitivities: &[f64],
) -> DetectorGraphEstimate {
    let mut graph = DetectorGraphEstimate {
        by_detector_edge: HashMap::new(),
        signed_by_detector_edge: HashMap::new(),
        by_detector: HashMap::new(),
        signed_by_detector: HashMap::new(),
        by_observable: HashMap::new(),
        signed_by_observable: HashMap::new(),
        by_location: HashMap::new(),
        signed_by_location: HashMap::new(),
    };
    for (edge_index, edge) in sampler.edges.iter().enumerate() {
        let sensitivity = edge_sensitivities.get(edge_index).copied().unwrap_or(0.0);
        let hotspot = sensitivity.abs();
        if hotspot == 0.0 {
            continue;
        }
        let key = DetectorGraphKey {
            detectors: edge.detectors.clone(),
            observables: edge.observables.clone(),
        };
        add_f64(&mut graph.by_detector_edge, key.clone(), hotspot);
        add_f64(&mut graph.signed_by_detector_edge, key, sensitivity);
        add_f64(&mut graph.by_location, edge.location_id.clone(), hotspot);
        add_f64(
            &mut graph.signed_by_location,
            edge.location_id.clone(),
            sensitivity,
        );
        if !edge.detectors.is_empty() {
            let share = hotspot / edge.detectors.len() as f64;
            let signed_share = sensitivity / edge.detectors.len() as f64;
            for detector_id in &edge.detectors {
                add_f64(&mut graph.by_detector, *detector_id, share);
                add_f64(&mut graph.signed_by_detector, *detector_id, signed_share);
            }
        }
        if !edge.observables.is_empty() {
            let share = hotspot / edge.observables.len() as f64;
            let signed_share = sensitivity / edge.observables.len() as f64;
            for observable_id in &edge.observables {
                add_f64(&mut graph.by_observable, *observable_id, share);
                add_f64(
                    &mut graph.signed_by_observable,
                    *observable_id,
                    signed_share,
                );
            }
        }
    }
    graph
}

pub(crate) fn top_location_ids(hotspots: &HashMap<String, f64>, top_k: usize) -> Vec<String> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots
        .iter()
        .map(|(location_id, hotspot)| (location_id, *hotspot))
        .collect::<Vec<_>>();
    let compare = |left: &(&String, f64), right: &(&String, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter()
        .map(|(location_id, _)| location_id.clone())
        .collect()
}

pub(crate) fn top_edge_indices(hotspots: &[f64], top_k: usize) -> Vec<usize> {
    if top_k == 0 || hotspots.is_empty() {
        return Vec::new();
    }
    let limit = top_k.min(hotspots.len());
    let mut rows = hotspots.iter().copied().enumerate().collect::<Vec<_>>();
    let compare = |left: &(usize, f64), right: &(usize, f64)| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    if limit < rows.len() {
        rows.select_nth_unstable_by(limit, compare);
    }
    rows.truncate(limit);
    rows.sort_by(compare);
    rows.into_iter().map(|(edge_index, _)| edge_index).collect()
}

pub(crate) fn add_f64<K>(values: &mut HashMap<K, f64>, key: K, value: f64)
where
    K: std::hash::Hash + Eq,
{
    values
        .entry(key)
        .and_modify(|existing| *existing += value)
        .or_insert(value);
}

pub(crate) fn score_pair(probability: f64) -> (f64, f64) {
    let p = probability.clamp(1e-12, 1.0 - 1e-12);
    (1.0 / p, -1.0 / (1.0 - p))
}
