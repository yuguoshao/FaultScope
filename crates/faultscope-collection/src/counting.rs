use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;

use faultscope_core::{
    convert_detector_batch, packed_residual_failure_count, run_sampler_program,
    select_detector_batch_format, validate_decoder_batch_formats, validate_decoder_detector_ids,
    word_count, CompiledDemLogicalCountPlan, CompiledDemSamplingPlan, CorrectionMaskBatch,
    DecoderCorrectionBatch, DemHotspotEstimator, DemSamplingResult, DetectorBatchFormat,
    DetectorMaskBatchView, Mask, NativeDecoderWorker, NpError, NpResult, RuntimeState, SmallRng,
};

use crate::api::{CollectionSampler, ForwardSamplerView};

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct BatchStats {
    pub(crate) shots: usize,
    pub(crate) errors: usize,
    pub(crate) discards: usize,
    pub(crate) seconds: f64,
    pub(crate) custom_counts: HashMap<String, usize>,
}

pub(crate) struct DetailedBatchResult {
    pub(crate) stats: BatchStats,
    pub(crate) loss_mask: Mask,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CountOptions<'a> {
    pub(crate) postselection_mask: Option<&'a [u8]>,
    pub(crate) postselected_observables_mask: Option<&'a [u8]>,
    pub(crate) count_observable_error_combos: bool,
    pub(crate) count_detection_events: bool,
}

impl CountOptions<'_> {
    pub(crate) fn uses_detailed_path(&self) -> bool {
        packed_mask_has_any(self.postselection_mask)
            || packed_mask_has_any(self.postselected_observables_mask)
            || self.count_observable_error_combos
            || self.count_detection_events
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PreparedDemCountPlan {
    Logical(CompiledDemLogicalCountPlan),
    Decoder(CompiledDemSamplingPlan),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PreparedForwardCountPlan {
    decoder_detector_ids: Vec<i64>,
    decoder_format: Option<DetectorBatchFormat>,
    record_events: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PreparedCountPlan {
    Dem(PreparedDemCountPlan),
    Forward(PreparedForwardCountPlan),
}

pub(crate) fn prepare_count_plan(
    sampler: &CollectionSampler,
    decoder_metadata: Option<(&[i64], &[DetectorBatchFormat])>,
    count_options: &CountOptions<'_>,
    record_attribution: bool,
) -> NpResult<PreparedCountPlan> {
    match sampler {
        CollectionSampler::Dem(sampler) => Ok(PreparedCountPlan::Dem(prepare_dem_count_plan(
            sampler,
            decoder_metadata,
            count_options,
            record_attribution,
        )?)),
        CollectionSampler::Forward(_) => Ok(PreparedCountPlan::Forward(
            prepare_forward_count_plan(decoder_metadata, record_attribution)?,
        )),
    }
}

pub(crate) fn prepare_forward_count_plan(
    decoder_metadata: Option<(&[i64], &[DetectorBatchFormat])>,
    record_attribution: bool,
) -> NpResult<PreparedForwardCountPlan> {
    let (decoder_detector_ids, decoder_format) = match decoder_metadata {
        Some((detector_ids, formats)) => {
            validate_decoder_detector_ids(detector_ids)?;
            validate_decoder_batch_formats(formats)?;
            let format = select_detector_batch_format(formats, &[DetectorBatchFormat::Masks])?;
            (detector_ids.to_vec(), Some(format))
        }
        None => (Vec::new(), None),
    };
    Ok(PreparedForwardCountPlan {
        decoder_detector_ids,
        decoder_format,
        record_events: record_attribution,
    })
}

pub(crate) fn sample_logical_error_stats(
    sampler: &CollectionSampler,
    shots: usize,
    seed: u64,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    started: Option<Instant>,
    count_options: &CountOptions<'_>,
    prepared_plan: &PreparedCountPlan,
) -> NpResult<BatchStats> {
    match (sampler, prepared_plan) {
        (CollectionSampler::Dem(sampler), PreparedCountPlan::Dem(plan)) => {
            let mut rng = SmallRng::new(seed);
            sample_dem_logical_error_stats_with_rng(
                sampler,
                shots,
                &mut rng,
                decoder,
                started,
                count_options,
                plan,
            )
        }
        (CollectionSampler::Forward(sampler), PreparedCountPlan::Forward(plan)) => {
            sample_forward_logical_error_stats(
                sampler.view(),
                shots,
                seed,
                decoder,
                started,
                count_options,
                plan,
            )
        }
        _ => Err(NpError::new(
            "collection sampler and prepared count plan do not match",
        )),
    }
}

pub(crate) fn prepare_dem_count_plan(
    sampler: &DemHotspotEstimator,
    decoder_metadata: Option<(&[i64], &[DetectorBatchFormat])>,
    count_options: &CountOptions<'_>,
    record_attribution: bool,
) -> NpResult<PreparedDemCountPlan> {
    if let Some((detector_ids, batch_formats)) = decoder_metadata {
        validate_decoder_detector_ids(detector_ids)?;
        validate_decoder_batch_formats(batch_formats)?;
    }
    match decoder_metadata {
        Some((detector_ids, batch_formats)) => Ok(PreparedDemCountPlan::Decoder(
            sampler.compile_decoder_sampling_plan(
                detector_ids,
                sampler.observable_ids(),
                batch_formats[0],
                &detail_detector_ids(sampler, count_options),
                record_attribution,
            )?,
        )),
        None if count_options.uses_detailed_path() || record_attribution => Ok(
            PreparedDemCountPlan::Decoder(sampler.compile_decoder_sampling_plan(
                &[],
                sampler.observable_ids(),
                DetectorBatchFormat::Masks,
                &detail_detector_ids(sampler, count_options),
                record_attribution,
            )?),
        ),
        None => Ok(PreparedDemCountPlan::Logical(
            sampler.compile_logical_count_plan(),
        )),
    }
}

fn detail_detector_ids(
    sampler: &DemHotspotEstimator,
    count_options: &CountOptions<'_>,
) -> Vec<i64> {
    if count_options.count_detection_events {
        return sampler.detector_ids().to_vec();
    }
    let Some(postselection_mask) = count_options.postselection_mask else {
        return Vec::new();
    };
    sampler
        .detector_ids()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, id)| packed_mask_bit(Some(postselection_mask), index).then_some(id))
        .collect()
}

pub(crate) fn sample_dem_logical_error_stats_with_rng(
    sampler: &DemHotspotEstimator,
    shots: usize,
    rng: &mut SmallRng,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    started: Option<Instant>,
    count_options: &CountOptions<'_>,
    prepared_plan: &PreparedDemCountPlan,
) -> NpResult<BatchStats> {
    if let Some(decoder) = decoder.as_deref() {
        validate_decoder_observable_layout(
            sampler.observable_ids(),
            decoder.observable_ids(),
            decoder.name(),
        )?;
    }
    let mut out = match (decoder, prepared_plan) {
        (decoder, PreparedDemCountPlan::Decoder(plan)) => {
            let sampled = plan.run_sampling_result_with_rng(shots, rng)?;
            if count_options.uses_detailed_path() {
                count_detailed_sampling_result(sampler, &sampled, decoder, count_options)?.stats
            } else {
                let decoder = decoder.ok_or_else(|| {
                    NpError::new("decoder sampling plan requires a native decoder worker")
                })?;
                BatchStats {
                    shots,
                    errors: count_decoder_sampling_result(&sampled, decoder)?,
                    discards: 0,
                    seconds: 0.0,
                    custom_counts: HashMap::new(),
                }
            }
        }
        (None, PreparedDemCountPlan::Logical(plan)) => BatchStats {
            shots,
            errors: plan.sample_logical_error_count_with_rng(shots, rng),
            discards: 0,
            seconds: 0.0,
            custom_counts: HashMap::new(),
        },
        (Some(_), PreparedDemCountPlan::Logical(_)) => {
            return Err(NpError::new(
                "native decoder worker cannot run with a logical-only sampling plan",
            ));
        }
    };
    out.seconds = started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    Ok(out)
}

pub(crate) struct ForwardDetailedBatchResult {
    pub(crate) state: RuntimeState,
    pub(crate) detailed: DetailedBatchResult,
}

pub(crate) fn sample_forward_detailed_batch(
    sampler: ForwardSamplerView<'_>,
    shots: usize,
    seed: u64,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    count_options: &CountOptions<'_>,
    prepared_plan: &PreparedForwardCountPlan,
) -> NpResult<ForwardDetailedBatchResult> {
    let state = run_sampler_program(
        sampler.program(),
        shots,
        Some(seed),
        prepared_plan.record_events,
    )?;
    let corrections =
        forward_corrections(&state, sampler.observable_ids(), decoder, prepared_plan)?;
    let residuals = forward_residual_masks(&state, sampler.observable_ids(), &corrections)?;
    let detailed = count_forward_masks(
        &state,
        sampler.detector_ids(),
        sampler.observable_ids(),
        residuals,
        count_options,
    )?;
    Ok(ForwardDetailedBatchResult { state, detailed })
}

pub(crate) fn sample_forward_logical_error_stats(
    sampler: ForwardSamplerView<'_>,
    shots: usize,
    seed: u64,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    started: Option<Instant>,
    count_options: &CountOptions<'_>,
    prepared_plan: &PreparedForwardCountPlan,
) -> NpResult<BatchStats> {
    let state = run_sampler_program(
        sampler.program(),
        shots,
        Some(seed),
        prepared_plan.record_events,
    )?;
    let corrections =
        forward_corrections(&state, sampler.observable_ids(), decoder, prepared_plan)?;
    let residuals = forward_residual_masks(&state, sampler.observable_ids(), &corrections)?;
    let mut out = if count_options.uses_detailed_path() {
        count_forward_masks(
            &state,
            sampler.detector_ids(),
            sampler.observable_ids(),
            residuals,
            count_options,
        )?
        .stats
    } else {
        let mut loss = Mask::zero(word_count(shots));
        for residual in residuals {
            loss.or_assign(&residual);
        }
        loss.and_assign(state.all_mask());
        BatchStats {
            shots,
            errors: loss.bit_count(),
            discards: 0,
            seconds: 0.0,
            custom_counts: HashMap::new(),
        }
    };
    out.seconds = started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    Ok(out)
}

fn forward_corrections(
    state: &RuntimeState,
    observable_ids: &[i64],
    decoder: Option<&mut dyn NativeDecoderWorker>,
    prepared_plan: &PreparedForwardCountPlan,
) -> NpResult<CorrectionMaskBatch> {
    let Some(decoder) = decoder else {
        return Ok(CorrectionMaskBatch::empty(state.shots()));
    };
    validate_decoder_observable_layout(observable_ids, decoder.observable_ids(), decoder.name())?;
    let masks = prepared_plan
        .decoder_detector_ids
        .iter()
        .map(|detector_id| {
            state
                .detectors
                .get(detector_id)
                .cloned()
                .ok_or_else(|| NpError::new(format!("missing detector id {detector_id}")))
        })
        .collect::<NpResult<Vec<_>>>()?;
    let view =
        DetectorMaskBatchView::new(&prepared_plan.decoder_detector_ids, &masks, state.shots())?;
    let format = prepared_plan.decoder_format.ok_or_else(|| {
        NpError::new("forward decoder count plan is missing a negotiated detector format")
    })?;
    let negotiated = convert_detector_batch(view.into(), format)?;
    let corrections = decoder
        .decode_batch_checked(negotiated.view())?
        .into_masks()?;
    corrections.validate_against(observable_ids, state.shots())?;
    Ok(corrections)
}

fn forward_residual_masks(
    state: &RuntimeState,
    observable_ids: &[i64],
    corrections: &CorrectionMaskBatch,
) -> NpResult<Vec<Mask>> {
    let zero = Mask::zero(word_count(state.shots()));
    observable_ids
        .iter()
        .map(|observable_id| {
            let actual = state.observables.get(observable_id).ok_or_else(|| {
                NpError::new(format!("missing logical observable id {observable_id}"))
            })?;
            let correction = corrections.get(*observable_id).unwrap_or(&zero);
            let mut residual = actual.clone();
            residual.xor_assign(correction);
            residual.clear_unused(state.shots());
            Ok(residual)
        })
        .collect()
}

fn count_forward_masks(
    state: &RuntimeState,
    detector_ids: &[i64],
    observable_ids: &[i64],
    residuals: Vec<Mask>,
    count_options: &CountOptions<'_>,
) -> NpResult<DetailedBatchResult> {
    let shots = state.shots();
    validate_mask_shape(
        count_options.postselection_mask,
        detector_ids.len(),
        "postselection_mask",
    )?;
    validate_mask_shape(
        count_options.postselected_observables_mask,
        observable_ids.len(),
        "postselected_observables_mask",
    )?;
    let mut detector_discard_mask = count_options
        .postselection_mask
        .map(|_| Mask::zero(word_count(shots)));
    if let Some(discard) = detector_discard_mask.as_mut() {
        for (index, detector_id) in detector_ids.iter().enumerate() {
            if packed_mask_bit(count_options.postselection_mask, index) {
                let detector = state
                    .detectors
                    .get(detector_id)
                    .ok_or_else(|| NpError::new(format!("missing detector id {detector_id}")))?;
                discard.or_assign(detector);
            }
        }
        discard.clear_unused(shots);
    }

    let mut custom_counts = HashMap::new();
    if count_options.count_detection_events {
        let mut events = 0usize;
        for detector_id in detector_ids {
            events += state
                .detectors
                .get(detector_id)
                .ok_or_else(|| NpError::new(format!("missing detector id {detector_id}")))?
                .bit_count();
        }
        custom_counts.insert("detection_events".to_string(), events);
        custom_counts.insert("detectors_checked".to_string(), shots * detector_ids.len());
    }

    let residuals = DetailedResidualBatch::Masks(Cow::Owned(residuals));
    let (errors, discards, loss_mask) = count_residual_batch(
        &residuals,
        observable_ids.len(),
        shots,
        detector_discard_mask.as_ref(),
        count_options.postselected_observables_mask,
        count_options.count_observable_error_combos,
        &mut custom_counts,
    );
    Ok(DetailedBatchResult {
        stats: BatchStats {
            shots,
            errors,
            discards,
            seconds: 0.0,
            custom_counts,
        },
        loss_mask,
    })
}

pub(crate) fn count_detailed_sampling_result(
    sampler: &DemHotspotEstimator,
    sampled: &DemSamplingResult,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    count_options: &CountOptions<'_>,
) -> NpResult<DetailedBatchResult> {
    let shots = sampled.shots();
    if let Some(decoder) = decoder.as_deref() {
        validate_decoder_observable_layout(
            sampler.observable_ids(),
            decoder.observable_ids(),
            decoder.name(),
        )?;
    }
    validate_mask_shape(
        count_options.postselection_mask,
        sampler.detector_ids().len(),
        "postselection_mask",
    )?;
    validate_mask_shape(
        count_options.postselected_observables_mask,
        sampler.observable_ids().len(),
        "postselected_observables_mask",
    )?;
    let corrections = match decoder {
        Some(decoder) => Some(decoder.decode_batch_checked(sampled.syndrome())?),
        None => None,
    };
    let residuals = detailed_residual_batch(
        sampled.observables(),
        corrections,
        sampler.observable_ids(),
        shots,
    )?;
    let detector_discard_mask = detector_postselection_loss_mask(
        sampled,
        sampler.detector_ids(),
        shots,
        count_options.postselection_mask,
    )?;
    let mut custom_counts = HashMap::new();
    if count_options.count_detection_events {
        let events = sampled
            .detail_detector_masks()
            .iter()
            .map(Mask::bit_count)
            .sum::<usize>();
        custom_counts.insert("detection_events".to_string(), events);
        custom_counts.insert(
            "detectors_checked".to_string(),
            shots * sampler.detector_ids().len(),
        );
    }

    let (errors, discards, loss_mask) = count_residual_batch(
        &residuals,
        sampler.observable_ids().len(),
        shots,
        detector_discard_mask.as_ref(),
        count_options.postselected_observables_mask,
        count_options.count_observable_error_combos,
        &mut custom_counts,
    );

    Ok(DetailedBatchResult {
        stats: BatchStats {
            shots,
            errors,
            discards,
            seconds: 0.0,
            custom_counts,
        },
        loss_mask,
    })
}

enum DetailedResidualBatch<'a> {
    Masks(Cow<'a, [Mask]>),
    Packed {
        data: Cow<'a, [u8]>,
        observable_byte_count: usize,
    },
}

impl DetailedResidualBatch<'_> {
    fn bit(&self, observable: usize, shot: usize) -> bool {
        match self {
            Self::Masks(masks) => mask_bit(&masks[observable], shot),
            Self::Packed {
                data,
                observable_byte_count,
            } => {
                data[shot * observable_byte_count + (observable >> 3)] & (1u8 << (observable & 7))
                    != 0
            }
        }
    }
}

fn detailed_residual_batch<'a>(
    observables: &'a faultscope_core::DemObservableBatch,
    corrections: Option<DecoderCorrectionBatch>,
    observable_ids: &[i64],
    shots: usize,
) -> NpResult<DetailedResidualBatch<'a>> {
    match (observables, corrections) {
        (faultscope_core::DemObservableBatch::Masks(actual), None) => {
            actual.validate_against(observable_ids, shots)?;
            Ok(DetailedResidualBatch::Masks(Cow::Borrowed(&actual.masks)))
        }
        (faultscope_core::DemObservableBatch::Packed(actual), None) => {
            actual.validate_against(observable_ids, shots)?;
            Ok(DetailedResidualBatch::Packed {
                data: Cow::Borrowed(&actual.data),
                observable_byte_count: actual.observable_byte_count,
            })
        }
        (
            faultscope_core::DemObservableBatch::Masks(actual),
            Some(DecoderCorrectionBatch::Masks(correction)),
        ) => {
            actual.validate_against(observable_ids, shots)?;
            correction.validate_against(observable_ids, shots)?;
            Ok(DetailedResidualBatch::Masks(Cow::Owned(
                residual_masks(actual, &correction, observable_ids, shots)
                    .into_iter()
                    .map(|(_, mask)| mask)
                    .collect(),
            )))
        }
        (
            faultscope_core::DemObservableBatch::Packed(actual),
            Some(DecoderCorrectionBatch::Packed(correction)),
        ) => {
            actual.validate_against(observable_ids, shots)?;
            correction.validate_against(observable_ids, shots)?;
            Ok(DetailedResidualBatch::Packed {
                data: Cow::Owned(
                    actual
                        .data
                        .iter()
                        .zip(&correction.data)
                        .map(|(actual, correction)| actual ^ correction)
                        .collect(),
                ),
                observable_byte_count: actual.observable_byte_count,
            })
        }
        _ => Err(NpError::new(
            "DEM observable truth layout does not match the decoder correction layout",
        )),
    }
}

fn count_residual_batch(
    residuals: &DetailedResidualBatch<'_>,
    observable_count: usize,
    shots: usize,
    detector_discard_mask: Option<&Mask>,
    postselected_observables_mask: Option<&[u8]>,
    count_observable_error_combos: bool,
    custom_counts: &mut HashMap<String, usize>,
) -> (usize, usize, Mask) {
    if count_observable_error_combos {
        if observable_count == 1 {
            let (errors, discards, loss) = count_residual_batch(
                residuals,
                observable_count,
                shots,
                detector_discard_mask,
                postselected_observables_mask,
                false,
                custom_counts,
            );
            if errors != 0 {
                *custom_counts
                    .entry("obs_mistake_mask=E".to_string())
                    .or_insert(0) += errors;
            }
            return (errors, discards, loss);
        }
        return count_residual_combos(
            residuals,
            observable_count,
            shots,
            detector_discard_mask,
            postselected_observables_mask,
            custom_counts,
        );
    }

    match residuals {
        DetailedResidualBatch::Masks(masks) => {
            let mut discard = detector_discard_mask
                .cloned()
                .unwrap_or_else(|| Mask::zero(word_count(shots)));
            let mut loss = Mask::zero(word_count(shots));
            for (index, residual) in masks.iter().enumerate() {
                if packed_mask_bit(postselected_observables_mask, index) {
                    discard.or_assign(residual);
                } else {
                    loss.or_assign(residual);
                }
            }
            discard.clear_unused(shots);
            for (loss_word, discard_word) in loss.words.iter_mut().zip(&discard.words) {
                *loss_word &= !discard_word;
            }
            loss.clear_unused(shots);
            (loss.bit_count(), discard.bit_count(), loss)
        }
        DetailedResidualBatch::Packed {
            data,
            observable_byte_count,
        } => {
            let mut discard = detector_discard_mask
                .cloned()
                .unwrap_or_else(|| Mask::zero(word_count(shots)));
            let mut loss = Mask::zero(word_count(shots));
            for shot in 0..shots {
                if mask_bit(&discard, shot) {
                    continue;
                }
                let start = shot * observable_byte_count;
                let row = &data[start..start + observable_byte_count];
                let mut observable_discard = false;
                let mut logical_error = false;
                for (byte_index, bits) in row.iter().copied().enumerate() {
                    let postselected = postselected_observables_mask
                        .and_then(|mask| mask.get(byte_index))
                        .copied()
                        .unwrap_or(0);
                    observable_discard |= bits & postselected != 0;
                    logical_error |= bits & !postselected != 0;
                }
                if observable_discard {
                    set_mask_bit(&mut discard, shot);
                } else if logical_error {
                    set_mask_bit(&mut loss, shot);
                }
            }
            discard.clear_unused(shots);
            loss.clear_unused(shots);
            (loss.bit_count(), discard.bit_count(), loss)
        }
    }
}

fn count_residual_combos(
    residuals: &DetailedResidualBatch<'_>,
    observable_count: usize,
    shots: usize,
    detector_discard_mask: Option<&Mask>,
    postselected_observables_mask: Option<&[u8]>,
    custom_counts: &mut HashMap<String, usize>,
) -> (usize, usize, Mask) {
    let mut discard = detector_discard_mask
        .cloned()
        .unwrap_or_else(|| Mask::zero(word_count(shots)));
    let mut loss = Mask::zero(word_count(shots));
    let use_numeric_combos = observable_count <= u64::BITS as usize;
    let mut numeric_combos = HashMap::<u64, usize>::new();
    for shot in 0..shots {
        if mask_bit(&discard, shot) {
            continue;
        }
        let mut observable_discard = false;
        let mut logical_error = false;
        let mut combo_bits = 0u64;
        for index in 0..observable_count {
            let bit = residuals.bit(index, shot);
            let postselected = packed_mask_bit(postselected_observables_mask, index);
            observable_discard |= bit && postselected;
            logical_error |= bit && !postselected;
            if use_numeric_combos && bit {
                combo_bits |= 1u64 << index;
            }
        }
        if observable_discard {
            set_mask_bit(&mut discard, shot);
            continue;
        }
        if !logical_error {
            continue;
        }

        set_mask_bit(&mut loss, shot);
        if use_numeric_combos {
            *numeric_combos.entry(combo_bits).or_insert(0) += 1;
        } else {
            let mut combo = String::with_capacity("obs_mistake_mask=".len() + observable_count);
            combo.push_str("obs_mistake_mask=");
            for index in 0..observable_count {
                combo.push(if residuals.bit(index, shot) { 'E' } else { '_' });
            }
            *custom_counts.entry(combo).or_insert(0) += 1;
        }
    }
    for (combo_bits, count) in numeric_combos {
        let mut combo = String::with_capacity("obs_mistake_mask=".len() + observable_count);
        combo.push_str("obs_mistake_mask=");
        for index in 0..observable_count {
            combo.push(if combo_bits & (1u64 << index) != 0 {
                'E'
            } else {
                '_'
            });
        }
        *custom_counts.entry(combo).or_insert(0) += count;
    }
    discard.clear_unused(shots);
    loss.clear_unused(shots);
    (loss.bit_count(), discard.bit_count(), loss)
}

pub(crate) fn validate_decoder_observable_layout(
    canonical_observable_ids: &[i64],
    decoder_observable_ids: &[i64],
    decoder_name: &str,
) -> NpResult<()> {
    if decoder_observable_ids != canonical_observable_ids {
        return Err(NpError::new(format!(
            "native decoder `{decoder_name}` observable layout mismatch: expected sampler canonical ids {:?}, got {:?}",
            canonical_observable_ids, decoder_observable_ids
        )));
    }
    Ok(())
}

fn count_decoder_sampling_result(
    sampled: &DemSamplingResult,
    decoder: &mut dyn NativeDecoderWorker,
) -> NpResult<usize> {
    let corrections = decoder.decode_batch_checked(sampled.syndrome())?;
    match (sampled.observables(), corrections) {
        (
            faultscope_core::DemObservableBatch::Masks(actual),
            DecoderCorrectionBatch::Masks(correction),
        ) => {
            let residuals = residual_masks(
                actual,
                &correction,
                actual.observable_ids.as_slice(),
                sampled.shots(),
            );
            let mut loss = Mask::zero(word_count(sampled.shots()));
            for (_, residual) in residuals {
                loss.or_assign(&residual);
            }
            loss.and_assign(sampled.all_mask());
            Ok(loss.bit_count())
        }
        (
            faultscope_core::DemObservableBatch::Packed(actual),
            DecoderCorrectionBatch::Packed(correction),
        ) => packed_residual_failure_count(
            &actual.observable_ids,
            &actual.data,
            actual.observable_byte_count,
            &correction,
            actual.shots,
        ),
        _ => Err(NpError::new(
            "DEM observable truth layout does not match the decoder correction layout",
        )),
    }
}

fn residual_masks(
    observables: &CorrectionMaskBatch,
    corrections: &CorrectionMaskBatch,
    observable_ids: &[i64],
    shots: usize,
) -> Vec<(i64, Mask)> {
    let words = word_count(shots);
    let zero = Mask::zero(words);
    observable_ids
        .iter()
        .map(|observable_id| {
            let actual = observables.get(*observable_id).unwrap_or(&zero);
            let correction = corrections.get(*observable_id).unwrap_or(&zero);
            let mut residual = actual.clone();
            residual.xor_assign(correction);
            residual.clear_unused(shots);
            (*observable_id, residual)
        })
        .collect()
}

fn detector_postselection_loss_mask(
    sampled: &DemSamplingResult,
    detector_ids: &[i64],
    shots: usize,
    postselection_mask: Option<&[u8]>,
) -> NpResult<Option<Mask>> {
    let Some(postselection_mask) = postselection_mask else {
        return Ok(None);
    };
    let mut discard = Mask::zero(word_count(shots));
    for (index, detector_id) in detector_ids.iter().enumerate() {
        if packed_mask_bit(Some(postselection_mask), index) {
            let detector = sampled
                .detector_detail_mask(*detector_id)
                .ok_or_else(|| NpError::new(format!("missing detector id {detector_id}")))?;
            discard.or_assign(detector);
        }
    }
    discard.clear_unused(shots);
    Ok(Some(discard))
}

pub(crate) fn validate_mask_shape(
    mask: Option<&[u8]>,
    bit_count: usize,
    name: &str,
) -> NpResult<()> {
    if let Some(mask) = mask {
        let expected = bit_count.div_ceil(8);
        if mask.len() != expected {
            return Err(NpError::new(format!(
                "{name} has {} bytes; expected {expected}",
                mask.len()
            )));
        }
        let used_bits_in_last_byte = bit_count & 7;
        if used_bits_in_last_byte != 0 {
            let valid_bits = (1u8 << used_bits_in_last_byte) - 1;
            if mask
                .last()
                .is_some_and(|last_byte| last_byte & !valid_bits != 0)
            {
                return Err(NpError::new(format!(
                    "{name} has non-zero unused padding bits in its final byte"
                )));
            }
        }
    }
    Ok(())
}

fn mask_bit(mask: &Mask, shot: usize) -> bool {
    mask.words
        .get(shot >> 6)
        .is_some_and(|word| ((word >> (shot & 63)) & 1) != 0)
}

fn set_mask_bit(mask: &mut Mask, shot: usize) {
    if let Some(word) = mask.words.get_mut(shot >> 6) {
        *word |= 1u64 << (shot & 63);
    }
}

fn packed_mask_bit(mask: Option<&[u8]>, index: usize) -> bool {
    mask.and_then(|mask| mask.get(index >> 3))
        .is_some_and(|byte| ((byte >> (index & 7)) & 1) != 0)
}

fn packed_mask_has_any(mask: Option<&[u8]>) -> bool {
    mask.is_some_and(|mask| mask.iter().any(|byte| *byte != 0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_masks_do_not_force_detailed_counting() {
        let zero = [0u8];
        let selected = [1u8];

        assert!(!CountOptions {
            postselection_mask: Some(&zero),
            postselected_observables_mask: Some(&zero),
            ..CountOptions::default()
        }
        .uses_detailed_path());
        assert!(CountOptions {
            postselection_mask: Some(&selected),
            ..CountOptions::default()
        }
        .uses_detailed_path());
        assert!(CountOptions {
            postselected_observables_mask: Some(&selected),
            ..CountOptions::default()
        }
        .uses_detailed_path());
    }

    #[test]
    fn mask_shapes_reject_nonzero_unused_padding_bits() {
        validate_mask_shape(Some(&[0b0000_0001]), 1, "mask").unwrap();
        validate_mask_shape(Some(&[0xff]), 8, "mask").unwrap();
        validate_mask_shape(Some(&[0xff, 0b0000_0001]), 9, "mask").unwrap();

        let err = validate_mask_shape(Some(&[0b1000_0000]), 1, "mask").unwrap_err();
        assert_eq!(
            err.message(),
            "mask has non-zero unused padding bits in its final byte"
        );
        let err = validate_mask_shape(Some(&[0xff, 0b0000_0010]), 9, "mask").unwrap_err();
        assert_eq!(
            err.message(),
            "mask has non-zero unused padding bits in its final byte"
        );
    }

    #[test]
    fn residual_counting_matches_for_mask_and_packed_layouts() {
        let detector_discard = Mask {
            words: vec![0b00_1001],
        };
        let postselected = [0b0000_0010];
        let mask_residuals = DetailedResidualBatch::Masks(Cow::Owned(vec![
            Mask {
                words: vec![0b01_0011],
            },
            Mask {
                words: vec![0b10_0110],
            },
        ]));
        let packed_residuals = DetailedResidualBatch::Packed {
            data: Cow::Owned(vec![0b01, 0b11, 0b10, 0b00, 0b01, 0b10]),
            observable_byte_count: 1,
        };

        for residuals in [&mask_residuals, &packed_residuals] {
            let mut counts = HashMap::new();
            let (errors, discards, loss) = count_residual_batch(
                residuals,
                2,
                6,
                Some(&detector_discard),
                Some(&postselected),
                false,
                &mut counts,
            );
            assert_eq!((errors, discards), (1, 5));
            assert_eq!(loss.words, vec![0b01_0000]);
            assert!(counts.is_empty());

            let (errors, discards, loss) = count_residual_batch(
                residuals,
                2,
                6,
                Some(&detector_discard),
                Some(&postselected),
                true,
                &mut counts,
            );
            assert_eq!((errors, discards), (1, 5));
            assert_eq!(loss.words, vec![0b01_0000]);
            assert_eq!(counts.get("obs_mistake_mask=E_"), Some(&1));
        }
    }

    struct FormatZeroDecoder {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        formats: Vec<DetectorBatchFormat>,
    }

    impl NativeDecoderWorker for FormatZeroDecoder {
        fn name(&self) -> &str {
            "format-zero"
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn batch_formats(&self) -> &[DetectorBatchFormat] {
            &self.formats
        }

        fn decode_batch(
            &mut self,
            detectors: faultscope_core::DetectorBatchView<'_>,
        ) -> NpResult<DecoderCorrectionBatch> {
            match detectors {
                faultscope_core::DetectorBatchView::Masks(view) => {
                    Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                        self.observable_ids.clone(),
                        vec![Mask::zero(word_count(view.shots)); self.observable_ids.len()],
                        view.shots,
                    )?))
                }
                faultscope_core::DetectorBatchView::Packed(view) => {
                    Ok(DecoderCorrectionBatch::Packed(
                        faultscope_core::PackedObservableShotBatch::new(
                            self.observable_ids.clone(),
                            vec![0; view.shots * self.observable_ids.len().div_ceil(8)],
                            view.shots,
                        )?,
                    ))
                }
                faultscope_core::DetectorBatchView::Events(view) => {
                    Ok(DecoderCorrectionBatch::Packed(
                        faultscope_core::PackedObservableShotBatch::new(
                            self.observable_ids.clone(),
                            vec![0; view.shots * self.observable_ids.len().div_ceil(8)],
                            view.shots,
                        )?,
                    ))
                }
            }
        }
    }

    #[test]
    fn prepared_plans_reject_duplicate_decoder_detector_ids() {
        let sampler =
            DemHotspotEstimator::from_sampling_parts(vec![10], vec![0], Vec::new()).unwrap();
        let duplicate_ids = [10, 10];
        let options = [
            CountOptions::default(),
            CountOptions {
                count_detection_events: true,
                ..CountOptions::default()
            },
        ];

        for option in options {
            let error = prepare_dem_count_plan(
                &sampler,
                Some((&duplicate_ids, &[DetectorBatchFormat::Masks])),
                &option,
                false,
            )
            .unwrap_err();
            assert_eq!(
                error.message(),
                "decoder detector ids must be unique; duplicate detector id 10"
            );
        }
    }

    #[test]
    fn detailed_counting_is_identical_for_masks_packed_and_events() {
        let sampler = DemHotspotEstimator::from_sampling_parts(
            vec![10, 20, 30],
            vec![0, 1],
            vec![
                faultscope_core::DetectorErrorEdge {
                    probability: 0.19,
                    detectors: vec![10, 20],
                    observables: vec![0],
                    location_id: "a".to_string(),
                    event: faultscope_core::DemEvent::Bool(true),
                    tags: HashMap::new(),
                },
                faultscope_core::DetectorErrorEdge {
                    probability: 0.43,
                    detectors: vec![20, 30],
                    observables: vec![1],
                    location_id: "b".to_string(),
                    event: faultscope_core::DemEvent::Bool(true),
                    tags: HashMap::new(),
                },
                faultscope_core::DetectorErrorEdge {
                    probability: 0.71,
                    detectors: vec![10, 30],
                    observables: vec![0, 1],
                    location_id: "c".to_string(),
                    event: faultscope_core::DemEvent::Bool(true),
                    tags: HashMap::new(),
                },
            ],
        )
        .unwrap();
        let postselection = [0b0000_0101];
        let postselected_observables = [0b0000_0010];
        let options = CountOptions {
            postselection_mask: Some(&postselection),
            postselected_observables_mask: Some(&postselected_observables),
            count_observable_error_combos: true,
            count_detection_events: true,
        };
        let mut expected: Option<(BatchStats, Mask, u64)> = None;
        for format in DetectorBatchFormat::STABLE_ORDER {
            let formats = [format];
            let PreparedDemCountPlan::Decoder(plan) = prepare_dem_count_plan(
                &sampler,
                Some((sampler.detector_ids(), &formats)),
                &options,
                false,
            )
            .unwrap() else {
                unreachable!()
            };
            assert_eq!(plan.format(), format);
            assert_eq!(plan.detail_detector_ids(), sampler.detector_ids());
            assert!(!plan.records_attribution());

            let mut rng = SmallRng::new(0x1234_5678);
            let sampled = plan.run_sampling_result_with_rng(129, &mut rng).unwrap();
            let mut decoder = FormatZeroDecoder {
                detector_ids: sampler.detector_ids().to_vec(),
                observable_ids: sampler.observable_ids().to_vec(),
                formats: formats.to_vec(),
            };
            let detailed =
                count_detailed_sampling_result(&sampler, &sampled, Some(&mut decoder), &options)
                    .unwrap();
            let actual = (detailed.stats, detailed.loss_mask, rng.next_u64());
            if let Some(expected) = &expected {
                assert_eq!(&actual, expected);
            } else {
                expected = Some(actual);
            }
        }
    }
}
