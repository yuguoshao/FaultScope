use std::collections::HashMap;
use std::time::Instant;

use faultscope_core::{
    logical_residual_loss_mask_native, packed_residual_failure_count, word_count,
    CompiledDemLogicalCountPlan, CompiledDemSamplingPlan, CorrectionMaskBatch, DemBatch,
    DemHotspotEstimator, DetectorEventShotBatchView, DetectorMaskBatchView, Mask,
    NativeDecoderWorker, NpError, NpResult, PackedDetectorShotBatchView, SmallRng,
};

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
    fn uses_detailed_path(&self) -> bool {
        self.postselection_mask.is_some()
            || self.postselected_observables_mask.is_some()
            || self.count_observable_error_combos
            || self.count_detection_events
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum PreparedDemCountPlan {
    Generic,
    Logical(CompiledDemLogicalCountPlan),
    Decoder(CompiledDemSamplingPlan),
}

pub(crate) fn prepare_dem_count_plan(
    sampler: &DemHotspotEstimator,
    detector_ids: Option<&[i64]>,
    count_options: &CountOptions<'_>,
) -> NpResult<PreparedDemCountPlan> {
    if count_options.uses_detailed_path() {
        return Ok(PreparedDemCountPlan::Generic);
    }
    match detector_ids {
        Some(detector_ids) => Ok(sampler
            .compile_sampling_plan(detector_ids, sampler.observable_ids())
            .map(PreparedDemCountPlan::Decoder)
            .unwrap_or(PreparedDemCountPlan::Generic)),
        None => Ok(PreparedDemCountPlan::Logical(
            sampler.compile_logical_count_plan(),
        )),
    }
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
    let mut out = if count_options.uses_detailed_path() {
        sample_detailed_batch(sampler, shots, rng, decoder, count_options)?
    } else {
        let errors = match decoder {
            Some(decoder) => sample_dem_logical_error_count_with_decoder(
                sampler,
                shots,
                rng,
                decoder,
                prepared_plan,
            )?,
            None => match prepared_plan {
                PreparedDemCountPlan::Logical(plan) => {
                    plan.sample_logical_error_count_with_rng(shots, rng)
                }
                PreparedDemCountPlan::Generic | PreparedDemCountPlan::Decoder(_) => sampler
                    .compile_logical_count_plan()
                    .sample_logical_error_count_with_rng(shots, rng),
            },
        };
        BatchStats {
            shots,
            errors,
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

fn sample_detailed_batch(
    sampler: &DemHotspotEstimator,
    shots: usize,
    rng: &mut SmallRng,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    count_options: &CountOptions<'_>,
) -> NpResult<BatchStats> {
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

    let batch = sampler.run_batch_with_rng(shots, rng, false);
    Ok(count_detailed_batch(sampler, &batch, decoder, count_options)?.stats)
}

pub(crate) fn count_detailed_batch(
    sampler: &DemHotspotEstimator,
    batch: &DemBatch,
    decoder: Option<&mut dyn NativeDecoderWorker>,
    count_options: &CountOptions<'_>,
) -> NpResult<DetailedBatchResult> {
    let shots = batch.shots();
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
        Some(decoder) => {
            let detector_ids = decoder.detector_ids().to_vec();
            let detector_masks = detector_mask_view_from_map(&batch.detectors, &detector_ids)?;
            let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, shots)?;
            decoder.decode_batch_checked(view)?
        }
        None => CorrectionMaskBatch::empty(shots),
    };

    let residuals = residual_masks(
        &batch.observables,
        &corrections,
        sampler.observable_ids(),
        shots,
    );
    let mut loss_mask = Mask::zero(word_count(shots));
    let detector_discard_mask = detector_postselection_loss_mask(
        &batch.detectors,
        sampler.detector_ids(),
        shots,
        count_options.postselection_mask,
    )?;
    let mut custom_counts = HashMap::new();
    if count_options.count_detection_events {
        let events = batch.detectors.values().map(Mask::bit_count).sum::<usize>();
        custom_counts.insert("detection_events".to_string(), events);
        custom_counts.insert(
            "detectors_checked".to_string(),
            shots * sampler.detector_ids().len(),
        );
    }

    let mut errors = 0usize;
    let mut discards = 0usize;
    for shot in 0..shots {
        if detector_discard_mask
            .as_ref()
            .is_some_and(|mask| mask_bit(mask, shot))
        {
            discards += 1;
            continue;
        }

        let mut observable_discard = false;
        let mut logical_error = false;
        let mut combo = String::with_capacity(sampler.observable_ids().len());
        for (index, (_, residual)) in residuals.iter().enumerate() {
            let bit = mask_bit(residual, shot);
            let postselected = packed_mask_bit(count_options.postselected_observables_mask, index);
            if bit && postselected {
                observable_discard = true;
            }
            if bit && !postselected {
                logical_error = true;
            }
            if count_options.count_observable_error_combos {
                combo.push(if bit { 'E' } else { '_' });
            }
        }
        if observable_discard {
            discards += 1;
            continue;
        }
        if logical_error {
            errors += 1;
            set_mask_bit(&mut loss_mask, shot);
            if count_options.count_observable_error_combos {
                *custom_counts
                    .entry(format!("obs_mistake_mask={combo}"))
                    .or_insert(0) += 1;
            }
        }
    }

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

fn sample_dem_logical_error_count_with_decoder(
    sampler: &DemHotspotEstimator,
    shots: usize,
    rng: &mut SmallRng,
    decoder: &mut dyn NativeDecoderWorker,
    prepared_plan: &PreparedDemCountPlan,
) -> NpResult<usize> {
    if decoder.supports_detector_event_batch() {
        let detector_ids = decoder.detector_ids().to_vec();
        let fallback;
        let plan = match prepared_plan {
            PreparedDemCountPlan::Decoder(plan)
                if plan.detector_ids() == detector_ids.as_slice()
                    && plan.observable_ids() == sampler.observable_ids() =>
            {
                plan
            }
            PreparedDemCountPlan::Generic
            | PreparedDemCountPlan::Logical(_)
            | PreparedDemCountPlan::Decoder(_) => {
                fallback =
                    sampler.compile_sampling_plan(&detector_ids, sampler.observable_ids())?;
                &fallback
            }
        };
        let event_batch = plan.run_detector_event_shot_batch_with_rng(shots, rng);
        let detector_view = DetectorEventShotBatchView::new(
            &detector_ids,
            &event_batch.offsets,
            &event_batch.events,
            event_batch.shots,
        )?;
        let corrections = decoder.decode_detector_event_batch_checked(detector_view)?;
        return packed_residual_failure_count(
            &event_batch.observable_ids,
            &event_batch.observable_data,
            event_batch.observable_byte_count,
            &corrections,
            event_batch.shots,
        );
    }

    if decoder.supports_packed_batch() {
        let detector_ids = decoder.detector_ids().to_vec();
        let fallback;
        let plan = match prepared_plan {
            PreparedDemCountPlan::Decoder(plan)
                if plan.detector_ids() == detector_ids.as_slice()
                    && plan.observable_ids() == sampler.observable_ids() =>
            {
                plan
            }
            PreparedDemCountPlan::Generic
            | PreparedDemCountPlan::Logical(_)
            | PreparedDemCountPlan::Decoder(_) => {
                fallback =
                    sampler.compile_sampling_plan(&detector_ids, sampler.observable_ids())?;
                &fallback
            }
        };
        let packed_batch = plan.run_packed_shot_batch_with_rng(shots, rng);
        let detector_view = PackedDetectorShotBatchView::new(
            &detector_ids,
            &packed_batch.detector_data,
            packed_batch.shots,
        )?;
        let corrections = decoder.decode_packed_batch_checked(detector_view)?;
        return packed_residual_failure_count(
            &packed_batch.observable_ids,
            &packed_batch.observable_data,
            packed_batch.observable_byte_count,
            &corrections,
            packed_batch.shots,
        );
    }

    let batch = sampler.run_batch_with_rng(shots, rng, false);
    let detector_ids = decoder.detector_ids().to_vec();
    let detector_masks = detector_mask_view_from_map(&batch.detectors, &detector_ids)?;
    let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, shots)?;
    let corrections = decoder.decode_batch_checked(view)?;
    let loss_mask = logical_residual_loss_mask_native(
        &batch.observables,
        &corrections,
        sampler.observable_ids(),
        batch.all_mask(),
    );
    Ok(loss_mask.bit_count())
}

fn detector_mask_view_from_map(
    detectors: &HashMap<i64, Mask>,
    detector_ids: &[i64],
) -> NpResult<Vec<Mask>> {
    detector_ids
        .iter()
        .map(|detector_id| {
            detectors
                .get(detector_id)
                .cloned()
                .ok_or_else(|| NpError::new(format!("missing detector id {detector_id}")))
        })
        .collect()
}

fn residual_masks(
    observables: &HashMap<i64, Mask>,
    corrections: &CorrectionMaskBatch,
    observable_ids: &[i64],
    shots: usize,
) -> Vec<(i64, Mask)> {
    let words = word_count(shots);
    let zero = Mask::zero(words);
    observable_ids
        .iter()
        .map(|observable_id| {
            let actual = observables.get(observable_id).unwrap_or(&zero);
            let correction = corrections.get(*observable_id).unwrap_or(&zero);
            let mut residual = actual.clone();
            residual.xor_assign(correction);
            residual.clear_unused(shots);
            (*observable_id, residual)
        })
        .collect()
}

fn detector_postselection_loss_mask(
    detectors: &HashMap<i64, Mask>,
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
            let detector = detectors
                .get(detector_id)
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
