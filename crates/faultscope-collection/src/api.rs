use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use faultscope_core::{
    logical_residual_loss_mask_native, DemHotspotEstimator, DetectorEventShotBatchView,
    DetectorMaskBatchView, Mask, NativeBatchDecoder, NpError, NpResult,
    PackedDetectorShotBatchView, PackedObservableShotBatch, SmallRng,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DemLogicalCollectionOptions {
    pub max_shots: usize,
    pub max_errors: Option<usize>,
    pub batch_size: usize,
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemLogicalCollectionStats {
    pub shots: usize,
    pub errors: usize,
    pub discards: usize,
    pub seconds: f64,
}

pub fn collect_dem_logical_error_stats(
    sampler: &DemHotspotEstimator,
    options: DemLogicalCollectionOptions,
    decoder: Option<&dyn NativeBatchDecoder>,
) -> NpResult<DemLogicalCollectionStats> {
    if options.max_shots == 0 {
        return Err(NpError::new("max_shots must be positive"));
    }
    if options.batch_size == 0 {
        return Err(NpError::new("batch_size must be positive"));
    }

    let started = Instant::now();
    let mut collection_rng = SmallRng::new(collection_seed(options.seed));
    let mut shots_done = 0usize;
    let mut errors = 0usize;

    while shots_done < options.max_shots {
        let batch_shots = options.batch_size.min(options.max_shots - shots_done);
        let batch_seed = collection_rng.next_u64();
        let mut batch_rng = SmallRng::new(batch_seed);
        let batch_stats = sample_dem_logical_error_stats_with_rng(
            sampler,
            batch_shots,
            &mut batch_rng,
            decoder,
            None,
        )?;
        shots_done += batch_stats.shots;
        errors += batch_stats.errors;
        if options.max_errors.is_some_and(|limit| errors >= limit) {
            break;
        }
    }

    Ok(DemLogicalCollectionStats {
        shots: shots_done,
        errors,
        discards: 0,
        seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn sample_dem_logical_error_stats(
    sampler: &DemHotspotEstimator,
    shots: usize,
    seed: Option<u64>,
    decoder: Option<&dyn NativeBatchDecoder>,
) -> NpResult<DemLogicalCollectionStats> {
    if shots == 0 {
        return Err(NpError::new("shots must be positive"));
    }
    let started = Instant::now();
    let mut rng = SmallRng::new(seed.unwrap_or(0x95f2_04dc_4291_a715));
    sample_dem_logical_error_stats_with_rng(sampler, shots, &mut rng, decoder, Some(started))
}

fn sample_dem_logical_error_stats_with_rng(
    sampler: &DemHotspotEstimator,
    shots: usize,
    rng: &mut SmallRng,
    decoder: Option<&dyn NativeBatchDecoder>,
    started: Option<Instant>,
) -> NpResult<DemLogicalCollectionStats> {
    let errors = match decoder {
        Some(decoder) => sample_dem_logical_error_count_with_decoder(sampler, shots, rng, decoder)?,
        None => {
            let batch = sampler.run_batch_with_rng(shots, rng, false);
            batch.loss_mask.bit_count()
        }
    };
    Ok(DemLogicalCollectionStats {
        shots,
        errors,
        discards: 0,
        seconds: started
            .map(|started| started.elapsed().as_secs_f64())
            .unwrap_or(0.0),
    })
}

fn sample_dem_logical_error_count_with_decoder(
    sampler: &DemHotspotEstimator,
    shots: usize,
    rng: &mut SmallRng,
    decoder: &dyn NativeBatchDecoder,
) -> NpResult<usize> {
    if decoder.supports_detector_event_batch() {
        let event_batch = sampler.run_detector_event_shot_batch_with_rng(
            shots,
            rng,
            decoder.detector_ids(),
            &sampler.observable_ids,
        )?;
        let detector_view = DetectorEventShotBatchView::new(
            decoder.detector_ids(),
            &event_batch.offsets,
            &event_batch.events,
            event_batch.shots,
        )?;
        let corrections = decoder.decode_detector_event_batch_checked(detector_view)?;
        return collection_packed_residual_failure_count_from_rows(
            &event_batch.observable_ids,
            &event_batch.observable_data,
            event_batch.observable_byte_count,
            &corrections,
            event_batch.shots,
        );
    }

    if decoder.supports_packed_batch() {
        let packed_batch = sampler.run_packed_shot_batch_with_rng(
            shots,
            rng,
            decoder.detector_ids(),
            &sampler.observable_ids,
        )?;
        let detector_view = PackedDetectorShotBatchView::new(
            decoder.detector_ids(),
            &packed_batch.detector_data,
            packed_batch.shots,
        )?;
        let corrections = decoder.decode_packed_batch_checked(detector_view)?;
        return collection_packed_residual_failure_count_from_rows(
            &packed_batch.observable_ids,
            &packed_batch.observable_data,
            packed_batch.observable_byte_count,
            &corrections,
            packed_batch.shots,
        );
    }

    let batch = sampler.run_batch_with_rng(shots, rng, false);
    let detector_masks = detector_mask_view_from_map(&batch.detectors, decoder.detector_ids())?;
    let view = DetectorMaskBatchView::new(decoder.detector_ids(), &detector_masks, shots)?;
    let corrections = decoder.decode_batch_checked(view)?;
    let loss_mask = logical_residual_loss_mask_native(
        &batch.observables,
        &corrections,
        &sampler.observable_ids,
        &batch.all_mask,
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

fn collection_packed_residual_failure_count_from_rows(
    observable_ids: &[i64],
    observable_data: &[u8],
    observable_byte_count: usize,
    corrections: &PackedObservableShotBatch,
    shots: usize,
) -> NpResult<usize> {
    if corrections.observable_ids.as_slice() == observable_ids
        && corrections.observable_byte_count == observable_byte_count
    {
        let failures = (0..shots)
            .filter(|shot| {
                let begin = shot * observable_byte_count;
                let end = begin + observable_byte_count;
                observable_data[begin..end]
                    .iter()
                    .zip(&corrections.data[begin..end])
                    .any(|(actual, correction)| (actual ^ correction) != 0)
            })
            .count();
        return Ok(failures);
    }

    let mut ids = observable_ids.to_vec();
    for observable_id in &corrections.observable_ids {
        if !ids.contains(observable_id) {
            ids.push(*observable_id);
        }
    }
    let actual_index = observable_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<HashMap<_, _>>();
    let correction_index = corrections
        .observable_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect::<HashMap<_, _>>();

    let mut failures = 0usize;
    for shot in 0..shots {
        let mut failed = false;
        for observable_id in &ids {
            let actual = actual_index
                .get(observable_id)
                .map(|index| {
                    let offset = shot * observable_byte_count + (index >> 3);
                    ((observable_data[offset] >> (index & 7)) & 1) != 0
                })
                .unwrap_or(false);
            let correction = correction_index
                .get(observable_id)
                .map(|index| {
                    let offset = shot * corrections.observable_byte_count + (index >> 3);
                    ((corrections.data[offset] >> (index & 7)) & 1) != 0
                })
                .unwrap_or(false);
            if actual ^ correction {
                failed = true;
                break;
            }
        }
        failures += usize::from(failed);
    }
    Ok(failures)
}

fn collection_seed(seed: Option<u64>) -> u64 {
    seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_nanos() as u64)
            .unwrap_or(0x95f2_04dc_4291_a715)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use faultscope_core::PackedObservableShotBatch;

    #[test]
    fn collection_counts_packed_residual_failures_locally() {
        let corrections =
            PackedObservableShotBatch::new(vec![1], vec![0b0000_0000, 0b0000_0001, 0b0000_0000], 3)
                .unwrap();

        let failures = collection_packed_residual_failure_count_from_rows(
            &[0],
            &[0b0000_0000, 0b0000_0001, 0b0000_0001],
            1,
            &corrections,
            3,
        )
        .unwrap();

        assert_eq!(failures, 2);
    }
}
