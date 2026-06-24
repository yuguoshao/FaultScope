use std::collections::HashMap;

use crate::{Mask, NpError, NpResult};

#[derive(Debug, Clone, Copy)]
pub struct DetectorMaskBatchView<'a> {
    pub detector_ids: &'a [i64],
    pub masks: &'a [Mask],
    pub shots: usize,
}

impl<'a> DetectorMaskBatchView<'a> {
    pub fn new(detector_ids: &'a [i64], masks: &'a [Mask], shots: usize) -> NpResult<Self> {
        if detector_ids.len() != masks.len() {
            return Err(NpError::new(
                "detector ids and detector masks must have the same length",
            ));
        }
        Ok(Self {
            detector_ids,
            masks,
            shots,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CorrectionMaskBatch {
    pub observable_ids: Vec<i64>,
    pub masks: Vec<Mask>,
    pub shots: usize,
}

impl CorrectionMaskBatch {
    pub fn new(observable_ids: Vec<i64>, masks: Vec<Mask>, shots: usize) -> NpResult<Self> {
        if observable_ids.len() != masks.len() {
            return Err(NpError::new(
                "observable ids and correction masks must have the same length",
            ));
        }
        Ok(Self {
            observable_ids,
            masks,
            shots,
        })
    }

    pub fn empty(shots: usize) -> Self {
        Self {
            observable_ids: Vec::new(),
            masks: Vec::new(),
            shots,
        }
    }

    pub fn get(&self, observable_id: i64) -> Option<&Mask> {
        self.observable_ids
            .iter()
            .position(|id| *id == observable_id)
            .and_then(|index| self.masks.get(index))
    }
}

pub trait NativeBatchDecoder: Send + Sync {
    fn detector_ids(&self) -> &[i64];

    fn observable_ids(&self) -> &[i64];

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeNoCorrectionDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeNoCorrectionDecoder {
    pub fn new(observable_ids: Vec<i64>) -> Self {
        Self {
            detector_ids: Vec::new(),
            observable_ids,
        }
    }

    pub fn with_detector_ids(detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> Self {
        Self {
            detector_ids,
            observable_ids,
        }
    }
}

impl NativeBatchDecoder for NativeNoCorrectionDecoder {
    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        let words = crate::word_count(detectors.shots);
        CorrectionMaskBatch::new(
            self.observable_ids.clone(),
            vec![Mask::zero(words); self.observable_ids.len()],
            detectors.shots,
        )
    }
}

pub fn logical_residual_loss_mask_native(
    observables: &HashMap<i64, Mask>,
    corrections: &CorrectionMaskBatch,
    observable_ids: &[i64],
    all_mask: &Mask,
) -> Mask {
    let mut ids = observable_ids.to_vec();
    for observable_id in observables.keys() {
        if !ids.contains(observable_id) {
            ids.push(*observable_id);
        }
    }
    for observable_id in &corrections.observable_ids {
        if !ids.contains(observable_id) {
            ids.push(*observable_id);
        }
    }

    let mut loss = Mask::zero(all_mask.words.len());
    let zero = Mask::zero(all_mask.words.len());
    for observable_id in ids {
        let observable = observables.get(&observable_id).unwrap_or(&zero);
        let correction = corrections.get(observable_id).unwrap_or(&zero);
        let mut residual = observable.clone();
        residual.xor_assign(correction);
        loss.or_assign(&residual);
    }
    loss.and_assign(all_mask);
    loss
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedCorrectionDecoder {
        observable_ids: Vec<i64>,
        correction: Mask,
    }

    impl NativeBatchDecoder for FixedCorrectionDecoder {
        fn detector_ids(&self) -> &[i64] {
            &[]
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn decode_batch(
            &self,
            detectors: DetectorMaskBatchView<'_>,
        ) -> NpResult<CorrectionMaskBatch> {
            CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                vec![self.correction.clone()],
                detectors.shots,
            )
        }
    }

    #[test]
    fn fixed_decoder_correction_controls_native_residual_loss() {
        let decoder = FixedCorrectionDecoder {
            observable_ids: vec![0],
            correction: Mask { words: vec![0b0011] },
        };
        let detector_ids = vec![0];
        let detector_masks = vec![Mask { words: vec![0] }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();
        let corrections = decoder.decode_batch(view).unwrap();
        let observables = HashMap::from([(0, Mask { words: vec![0b0110] })]);

        let loss =
            logical_residual_loss_mask_native(&observables, &corrections, &[0], &Mask::all(4));

        assert_eq!(loss, Mask { words: vec![0b0101] });
    }

    #[test]
    fn no_correction_decoder_returns_zero_masks_for_observables() {
        let decoder = NativeNoCorrectionDecoder::new(vec![0, 2]);
        let detector_ids = vec![1];
        let detector_masks = vec![Mask { words: vec![0b1010] }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let corrections = decoder.decode_batch(view).unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 2]);
        assert_eq!(corrections.masks, vec![Mask { words: vec![0] }; 2]);
    }
}
