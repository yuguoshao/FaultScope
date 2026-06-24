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
        let expected_words = crate::word_count(shots);
        for (detector_id, mask) in detector_ids.iter().zip(masks) {
            if mask.words.len() != expected_words {
                return Err(NpError::new(format!(
                    "detector mask for detector id {detector_id} has {} words; expected {expected_words} for {shots} shots",
                    mask.words.len()
                )));
            }
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
        let batch = Self {
            observable_ids,
            masks,
            shots,
        };
        batch.validate_shape()?;
        Ok(batch)
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

    pub fn validate_against(&self, declared_observable_ids: &[i64], shots: usize) -> NpResult<()> {
        if self.shots != shots {
            return Err(NpError::new(format!(
                "correction batch has {} shots; expected {shots}",
                self.shots
            )));
        }
        self.validate_shape()?;
        for observable_id in &self.observable_ids {
            if !declared_observable_ids.contains(observable_id) {
                return Err(NpError::new(format!(
                    "correction mask for undeclared observable id {observable_id}"
                )));
            }
        }
        Ok(())
    }

    fn validate_shape(&self) -> NpResult<()> {
        let expected_words = crate::word_count(self.shots);
        for (observable_id, mask) in self.observable_ids.iter().zip(&self.masks) {
            if mask.words.len() != expected_words {
                return Err(NpError::new(format!(
                    "correction mask for observable id {observable_id} has {} words; expected {expected_words} for {} shots",
                    mask.words.len(),
                    self.shots
                )));
            }
        }
        for (index, observable_id) in self.observable_ids.iter().enumerate() {
            if self.observable_ids[..index].contains(observable_id) {
                return Err(NpError::new(format!(
                    "duplicate correction observable id {observable_id}"
                )));
            }
        }
        Ok(())
    }
}

pub trait NativeBatchDecoder: Send + Sync {
    /// Stable backend name used for lightweight Python introspection.
    fn name(&self) -> &'static str {
        "native"
    }

    /// Detector ids define the only syndrome order passed to `decode_batch`.
    fn detector_ids(&self) -> &[i64];

    /// Observable ids define the allowed correction-mask output ids.
    fn observable_ids(&self) -> &[i64];

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch>;

    fn decode_batch_checked(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> NpResult<CorrectionMaskBatch> {
        let shots = detectors.shots;
        let corrections = self.decode_batch(detectors)?;
        corrections.validate_against(self.observable_ids(), shots)?;
        Ok(corrections)
    }
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
    fn name(&self) -> &'static str {
        "no-correction"
    }

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
            correction: Mask {
                words: vec![0b0011],
            },
        };
        let detector_ids = vec![0];
        let detector_masks = vec![Mask { words: vec![0] }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();
        let corrections = decoder.decode_batch(view).unwrap();
        let observables = HashMap::from([(
            0,
            Mask {
                words: vec![0b0110],
            },
        )]);

        let loss =
            logical_residual_loss_mask_native(&observables, &corrections, &[0], &Mask::all(4));

        assert_eq!(
            loss,
            Mask {
                words: vec![0b0101]
            }
        );
    }

    #[test]
    fn correction_batch_rejects_wrong_word_count() {
        let err =
            CorrectionMaskBatch::new(vec![0], vec![Mask { words: vec![0, 0] }], 4).unwrap_err();

        assert!(err.to_string().contains("expected 1"));
    }

    #[test]
    fn correction_batch_rejects_duplicate_observable_ids() {
        let err = CorrectionMaskBatch::new(
            vec![0, 0],
            vec![Mask { words: vec![0] }, Mask { words: vec![0] }],
            4,
        )
        .unwrap_err();

        assert!(err
            .to_string()
            .contains("duplicate correction observable id 0"));
    }

    #[test]
    fn checked_decode_rejects_unknown_observable_id() {
        struct UnknownObservableDecoder;

        impl NativeBatchDecoder for UnknownObservableDecoder {
            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn decode_batch(
                &self,
                detectors: DetectorMaskBatchView<'_>,
            ) -> NpResult<CorrectionMaskBatch> {
                CorrectionMaskBatch::new(
                    vec![1],
                    vec![Mask::zero(crate::word_count(detectors.shots))],
                    detectors.shots,
                )
            }
        }

        let view = DetectorMaskBatchView::new(&[], &[], 4).unwrap();
        let err = UnknownObservableDecoder
            .decode_batch_checked(view)
            .unwrap_err();

        assert!(err
            .to_string()
            .contains("correction mask for undeclared observable id 1"));
    }

    #[test]
    fn checked_decode_rejects_shots_mismatch() {
        struct MismatchedShotsDecoder;

        impl NativeBatchDecoder for MismatchedShotsDecoder {
            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn decode_batch(
                &self,
                _detectors: DetectorMaskBatchView<'_>,
            ) -> NpResult<CorrectionMaskBatch> {
                CorrectionMaskBatch::new(vec![0], vec![Mask::zero(crate::word_count(5))], 5)
            }
        }

        let view = DetectorMaskBatchView::new(&[], &[], 4).unwrap();
        let err = MismatchedShotsDecoder
            .decode_batch_checked(view)
            .unwrap_err();

        assert!(err.to_string().contains("expected 4"));
    }

    #[test]
    fn missing_correction_observable_is_zero_residual_correction() {
        let corrections = CorrectionMaskBatch::empty(4);
        let observables = HashMap::from([(
            0,
            Mask {
                words: vec![0b1010],
            },
        )]);

        let loss =
            logical_residual_loss_mask_native(&observables, &corrections, &[0], &Mask::all(4));

        assert_eq!(
            loss,
            Mask {
                words: vec![0b1010]
            }
        );
    }

    #[test]
    fn no_correction_decoder_returns_zero_masks_for_observables() {
        let decoder = NativeNoCorrectionDecoder::new(vec![0, 2]);
        let detector_ids = vec![1];
        let detector_masks = vec![Mask {
            words: vec![0b1010],
        }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let corrections = decoder.decode_batch(view).unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 2]);
        assert_eq!(corrections.masks, vec![Mask { words: vec![0] }; 2]);
    }
}
