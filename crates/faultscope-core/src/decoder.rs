use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Arc;

use crate::{GraphlikeDecodingProblem, Mask, NpError, NpResult};

pub const NATIVE_DECODER_PLUGIN_ABI_VERSION: u32 = 1;
pub const NATIVE_DECODER_PLUGIN_ABI_NAME: &str = "faultscope.native_decoder_plugin.v1";
pub const NATIVE_DECODER_PLUGIN_CAPSULE_NAME: &str = "faultscope.native_decoder_plugin.v1";
pub const NATIVE_DECODER_PLUGIN_CAPSULE_METHOD: &str = "__faultscope_native_decoder_capsule__";
pub const NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP: &str = "faultscope.native_decoders";
pub const NATIVE_DECODER_PLUGIN_FLAG_THREAD_SAFE: u64 = 1 << 0;
pub const NATIVE_DECODER_PLUGIN_STATUS_OK: i32 = 0;
pub const NATIVE_DECODER_PLUGIN_STATUS_ERROR: i32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDecoderStringViewV1 {
    pub ptr: *const c_char,
    pub len: usize,
}

impl FaultScopeNativeDecoderStringViewV1 {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDecoderI64SliceV1 {
    pub ptr: *const i64,
    pub len: usize,
}

impl FaultScopeNativeDecoderI64SliceV1 {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDecoderMaskViewV1 {
    pub words: *const u64,
    pub word_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDecoderMaskMutViewV1 {
    pub words: *mut u64,
    pub word_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDetectorMaskBatchViewV1 {
    pub detector_ids: *const i64,
    pub detector_count: usize,
    pub masks: *const FaultScopeNativeDecoderMaskViewV1,
    pub shots: usize,
    pub word_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeCorrectionMaskBatchMutViewV1 {
    pub observable_ids: *const i64,
    pub observable_count: usize,
    pub masks: *mut FaultScopeNativeDecoderMaskMutViewV1,
    pub shots: usize,
    pub word_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativePackedDetectorShotBatchViewV1 {
    pub detector_ids: *const i64,
    pub detector_count: usize,
    pub data: *const u8,
    pub shots: usize,
    pub detector_byte_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDetectorEventShotBatchViewV1 {
    pub detector_ids: *const i64,
    pub detector_count: usize,
    pub offsets: *const usize,
    pub offsets_len: usize,
    pub events: *const usize,
    pub event_count: usize,
    pub shots: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativePackedObservableShotBatchMutViewV1 {
    pub observable_ids: *const i64,
    pub observable_count: usize,
    pub data: *mut u8,
    pub shots: usize,
    pub observable_byte_count: usize,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FaultScopeNativeDecoderStatusV1 {
    pub code: i32,
    pub message: FaultScopeNativeDecoderStringViewV1,
}

impl FaultScopeNativeDecoderStatusV1 {
    pub const fn ok() -> Self {
        Self {
            code: NATIVE_DECODER_PLUGIN_STATUS_OK,
            message: FaultScopeNativeDecoderStringViewV1::empty(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaultScopeNativeDecoderV1 {
    pub abi_version: u32,
    pub struct_size: usize,
    pub flags: u64,
    pub state: *mut c_void,
    pub drop_state: Option<unsafe extern "C" fn(*mut c_void)>,
    pub name: Option<
        unsafe extern "C" fn(
            *const c_void,
            *mut FaultScopeNativeDecoderStringViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub detector_ids: Option<
        unsafe extern "C" fn(
            *const c_void,
            *mut FaultScopeNativeDecoderI64SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub observable_ids: Option<
        unsafe extern "C" fn(
            *const c_void,
            *mut FaultScopeNativeDecoderI64SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub decode_batch: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const FaultScopeNativeDetectorMaskBatchViewV1,
            *mut FaultScopeNativeCorrectionMaskBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub decode_packed_batch: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const FaultScopeNativePackedDetectorShotBatchViewV1,
            *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub decode_detector_event_batch: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const FaultScopeNativeDetectorEventShotBatchViewV1,
            *mut FaultScopeNativePackedObservableShotBatchMutViewV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub create_worker_state: Option<
        unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> FaultScopeNativeDecoderStatusV1,
    >,
}

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

#[derive(Debug, Clone, Copy)]
pub struct PackedDetectorShotBatchView<'a> {
    pub detector_ids: &'a [i64],
    pub data: &'a [u8],
    pub shots: usize,
    pub detector_byte_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct DetectorEventShotBatchView<'a> {
    pub detector_ids: &'a [i64],
    pub offsets: &'a [usize],
    pub events: &'a [usize],
    pub shots: usize,
}

impl<'a> DetectorEventShotBatchView<'a> {
    pub fn new(
        detector_ids: &'a [i64],
        offsets: &'a [usize],
        events: &'a [usize],
        shots: usize,
    ) -> NpResult<Self> {
        if offsets.len() != shots + 1 {
            return Err(NpError::new(format!(
                "detector event batch offsets has length {}; expected {}",
                offsets.len(),
                shots + 1
            )));
        }
        if offsets.first().copied().unwrap_or(0) != 0
            || offsets.last().copied().unwrap_or(0) != events.len()
        {
            return Err(NpError::new(
                "detector event batch offsets must start at 0 and end at events length",
            ));
        }
        for pair in offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(NpError::new(
                    "detector event batch offsets must be nondecreasing",
                ));
            }
        }
        for &event in events {
            if event >= detector_ids.len() {
                return Err(NpError::new(format!(
                    "detector event index {event} exceeds detector count {}",
                    detector_ids.len()
                )));
            }
        }
        Ok(Self {
            detector_ids,
            offsets,
            events,
            shots,
        })
    }
}

impl<'a> PackedDetectorShotBatchView<'a> {
    pub fn new(detector_ids: &'a [i64], data: &'a [u8], shots: usize) -> NpResult<Self> {
        let detector_byte_count = detector_ids.len().div_ceil(8);
        let expected_len = shots.checked_mul(detector_byte_count).ok_or_else(|| {
            NpError::new("packed detector shot batch byte length overflowed usize")
        })?;
        if data.len() != expected_len {
            return Err(NpError::new(format!(
                "packed detector shot batch has {} bytes; expected {expected_len} for {shots} shots and {} detectors",
                data.len(),
                detector_ids.len()
            )));
        }
        Ok(Self {
            detector_ids,
            data,
            shots,
            detector_byte_count,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PackedObservableShotBatch {
    pub observable_ids: Vec<i64>,
    pub data: Vec<u8>,
    pub shots: usize,
    pub observable_byte_count: usize,
}

impl PackedObservableShotBatch {
    pub fn new(observable_ids: Vec<i64>, data: Vec<u8>, shots: usize) -> NpResult<Self> {
        let observable_byte_count = observable_ids.len().div_ceil(8);
        let batch = Self {
            observable_ids,
            data,
            shots,
            observable_byte_count,
        };
        batch.validate_shape()?;
        Ok(batch)
    }

    pub fn zero(observable_ids: Vec<i64>, shots: usize) -> Self {
        let observable_byte_count = observable_ids.len().div_ceil(8);
        Self {
            data: vec![0; shots * observable_byte_count],
            observable_ids,
            shots,
            observable_byte_count,
        }
    }

    pub fn validate_against(&self, declared_observable_ids: &[i64], shots: usize) -> NpResult<()> {
        if self.shots != shots {
            return Err(NpError::new(format!(
                "packed correction batch has {} shots; expected {shots}",
                self.shots
            )));
        }
        self.validate_shape()?;
        for observable_id in &self.observable_ids {
            if !declared_observable_ids.contains(observable_id) {
                return Err(NpError::new(format!(
                    "packed correction for undeclared observable id {observable_id}"
                )));
            }
        }
        Ok(())
    }

    fn validate_shape(&self) -> NpResult<()> {
        let expected_byte_count = self.observable_ids.len().div_ceil(8);
        if self.observable_byte_count != expected_byte_count {
            return Err(NpError::new(format!(
                "packed correction observable byte count is {}; expected {expected_byte_count}",
                self.observable_byte_count
            )));
        }
        let expected_len = self
            .shots
            .checked_mul(self.observable_byte_count)
            .ok_or_else(|| NpError::new("packed correction byte length overflowed usize"))?;
        if self.data.len() != expected_len {
            return Err(NpError::new(format!(
                "packed correction batch has {} bytes; expected {expected_len} for {} shots and {} observables",
                self.data.len(),
                self.shots,
                self.observable_ids.len()
            )));
        }
        for (index, observable_id) in self.observable_ids.iter().enumerate() {
            if self.observable_ids[..index].contains(observable_id) {
                return Err(NpError::new(format!(
                    "duplicate packed correction observable id {observable_id}"
                )));
            }
        }
        Ok(())
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
    fn name(&self) -> &str {
        "native"
    }

    /// Detector ids define the only syndrome order passed to `decode_batch`.
    fn detector_ids(&self) -> &[i64];

    /// Observable ids define the allowed correction-mask output ids.
    fn observable_ids(&self) -> &[i64];

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        Err(NpError::new(format!(
            "{} does not support collection worker instances",
            self.name()
        )))
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch>;

    fn supports_packed_batch(&self) -> bool {
        false
    }

    fn decode_packed_batch(
        &self,
        _detectors: PackedDetectorShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        Err(NpError::new(format!(
            "{} does not support packed-row batch decode",
            self.name()
        )))
    }

    fn supports_detector_event_batch(&self) -> bool {
        false
    }

    fn decode_detector_event_batch(
        &self,
        _detectors: DetectorEventShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        Err(NpError::new(format!(
            "{} does not support detector-event batch decode",
            self.name()
        )))
    }

    fn decode_batch_checked(
        &self,
        detectors: DetectorMaskBatchView<'_>,
    ) -> NpResult<CorrectionMaskBatch> {
        let shots = detectors.shots;
        let corrections = self.decode_batch(detectors)?;
        corrections.validate_against(self.observable_ids(), shots)?;
        Ok(corrections)
    }

    fn decode_packed_batch_checked(
        &self,
        detectors: PackedDetectorShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        let shots = detectors.shots;
        let corrections = self.decode_packed_batch(detectors)?;
        corrections.validate_against(self.observable_ids(), shots)?;
        Ok(corrections)
    }

    fn decode_detector_event_batch_checked(
        &self,
        detectors: DetectorEventShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        let shots = detectors.shots;
        let corrections = self.decode_detector_event_batch(detectors)?;
        corrections.validate_against(self.observable_ids(), shots)?;
        Ok(corrections)
    }
}

pub struct NativeCompositeDecoder {
    children: Vec<Arc<dyn NativeBatchDecoder>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    child_detector_indices: Vec<Vec<usize>>,
}

impl std::fmt::Debug for NativeCompositeDecoder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("NativeCompositeDecoder")
            .field("detector_ids", &self.detector_ids)
            .field("observable_ids", &self.observable_ids)
            .field("child_count", &self.children.len())
            .finish()
    }
}

impl NativeCompositeDecoder {
    pub fn new(children: Vec<Arc<dyn NativeBatchDecoder>>) -> NpResult<Self> {
        if children.is_empty() {
            return Err(NpError::new(
                "native composite decoder requires at least one child decoder",
            ));
        }
        let mut detector_ids = Vec::new();
        let mut detector_indices = HashMap::new();
        let mut observable_ids = Vec::new();
        let mut seen_observables = HashSet::new();
        let mut child_detector_indices = Vec::with_capacity(children.len());
        for child in &children {
            let mut indices = Vec::with_capacity(child.detector_ids().len());
            for &detector_id in child.detector_ids() {
                let index = *detector_indices.entry(detector_id).or_insert_with(|| {
                    let index = detector_ids.len();
                    detector_ids.push(detector_id);
                    index
                });
                indices.push(index);
            }
            child_detector_indices.push(indices);
            for &observable_id in child.observable_ids() {
                if !seen_observables.insert(observable_id) {
                    return Err(NpError::new(format!(
                        "native composite decoder has duplicate observable id {observable_id}"
                    )));
                }
                observable_ids.push(observable_id);
            }
        }
        Ok(Self {
            children,
            detector_ids,
            observable_ids,
            child_detector_indices,
        })
    }

    fn validate_detector_order(&self, detector_ids: &[i64]) -> NpResult<()> {
        if detector_ids != self.detector_ids {
            return Err(NpError::new(
                "native composite decoder received detectors in an unexpected order",
            ));
        }
        Ok(())
    }

    fn merge_packed_child(
        &self,
        output: &mut [u8],
        child: &PackedObservableShotBatch,
    ) -> NpResult<()> {
        let output_byte_count = self.observable_ids.len().div_ceil(8);
        let observable_indices = self
            .observable_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect::<HashMap<_, _>>();
        for shot in 0..child.shots {
            for (child_index, observable_id) in child.observable_ids.iter().enumerate() {
                if child.data[shot * child.observable_byte_count + (child_index >> 3)]
                    & (1 << (child_index & 7))
                    == 0
                {
                    continue;
                }
                let output_index = observable_indices.get(observable_id).ok_or_else(|| {
                    NpError::new(format!(
                        "native composite child returned unknown observable id {observable_id}"
                    ))
                })?;
                output[shot * output_byte_count + (output_index >> 3)] |= 1 << (output_index & 7);
            }
        }
        Ok(())
    }
}

impl NativeBatchDecoder for NativeCompositeDecoder {
    fn name(&self) -> &str {
        "composite"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        let children = self
            .children
            .iter()
            .map(|child| child.create_worker_instance())
            .collect::<NpResult<Vec<_>>>()?;
        Ok(Arc::new(NativeCompositeDecoder::new(children)?))
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        self.validate_detector_order(detectors.detector_ids)?;
        let observable_indices = self
            .observable_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect::<HashMap<_, _>>();
        let mut output_masks =
            vec![Mask::zero(crate::word_count(detectors.shots)); self.observable_ids.len()];
        for (child, indices) in self.children.iter().zip(&self.child_detector_indices) {
            let child_masks = indices
                .iter()
                .map(|&index| detectors.masks[index].clone())
                .collect::<Vec<_>>();
            let view =
                DetectorMaskBatchView::new(child.detector_ids(), &child_masks, detectors.shots)?;
            let corrections = child.decode_batch_checked(view)?;
            for (observable_id, mask) in corrections
                .observable_ids
                .into_iter()
                .zip(corrections.masks)
            {
                let output_index = observable_indices.get(&observable_id).ok_or_else(|| {
                    NpError::new(format!(
                        "native composite child returned unknown observable id {observable_id}"
                    ))
                })?;
                output_masks[*output_index] = mask;
            }
        }
        CorrectionMaskBatch::new(self.observable_ids.clone(), output_masks, detectors.shots)
    }

    fn supports_packed_batch(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.supports_packed_batch())
    }

    fn decode_packed_batch(
        &self,
        detectors: PackedDetectorShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        self.validate_detector_order(detectors.detector_ids)?;
        if !self.supports_packed_batch() {
            return Err(NpError::new(
                "native composite decoder has a child without packed-row support",
            ));
        }
        let output_byte_count = self.observable_ids.len().div_ceil(8);
        let mut output = vec![0u8; detectors.shots * output_byte_count];
        for (child, indices) in self.children.iter().zip(&self.child_detector_indices) {
            let child_byte_count = indices.len().div_ceil(8);
            let mut child_data = vec![0u8; detectors.shots * child_byte_count];
            for shot in 0..detectors.shots {
                for (child_index, &source_index) in indices.iter().enumerate() {
                    if detectors.data[shot * detectors.detector_byte_count + (source_index >> 3)]
                        & (1 << (source_index & 7))
                        != 0
                    {
                        child_data[shot * child_byte_count + (child_index >> 3)] |=
                            1 << (child_index & 7);
                    }
                }
            }
            let view = PackedDetectorShotBatchView::new(
                child.detector_ids(),
                &child_data,
                detectors.shots,
            )?;
            let corrections = child.decode_packed_batch_checked(view)?;
            self.merge_packed_child(&mut output, &corrections)?;
        }
        PackedObservableShotBatch::new(self.observable_ids.clone(), output, detectors.shots)
    }

    fn supports_detector_event_batch(&self) -> bool {
        self.children
            .iter()
            .all(|child| child.supports_detector_event_batch())
    }

    fn decode_detector_event_batch(
        &self,
        detectors: DetectorEventShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        self.validate_detector_order(detectors.detector_ids)?;
        if !self.supports_detector_event_batch() {
            return Err(NpError::new(
                "native composite decoder has a child without detector-event support",
            ));
        }
        let output_byte_count = self.observable_ids.len().div_ceil(8);
        let mut output = vec![0u8; detectors.shots * output_byte_count];
        for (child, indices) in self.children.iter().zip(&self.child_detector_indices) {
            let mut global_to_local = vec![None; self.detector_ids.len()];
            for (local, &global) in indices.iter().enumerate() {
                global_to_local[global] = Some(local);
            }
            let mut child_offsets = Vec::with_capacity(detectors.shots + 1);
            let mut child_events = Vec::new();
            child_offsets.push(0);
            for shot in 0..detectors.shots {
                for &event in
                    &detectors.events[detectors.offsets[shot]..detectors.offsets[shot + 1]]
                {
                    if let Some(local) = global_to_local[event] {
                        child_events.push(local);
                    }
                }
                child_offsets.push(child_events.len());
            }
            let view = DetectorEventShotBatchView::new(
                child.detector_ids(),
                &child_offsets,
                &child_events,
                detectors.shots,
            )?;
            let corrections = child.decode_detector_event_batch_checked(view)?;
            self.merge_packed_child(&mut output, &corrections)?;
        }
        PackedObservableShotBatch::new(self.observable_ids.clone(), output, detectors.shots)
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
    fn name(&self) -> &str {
        "no-correction"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(self.clone()))
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        let words = crate::word_count(detectors.shots);
        CorrectionMaskBatch::new(
            self.observable_ids.clone(),
            vec![Mask::zero(words); self.observable_ids.len()],
            detectors.shots,
        )
    }

    fn supports_packed_batch(&self) -> bool {
        true
    }

    fn decode_packed_batch(
        &self,
        detectors: PackedDetectorShotBatchView<'_>,
    ) -> NpResult<PackedObservableShotBatch> {
        PackedObservableShotBatch::new(
            self.observable_ids.clone(),
            vec![0; detectors.shots * self.observable_ids.len().div_ceil(8)],
            detectors.shots,
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NativeGraphlikeDetectorCopyDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_detector_indices: Vec<Option<usize>>,
}

impl NativeGraphlikeDetectorCopyDecoder {
    pub fn from_graphlike_problem(problem: GraphlikeDecodingProblem) -> NpResult<Self> {
        let mut observable_detector_indices = vec![None; problem.observable_ids.len()];
        for edge in &problem.edges {
            if edge.detectors.len() != 1 || edge.fault_observables.len() != 1 {
                continue;
            }
            let detector_index = edge.detectors[0];
            let observable_index = edge.fault_observables[0];
            if detector_index >= problem.detector_ids.len() {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy edge {} references detector index {} but only {} detectors exist",
                    edge.dem_edge_index,
                    detector_index,
                    problem.detector_ids.len()
                )));
            }
            if observable_index >= problem.observable_ids.len() {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy edge {} references observable index {} but only {} observables exist",
                    edge.dem_edge_index,
                    observable_index,
                    problem.observable_ids.len()
                )));
            }
            if let Some(existing_detector_index) = observable_detector_indices[observable_index] {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy found multiple single-detector candidate edges for observable id {}; detector indices {} and {}",
                    problem.observable_ids[observable_index],
                    existing_detector_index,
                    detector_index
                )));
            }
            observable_detector_indices[observable_index] = Some(detector_index);
        }
        Ok(Self {
            detector_ids: problem.detector_ids,
            observable_ids: problem.observable_ids,
            observable_detector_indices,
        })
    }

    pub fn observable_detector_indices(&self) -> &[Option<usize>] {
        &self.observable_detector_indices
    }
}

impl NativeBatchDecoder for NativeGraphlikeDetectorCopyDecoder {
    fn name(&self) -> &str {
        "graphlike-detector-copy"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(self.clone()))
    }

    fn decode_batch(&self, detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        if detectors.detector_ids != self.detector_ids.as_slice() {
            return Err(NpError::new(
                "graphlike-detector-copy received detector masks in an unexpected order",
            ));
        }
        let words = crate::word_count(detectors.shots);
        let masks = self
            .observable_detector_indices
            .iter()
            .map(|detector_index| match detector_index {
                Some(detector_index) => detectors
                    .masks
                    .get(*detector_index)
                    .cloned()
                    .ok_or_else(|| {
                        NpError::new(format!(
                            "graphlike-detector-copy missing detector mask at index {detector_index}"
                        ))
                    }),
                None => Ok(Mask::zero(words)),
            })
            .collect::<NpResult<Vec<_>>>()?;
        CorrectionMaskBatch::new(self.observable_ids.clone(), masks, detectors.shots)
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
#[derive(Debug, Clone, PartialEq)]
pub struct NativeFusionBlossomDecoder {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    edge_count: usize,
}

#[cfg(feature = "decoder-fusion-blossom")]
impl NativeFusionBlossomDecoder {
    pub fn from_graphlike_problem(problem: GraphlikeDecodingProblem) -> NpResult<Self> {
        Ok(Self {
            detector_ids: problem.detector_ids,
            observable_ids: problem.observable_ids,
            edge_count: problem.edges.len(),
        })
    }

    pub fn edge_count(&self) -> usize {
        self.edge_count
    }

    fn unavailable_error() -> NpError {
        NpError::new(
            "fusion-blossom native backend scaffold is compiled, but no stable Rust dependency is linked yet",
        )
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
impl NativeBatchDecoder for NativeFusionBlossomDecoder {
    fn name(&self) -> &str {
        "fusion-blossom"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
        Ok(Arc::new(self.clone()))
    }

    fn decode_batch(&self, _detectors: DetectorMaskBatchView<'_>) -> NpResult<CorrectionMaskBatch> {
        Err(Self::unavailable_error())
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
    use crate::GraphlikeEdge;
    use std::sync::Arc;

    struct FixedCorrectionDecoder {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        correction: Mask,
    }

    struct FastCopyDecoder {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
    }

    fn graphlike_problem() -> GraphlikeDecodingProblem {
        GraphlikeDecodingProblem {
            detector_ids: vec![10, 20],
            detector_coords: vec![Vec::new(), Vec::new()],
            observable_ids: vec![0, 1],
            edges: vec![
                GraphlikeEdge {
                    detectors: vec![1],
                    fault_observables: vec![0],
                    probability: 0.1,
                    weight: 2.0,
                    dem_edge_index: 0,
                },
                GraphlikeEdge {
                    detectors: vec![0, 1],
                    fault_observables: vec![1],
                    probability: 0.2,
                    weight: 1.0,
                    dem_edge_index: 1,
                },
            ],
        }
    }

    impl NativeBatchDecoder for FixedCorrectionDecoder {
        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
            Ok(Arc::new(FixedCorrectionDecoder {
                detector_ids: self.detector_ids.clone(),
                observable_ids: self.observable_ids.clone(),
                correction: self.correction.clone(),
            }))
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

    impl NativeBatchDecoder for FastCopyDecoder {
        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn create_worker_instance(&self) -> NpResult<Arc<dyn NativeBatchDecoder>> {
            Ok(Arc::new(FastCopyDecoder {
                detector_ids: self.detector_ids.clone(),
                observable_ids: self.observable_ids.clone(),
            }))
        }

        fn decode_batch(
            &self,
            detectors: DetectorMaskBatchView<'_>,
        ) -> NpResult<CorrectionMaskBatch> {
            CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                vec![detectors.masks[0].clone()],
                detectors.shots,
            )
        }

        fn supports_packed_batch(&self) -> bool {
            true
        }

        fn decode_packed_batch(
            &self,
            detectors: PackedDetectorShotBatchView<'_>,
        ) -> NpResult<PackedObservableShotBatch> {
            let data = (0..detectors.shots)
                .map(|shot| detectors.data[shot * detectors.detector_byte_count] & 1)
                .collect();
            PackedObservableShotBatch::new(self.observable_ids.clone(), data, detectors.shots)
        }

        fn supports_detector_event_batch(&self) -> bool {
            true
        }

        fn decode_detector_event_batch(
            &self,
            detectors: DetectorEventShotBatchView<'_>,
        ) -> NpResult<PackedObservableShotBatch> {
            let data = (0..detectors.shots)
                .map(|shot| {
                    (detectors.events[detectors.offsets[shot]..detectors.offsets[shot + 1]]
                        .iter()
                        .filter(|&&event| event == 0)
                        .count()
                        & 1) as u8
                })
                .collect();
            PackedObservableShotBatch::new(self.observable_ids.clone(), data, detectors.shots)
        }
    }

    #[test]
    fn fixed_decoder_correction_controls_native_residual_loss() {
        let decoder = FixedCorrectionDecoder {
            detector_ids: vec![],
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
    fn composite_decoder_builds_stable_unions_and_merges_corrections() {
        let first: Arc<dyn NativeBatchDecoder> = Arc::new(FixedCorrectionDecoder {
            detector_ids: vec![10, 20],
            observable_ids: vec![2],
            correction: Mask {
                words: vec![0b0011],
            },
        });
        let second: Arc<dyn NativeBatchDecoder> = Arc::new(FixedCorrectionDecoder {
            detector_ids: vec![20, 30],
            observable_ids: vec![5],
            correction: Mask {
                words: vec![0b1100],
            },
        });
        let decoder = NativeCompositeDecoder::new(vec![first, second]).unwrap();
        let detector_masks = vec![Mask::zero(1); 3];
        let view = DetectorMaskBatchView::new(decoder.detector_ids(), &detector_masks, 4).unwrap();

        let corrections = decoder.decode_batch_checked(view).unwrap();

        assert_eq!(decoder.detector_ids(), &[10, 20, 30]);
        assert_eq!(decoder.observable_ids(), &[2, 5]);
        assert_eq!(
            corrections.masks,
            vec![
                Mask {
                    words: vec![0b0011]
                },
                Mask {
                    words: vec![0b1100]
                }
            ]
        );
    }

    #[test]
    fn composite_decoder_rejects_empty_and_duplicate_observables() {
        assert!(NativeCompositeDecoder::new(vec![]).is_err());
        let children: Vec<Arc<dyn NativeBatchDecoder>> = vec![
            Arc::new(NativeNoCorrectionDecoder::new(vec![1])),
            Arc::new(NativeNoCorrectionDecoder::new(vec![1])),
        ];
        let err = NativeCompositeDecoder::new(children).unwrap_err();
        assert!(err.to_string().contains("duplicate observable id 1"));
    }

    #[test]
    fn composite_decoder_dispatches_packed_and_event_fast_paths() {
        let children: Vec<Arc<dyn NativeBatchDecoder>> = vec![
            Arc::new(FastCopyDecoder {
                detector_ids: vec![10],
                observable_ids: vec![2],
            }),
            Arc::new(FastCopyDecoder {
                detector_ids: vec![20],
                observable_ids: vec![5],
            }),
        ];
        let decoder = NativeCompositeDecoder::new(children).unwrap();
        let packed_data = [0b01, 0b10, 0b11, 0b00];
        let packed_view =
            PackedDetectorShotBatchView::new(decoder.detector_ids(), &packed_data, 4).unwrap();
        let packed = decoder.decode_packed_batch_checked(packed_view).unwrap();
        assert_eq!(packed.observable_ids, vec![2, 5]);
        assert_eq!(packed.data, vec![0b01, 0b10, 0b11, 0b00]);

        let offsets = [0, 1, 2, 4, 4];
        let events = [0, 1, 0, 1];
        let event_view =
            DetectorEventShotBatchView::new(decoder.detector_ids(), &offsets, &events, 4).unwrap();
        let event = decoder
            .decode_detector_event_batch_checked(event_view)
            .unwrap();
        assert_eq!(event.observable_ids, vec![2, 5]);
        assert_eq!(event.data, vec![0b01, 0b10, 0b11, 0b00]);
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

    #[test]
    fn no_correction_decoder_returns_zero_packed_rows_for_observables() {
        let decoder = NativeNoCorrectionDecoder::with_detector_ids(vec![1, 2], vec![0, 3]);
        let detector_data = vec![0b11, 0b01, 0b10, 0b00];
        let view =
            PackedDetectorShotBatchView::new(decoder.detector_ids(), &detector_data, 4).unwrap();

        let corrections = decoder.decode_packed_batch_checked(view).unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 3]);
        assert_eq!(corrections.shots, 4);
        assert_eq!(corrections.observable_byte_count, 1);
        assert_eq!(corrections.data, vec![0; 4]);
    }

    #[test]
    fn graphlike_detector_copy_constructs_unique_mapping() {
        let decoder =
            NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(graphlike_problem())
                .unwrap();

        assert_eq!(decoder.detector_ids(), &[10, 20]);
        assert_eq!(decoder.observable_ids(), &[0, 1]);
        assert_eq!(decoder.observable_detector_indices(), &[Some(1), None]);
    }

    #[test]
    fn graphlike_detector_copy_rejects_ambiguous_observable_mapping() {
        let mut problem = graphlike_problem();
        problem.edges.push(GraphlikeEdge {
            detectors: vec![0],
            fault_observables: vec![0],
            probability: 0.3,
            weight: 0.8,
            dem_edge_index: 2,
        });

        let err = NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(problem).unwrap_err();

        assert!(err
            .to_string()
            .contains("multiple single-detector candidate edges for observable id 0"));
    }

    #[test]
    fn graphlike_detector_copy_decodes_by_copying_detector_masks() {
        let decoder =
            NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(graphlike_problem())
                .unwrap();
        let detector_masks = vec![
            Mask {
                words: vec![0b0011],
            },
            Mask {
                words: vec![0b1010],
            },
        ];
        let view = DetectorMaskBatchView::new(decoder.detector_ids(), &detector_masks, 4).unwrap();

        let corrections = decoder.decode_batch_checked(view).unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 1]);
        assert_eq!(
            corrections.masks,
            vec![
                Mask {
                    words: vec![0b1010]
                },
                Mask { words: vec![0] }
            ]
        );
    }

    #[test]
    fn graphlike_detector_copy_rejects_wrong_detector_order() {
        let decoder =
            NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(graphlike_problem())
                .unwrap();
        let detector_ids = vec![20, 10];
        let detector_masks = vec![Mask { words: vec![0] }, Mask { words: vec![0] }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let err = decoder.decode_batch(view).unwrap_err();

        assert!(err.to_string().contains("unexpected order"));
    }

    #[cfg(feature = "decoder-fusion-blossom")]
    #[test]
    fn fusion_blossom_scaffold_constructs_from_graphlike_problem() {
        let decoder =
            NativeFusionBlossomDecoder::from_graphlike_problem(graphlike_problem()).unwrap();

        assert_eq!(decoder.name(), "fusion-blossom");
        assert_eq!(decoder.detector_ids(), &[10, 20]);
        assert_eq!(decoder.observable_ids(), &[0, 1]);
        assert_eq!(decoder.edge_count(), 2);
    }

    #[cfg(feature = "decoder-fusion-blossom")]
    #[test]
    fn fusion_blossom_scaffold_reports_unavailable_on_decode() {
        let decoder =
            NativeFusionBlossomDecoder::from_graphlike_problem(graphlike_problem()).unwrap();
        let detector_masks = vec![Mask { words: vec![0] }, Mask { words: vec![0] }];
        let view = DetectorMaskBatchView::new(decoder.detector_ids(), &detector_masks, 4).unwrap();

        let err = decoder.decode_batch_checked(view).unwrap_err();

        assert!(err
            .to_string()
            .contains("no stable Rust dependency is linked yet"));
    }
}
