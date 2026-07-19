use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::Arc;

use crate::{GraphlikeDecodingProblem, Mask, NpError, NpResult};

pub const NATIVE_DECODER_PLUGIN_ABI_VERSION: u32 = 4;
pub const NATIVE_DECODER_PLUGIN_ABI_NAME: &str = "faultscope.native_decoder_plugin.v4";
pub const NATIVE_DECODER_PLUGIN_CAPSULE_NAME: &str = "faultscope.native_decoder_plugin.v4";
pub const NATIVE_DECODER_PLUGIN_CAPSULE_METHOD: &str = "__faultscope_native_decoder_capsule__";
pub const NATIVE_DECODER_PLUGIN_ENTRY_POINT_GROUP: &str = "faultscope.native_decoders";
pub const NATIVE_DECODER_FACTORY_FLAG_THREAD_SAFE: u64 = 1 << 0;
pub const NATIVE_DECODER_PLUGIN_STATUS_OK: i32 = 0;
pub const NATIVE_DECODER_PLUGIN_STATUS_ERROR: i32 = 1;
pub const NATIVE_DECODER_BATCH_FORMAT_MASKS: u32 = 1;
pub const NATIVE_DECODER_BATCH_FORMAT_PACKED: u32 = 2;
pub const NATIVE_DECODER_BATCH_FORMAT_EVENTS: u32 = 3;

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
pub struct FaultScopeNativeDecoderU32SliceV1 {
    pub ptr: *const u32,
    pub len: usize,
}

impl FaultScopeNativeDecoderU32SliceV1 {
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
pub union FaultScopeNativeDetectorBatchPayloadV4 {
    pub masks: FaultScopeNativeDetectorMaskBatchViewV1,
    pub packed: FaultScopeNativePackedDetectorShotBatchViewV1,
    pub events: FaultScopeNativeDetectorEventShotBatchViewV1,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaultScopeNativeDetectorBatchViewV4 {
    pub format: u32,
    pub reserved: u32,
    pub payload: FaultScopeNativeDetectorBatchPayloadV4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub union FaultScopeNativeCorrectionBatchPayloadV4 {
    pub masks: FaultScopeNativeCorrectionMaskBatchMutViewV1,
    pub packed: FaultScopeNativePackedObservableShotBatchMutViewV1,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaultScopeNativeCorrectionBatchMutViewV4 {
    pub format: u32,
    pub reserved: u32,
    pub payload: FaultScopeNativeCorrectionBatchPayloadV4,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaultScopeNativeDecoderFactoryV4 {
    pub abi_version: u32,
    pub struct_size: usize,
    pub flags: u64,
    pub factory_state: *mut c_void,
    pub drop_factory_state: Option<unsafe extern "C" fn(*mut c_void)>,
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
    pub batch_formats: Option<
        unsafe extern "C" fn(
            *const c_void,
            *mut FaultScopeNativeDecoderU32SliceV1,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
    pub create_worker: Option<
        unsafe extern "C" fn(
            *const c_void,
            *mut FaultScopeNativeDecoderWorkerV4,
            usize,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FaultScopeNativeDecoderWorkerV4 {
    pub struct_size: usize,
    pub worker_state: *mut c_void,
    pub drop_worker_state: Option<unsafe extern "C" fn(*mut c_void)>,
    pub decode_batch: Option<
        unsafe extern "C" fn(
            *mut c_void,
            *const FaultScopeNativeDetectorBatchViewV4,
            *mut FaultScopeNativeCorrectionBatchMutViewV4,
        ) -> FaultScopeNativeDecoderStatusV1,
    >,
}

/// Detector syndrome layouts accepted by native decoders, in preference order.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DetectorBatchFormat {
    Masks = NATIVE_DECODER_BATCH_FORMAT_MASKS,
    Packed = NATIVE_DECODER_BATCH_FORMAT_PACKED,
    Events = NATIVE_DECODER_BATCH_FORMAT_EVENTS,
}

impl DetectorBatchFormat {
    pub const STABLE_ORDER: [Self; 3] = [Self::Masks, Self::Packed, Self::Events];

    pub const fn correction_format(self) -> Self {
        match self {
            Self::Masks => Self::Masks,
            Self::Packed | Self::Events => Self::Packed,
        }
    }
}

impl TryFrom<u32> for DetectorBatchFormat {
    type Error = NpError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            NATIVE_DECODER_BATCH_FORMAT_MASKS => Ok(Self::Masks),
            NATIVE_DECODER_BATCH_FORMAT_PACKED => Ok(Self::Packed),
            NATIVE_DECODER_BATCH_FORMAT_EVENTS => Ok(Self::Events),
            _ => Err(NpError::new(format!(
                "unknown native decoder batch format tag {value}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DetectorMaskBatchView<'a> {
    pub detector_ids: &'a [i64],
    pub masks: &'a [Mask],
    pub shots: usize,
}

impl<'a> DetectorMaskBatchView<'a> {
    pub fn new(detector_ids: &'a [i64], masks: &'a [Mask], shots: usize) -> NpResult<Self> {
        validate_decoder_detector_ids(detector_ids)?;
        let view = Self {
            detector_ids,
            masks,
            shots,
        };
        view.validate_shape()?;
        Ok(view)
    }

    fn validate_shape(self) -> NpResult<()> {
        if self.detector_ids.len() != self.masks.len() {
            return Err(NpError::new(
                "detector ids and detector masks must have the same length",
            ));
        }
        let expected_words = crate::word_count(self.shots);
        for (detector_id, mask) in self.detector_ids.iter().zip(self.masks) {
            if mask.words.len() != expected_words {
                return Err(NpError::new(format!(
                    "detector mask for detector id {detector_id} has {} words; expected {expected_words} for {} shots",
                    mask.words.len(),
                    self.shots
                )));
            }
            validate_zero_mask_padding(mask, self.shots, &format!("detector mask {detector_id}"))?;
        }
        Ok(())
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

/// One borrowed detector-syndrome batch tagged with its physical layout.
#[derive(Debug, Clone, Copy)]
pub enum DetectorBatchView<'a> {
    Masks(DetectorMaskBatchView<'a>),
    Packed(PackedDetectorShotBatchView<'a>),
    Events(DetectorEventShotBatchView<'a>),
}

impl<'a> DetectorBatchView<'a> {
    pub const fn format(self) -> DetectorBatchFormat {
        match self {
            Self::Masks(_) => DetectorBatchFormat::Masks,
            Self::Packed(_) => DetectorBatchFormat::Packed,
            Self::Events(_) => DetectorBatchFormat::Events,
        }
    }

    pub const fn detector_ids(self) -> &'a [i64] {
        match self {
            Self::Masks(view) => view.detector_ids,
            Self::Packed(view) => view.detector_ids,
            Self::Events(view) => view.detector_ids,
        }
    }

    pub const fn shots(self) -> usize {
        match self {
            Self::Masks(view) => view.shots,
            Self::Packed(view) => view.shots,
            Self::Events(view) => view.shots,
        }
    }

    pub fn validate(self) -> NpResult<()> {
        validate_decoder_detector_ids(self.detector_ids())?;
        self.validate_shape()
    }

    pub(crate) fn validate_shape(self) -> NpResult<()> {
        match self {
            Self::Masks(view) => view.validate_shape()?,
            Self::Packed(view) => {
                view.validate_shape()?;
            }
            Self::Events(view) => view.validate_shape()?,
        }
        Ok(())
    }
}

impl<'a> From<DetectorMaskBatchView<'a>> for DetectorBatchView<'a> {
    fn from(value: DetectorMaskBatchView<'a>) -> Self {
        Self::Masks(value)
    }
}

impl<'a> From<PackedDetectorShotBatchView<'a>> for DetectorBatchView<'a> {
    fn from(value: PackedDetectorShotBatchView<'a>) -> Self {
        Self::Packed(value)
    }
}

impl<'a> From<DetectorEventShotBatchView<'a>> for DetectorBatchView<'a> {
    fn from(value: DetectorEventShotBatchView<'a>) -> Self {
        Self::Events(value)
    }
}

impl<'a> DetectorEventShotBatchView<'a> {
    pub fn new(
        detector_ids: &'a [i64],
        offsets: &'a [usize],
        events: &'a [usize],
        shots: usize,
    ) -> NpResult<Self> {
        validate_decoder_detector_ids(detector_ids)?;
        let view = Self {
            detector_ids,
            offsets,
            events,
            shots,
        };
        view.validate_shape()?;
        Ok(view)
    }

    fn validate_shape(self) -> NpResult<()> {
        let expected_offsets_len = self
            .shots
            .checked_add(1)
            .ok_or_else(|| NpError::new("detector event batch offset length overflowed usize"))?;
        if self.offsets.len() != expected_offsets_len {
            return Err(NpError::new(format!(
                "detector event batch offsets has length {}; expected {}",
                self.offsets.len(),
                expected_offsets_len
            )));
        }
        if self.offsets.first().copied().unwrap_or(0) != 0
            || self.offsets.last().copied().unwrap_or(0) != self.events.len()
        {
            return Err(NpError::new(
                "detector event batch offsets must start at 0 and end at events length",
            ));
        }
        for pair in self.offsets.windows(2) {
            if pair[0] > pair[1] {
                return Err(NpError::new(
                    "detector event batch offsets must be nondecreasing",
                ));
            }
        }
        for &event in self.events {
            if event >= self.detector_ids.len() {
                return Err(NpError::new(format!(
                    "detector event index {event} exceeds detector count {}",
                    self.detector_ids.len()
                )));
            }
        }
        Ok(())
    }
}

impl<'a> PackedDetectorShotBatchView<'a> {
    pub fn new(detector_ids: &'a [i64], data: &'a [u8], shots: usize) -> NpResult<Self> {
        validate_decoder_detector_ids(detector_ids)?;
        let view = Self {
            detector_ids,
            data,
            shots,
            detector_byte_count: detector_ids.len().div_ceil(8),
        };
        view.validate_shape()?;
        Ok(view)
    }

    fn validate_shape(self) -> NpResult<()> {
        let expected_byte_count = self.detector_ids.len().div_ceil(8);
        if self.detector_byte_count != expected_byte_count {
            return Err(NpError::new(format!(
                "packed detector byte count is {}; expected {expected_byte_count}",
                self.detector_byte_count
            )));
        }
        let expected_len = self.shots.checked_mul(expected_byte_count).ok_or_else(|| {
            NpError::new("packed detector shot batch byte length overflowed usize")
        })?;
        if self.data.len() != expected_len {
            return Err(NpError::new(format!(
                "packed detector shot batch has {} bytes; expected {expected_len} for {} shots and {} detectors",
                self.data.len(),
                self.shots,
                self.detector_ids.len()
            )));
        }
        validate_zero_packed_padding(
            self.data,
            self.shots,
            expected_byte_count,
            self.detector_ids.len(),
            "packed detector batch",
        )?;
        Ok(())
    }
}

/// Owned detector syndrome storage used when a producer and decoder negotiate a layout.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectorBatch {
    Masks {
        detector_ids: Vec<i64>,
        masks: Vec<Mask>,
        shots: usize,
    },
    Packed {
        detector_ids: Vec<i64>,
        data: Vec<u8>,
        shots: usize,
        detector_byte_count: usize,
    },
    Events {
        detector_ids: Vec<i64>,
        offsets: Vec<usize>,
        events: Vec<usize>,
        shots: usize,
    },
}

impl DetectorBatch {
    pub const fn format(&self) -> DetectorBatchFormat {
        match self {
            Self::Masks { .. } => DetectorBatchFormat::Masks,
            Self::Packed { .. } => DetectorBatchFormat::Packed,
            Self::Events { .. } => DetectorBatchFormat::Events,
        }
    }

    pub fn detector_ids(&self) -> &[i64] {
        match self {
            Self::Masks { detector_ids, .. }
            | Self::Packed { detector_ids, .. }
            | Self::Events { detector_ids, .. } => detector_ids,
        }
    }

    pub const fn shots(&self) -> usize {
        match self {
            Self::Masks { shots, .. } | Self::Packed { shots, .. } | Self::Events { shots, .. } => {
                *shots
            }
        }
    }

    pub fn view(&self) -> DetectorBatchView<'_> {
        match self {
            Self::Masks {
                detector_ids,
                masks,
                shots,
            } => DetectorBatchView::Masks(DetectorMaskBatchView {
                detector_ids,
                masks,
                shots: *shots,
            }),
            Self::Packed {
                detector_ids,
                data,
                shots,
                detector_byte_count,
            } => DetectorBatchView::Packed(PackedDetectorShotBatchView {
                detector_ids,
                data,
                shots: *shots,
                detector_byte_count: *detector_byte_count,
            }),
            Self::Events {
                detector_ids,
                offsets,
                events,
                shots,
            } => DetectorBatchView::Events(DetectorEventShotBatchView {
                detector_ids,
                offsets,
                events,
                shots: *shots,
            }),
        }
    }

    pub fn validate(&self) -> NpResult<()> {
        self.view().validate()
    }

    pub(crate) fn validate_shape(&self) -> NpResult<()> {
        self.view().validate_shape()
    }
}

pub(crate) fn empty_detector_event_offsets(shots: usize) -> NpResult<Vec<usize>> {
    let offset_count = shots
        .checked_add(1)
        .ok_or_else(|| NpError::new("detector event offset length overflowed usize"))?;
    let mut offsets = Vec::new();
    offsets.try_reserve_exact(offset_count).map_err(|error| {
        NpError::new(format!(
            "detector event offset storage for {shots} shots could not be allocated: {error}"
        ))
    })?;
    offsets.push(0);
    Ok(offsets)
}

/// Convert a detector syndrome without changing its detector ID order or GF(2) value.
///
/// Duplicate indices in an Events input are combined with XOR. Events output is
/// canonical: each non-zero detector occurs exactly once per shot.
pub fn convert_detector_batch(
    source: DetectorBatchView<'_>,
    target: DetectorBatchFormat,
) -> NpResult<DetectorBatch> {
    convert_detector_batch_impl(source, target, true)
}

fn convert_detector_batch_prevalidated(
    source: DetectorBatchView<'_>,
    target: DetectorBatchFormat,
) -> NpResult<DetectorBatch> {
    convert_detector_batch_impl(source, target, false)
}

fn convert_detector_batch_impl(
    source: DetectorBatchView<'_>,
    target: DetectorBatchFormat,
    validate_metadata: bool,
) -> NpResult<DetectorBatch> {
    if validate_metadata {
        source.validate()?;
    } else {
        source.validate_shape()?;
    }
    let detector_ids = source.detector_ids().to_vec();
    let shots = source.shots();
    let detector_count = detector_ids.len();
    let detector_byte_count = detector_count.div_ceil(8);
    let words = crate::word_count(shots);

    let converted = match (source, target) {
        (DetectorBatchView::Masks(view), DetectorBatchFormat::Masks) => DetectorBatch::Masks {
            detector_ids,
            masks: view.masks.to_vec(),
            shots,
        },
        (DetectorBatchView::Packed(view), DetectorBatchFormat::Packed) => DetectorBatch::Packed {
            detector_ids,
            data: view.data.to_vec(),
            shots,
            detector_byte_count,
        },
        (DetectorBatchView::Events(view), DetectorBatchFormat::Events) => {
            let mut offsets = empty_detector_event_offsets(shots)?;
            let mut events = Vec::new();
            let mut parity = vec![false; detector_count];
            for shot in 0..shots {
                parity.fill(false);
                for &detector in &view.events[view.offsets[shot]..view.offsets[shot + 1]] {
                    parity[detector] ^= true;
                }
                events.extend(
                    parity
                        .iter()
                        .enumerate()
                        .filter_map(|(detector, set)| set.then_some(detector)),
                );
                offsets.push(events.len());
            }
            DetectorBatch::Events {
                detector_ids,
                offsets,
                events,
                shots,
            }
        }
        (DetectorBatchView::Masks(view), DetectorBatchFormat::Packed) => {
            let len = shots.checked_mul(detector_byte_count).ok_or_else(|| {
                NpError::new("packed detector conversion length overflowed usize")
            })?;
            let mut data = vec![0u8; len];
            for (detector, mask) in view.masks.iter().enumerate() {
                for shot in 0..shots {
                    if mask.words[shot >> 6] & (1u64 << (shot & 63)) != 0 {
                        data[shot * detector_byte_count + (detector >> 3)] |= 1u8 << (detector & 7);
                    }
                }
            }
            DetectorBatch::Packed {
                detector_ids,
                data,
                shots,
                detector_byte_count,
            }
        }
        (DetectorBatchView::Packed(view), DetectorBatchFormat::Masks) => {
            let mut masks = vec![Mask::zero(words); detector_count];
            for shot in 0..shots {
                let row = shot * detector_byte_count;
                for (detector, mask) in masks.iter_mut().enumerate() {
                    if view.data[row + (detector >> 3)] & (1u8 << (detector & 7)) != 0 {
                        mask.words[shot >> 6] |= 1u64 << (shot & 63);
                    }
                }
            }
            DetectorBatch::Masks {
                detector_ids,
                masks,
                shots,
            }
        }
        (DetectorBatchView::Events(view), DetectorBatchFormat::Masks) => {
            let mut masks = vec![Mask::zero(words); detector_count];
            for shot in 0..shots {
                for &detector in &view.events[view.offsets[shot]..view.offsets[shot + 1]] {
                    masks[detector].words[shot >> 6] ^= 1u64 << (shot & 63);
                }
            }
            DetectorBatch::Masks {
                detector_ids,
                masks,
                shots,
            }
        }
        (DetectorBatchView::Events(view), DetectorBatchFormat::Packed) => {
            let len = shots.checked_mul(detector_byte_count).ok_or_else(|| {
                NpError::new("packed detector conversion length overflowed usize")
            })?;
            let mut data = vec![0u8; len];
            for shot in 0..shots {
                let row = shot * detector_byte_count;
                for &detector in &view.events[view.offsets[shot]..view.offsets[shot + 1]] {
                    data[row + (detector >> 3)] ^= 1u8 << (detector & 7);
                }
            }
            DetectorBatch::Packed {
                detector_ids,
                data,
                shots,
                detector_byte_count,
            }
        }
        (DetectorBatchView::Masks(view), DetectorBatchFormat::Events) => {
            let mut offsets = empty_detector_event_offsets(shots)?;
            let mut events = Vec::new();
            for shot in 0..shots {
                for (detector, mask) in view.masks.iter().enumerate() {
                    if mask.words[shot >> 6] & (1u64 << (shot & 63)) != 0 {
                        events.push(detector);
                    }
                }
                offsets.push(events.len());
            }
            DetectorBatch::Events {
                detector_ids,
                offsets,
                events,
                shots,
            }
        }
        (DetectorBatchView::Packed(view), DetectorBatchFormat::Events) => {
            let mut offsets = empty_detector_event_offsets(shots)?;
            let mut events = Vec::new();
            for shot in 0..shots {
                let row = shot * detector_byte_count;
                for detector in 0..detector_count {
                    if view.data[row + (detector >> 3)] & (1u8 << (detector & 7)) != 0 {
                        events.push(detector);
                    }
                }
                offsets.push(events.len());
            }
            DetectorBatch::Events {
                detector_ids,
                offsets,
                events,
                shots,
            }
        }
    };
    if validate_metadata {
        converted.validate()?;
    } else {
        converted.validate_shape()?;
    }
    Ok(converted)
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

    pub fn zero(observable_ids: Vec<i64>, shots: usize) -> NpResult<Self> {
        let observable_byte_count = observable_ids.len().div_ceil(8);
        let len = shots
            .checked_mul(observable_byte_count)
            .ok_or_else(|| NpError::new("packed correction byte length overflowed usize"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(len).map_err(|error| {
            NpError::new(format!(
                "packed correction storage for {shots} shots could not be allocated: {error}"
            ))
        })?;
        data.resize(len, 0);
        Self::new(observable_ids, data, shots)
    }

    pub fn validate_against(&self, declared_observable_ids: &[i64], shots: usize) -> NpResult<()> {
        if self.shots != shots {
            return Err(NpError::new(format!(
                "packed correction batch has {} shots; expected {shots}",
                self.shots
            )));
        }
        self.validate_shape()?;
        if self.observable_ids != declared_observable_ids {
            return Err(NpError::new(format!(
                "packed correction observable layout mismatch: expected {:?}, got {:?}",
                declared_observable_ids, self.observable_ids
            )));
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
        validate_zero_packed_padding(
            &self.data,
            self.shots,
            self.observable_byte_count,
            self.observable_ids.len(),
            "packed correction batch",
        )?;
        Ok(())
    }
}

fn validate_zero_packed_padding(
    data: &[u8],
    shots: usize,
    byte_count: usize,
    bit_count: usize,
    batch_name: &str,
) -> NpResult<()> {
    let used_bits = bit_count & 7;
    if used_bits == 0 {
        return Ok(());
    }
    let padding_mask = !((1u8 << used_bits) - 1);
    for shot in 0..shots {
        let final_byte = data[(shot + 1) * byte_count - 1];
        if final_byte & padding_mask != 0 {
            return Err(NpError::new(format!(
                "{batch_name} shot {shot} has non-zero padding bits"
            )));
        }
    }
    Ok(())
}

fn validate_zero_mask_padding(mask: &Mask, shots: usize, name: &str) -> NpResult<()> {
    let used_bits = shots & 63;
    if used_bits == 0 {
        return Ok(());
    }
    let padding_mask = !((1u64 << used_bits) - 1);
    if mask.words.last().copied().unwrap_or(0) & padding_mask != 0 {
        return Err(NpError::new(format!(
            "{name} has non-zero padding bits for {shots} shots"
        )));
    }
    Ok(())
}

/// Count shots whose packed observable bits disagree with packed corrections.
///
/// Both inputs must use the same canonical observable layout and zero their
/// unused padding bits.
pub fn packed_residual_failure_count(
    observable_ids: &[i64],
    observable_data: &[u8],
    observable_byte_count: usize,
    corrections: &PackedObservableShotBatch,
    shots: usize,
) -> NpResult<usize> {
    let expected_observable_byte_count = observable_ids.len().div_ceil(8);
    if observable_byte_count != expected_observable_byte_count {
        return Err(NpError::new(format!(
            "packed observable byte count is {observable_byte_count}; expected {expected_observable_byte_count}"
        )));
    }
    let expected_observable_data_len = shots
        .checked_mul(observable_byte_count)
        .ok_or_else(|| NpError::new("packed observable byte length overflowed usize"))?;
    if observable_data.len() != expected_observable_data_len {
        return Err(NpError::new(format!(
            "packed observable batch has {} bytes; expected {expected_observable_data_len} for {shots} shots and {} observables",
            observable_data.len(),
            observable_ids.len()
        )));
    }
    validate_zero_packed_padding(
        observable_data,
        shots,
        observable_byte_count,
        observable_ids.len(),
        "packed observable batch",
    )?;
    corrections.validate_against(observable_ids, shots)?;

    Ok((0..shots)
        .filter(|shot| {
            let begin = shot * observable_byte_count;
            let end = begin + observable_byte_count;
            observable_data[begin..end]
                .iter()
                .zip(&corrections.data[begin..end])
                .any(|(actual, correction)| (actual ^ correction) != 0)
        })
        .count())
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
        if self.observable_ids != declared_observable_ids {
            return Err(NpError::new(format!(
                "correction observable layout mismatch: expected {:?}, got {:?}",
                declared_observable_ids, self.observable_ids
            )));
        }
        Ok(())
    }

    fn validate_shape(&self) -> NpResult<()> {
        if self.observable_ids.len() != self.masks.len() {
            return Err(NpError::new(
                "observable ids and correction masks must have the same length",
            ));
        }
        let expected_words = crate::word_count(self.shots);
        for (observable_id, mask) in self.observable_ids.iter().zip(&self.masks) {
            if mask.words.len() != expected_words {
                return Err(NpError::new(format!(
                    "correction mask for observable id {observable_id} has {} words; expected {expected_words} for {} shots",
                    mask.words.len(),
                    self.shots
                )));
            }
            validate_zero_mask_padding(
                mask,
                self.shots,
                &format!("correction mask {observable_id}"),
            )?;
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

/// Decoder output tagged with the only two correction layouts allowed by V4.
#[derive(Debug, Clone, PartialEq)]
pub enum DecoderCorrectionBatch {
    Masks(CorrectionMaskBatch),
    Packed(PackedObservableShotBatch),
}

impl DecoderCorrectionBatch {
    pub const fn format(&self) -> DetectorBatchFormat {
        match self {
            Self::Masks(_) => DetectorBatchFormat::Masks,
            Self::Packed(_) => DetectorBatchFormat::Packed,
        }
    }

    pub const fn shots(&self) -> usize {
        match self {
            Self::Masks(batch) => batch.shots,
            Self::Packed(batch) => batch.shots,
        }
    }

    pub fn validate_against(&self, observable_ids: &[i64], shots: usize) -> NpResult<()> {
        match self {
            Self::Masks(batch) => batch.validate_against(observable_ids, shots),
            Self::Packed(batch) => batch.validate_against(observable_ids, shots),
        }
    }

    pub fn into_masks(self) -> NpResult<CorrectionMaskBatch> {
        match self {
            Self::Masks(batch) => Ok(batch),
            Self::Packed(batch) => packed_corrections_to_masks(batch),
        }
    }

    pub fn into_packed(self) -> NpResult<PackedObservableShotBatch> {
        match self {
            Self::Masks(batch) => mask_corrections_to_packed(batch),
            Self::Packed(batch) => Ok(batch),
        }
    }
}

pub fn mask_corrections_to_packed(
    batch: CorrectionMaskBatch,
) -> NpResult<PackedObservableShotBatch> {
    batch.validate_shape()?;
    let observable_byte_count = batch.observable_ids.len().div_ceil(8);
    let len = batch
        .shots
        .checked_mul(observable_byte_count)
        .ok_or_else(|| NpError::new("packed correction conversion length overflowed usize"))?;
    let mut data = vec![0u8; len];
    for (observable, mask) in batch.masks.iter().enumerate() {
        for shot in 0..batch.shots {
            if mask.words[shot >> 6] & (1u64 << (shot & 63)) != 0 {
                data[shot * observable_byte_count + (observable >> 3)] |= 1u8 << (observable & 7);
            }
        }
    }
    PackedObservableShotBatch::new(batch.observable_ids, data, batch.shots)
}

pub fn packed_corrections_to_masks(
    batch: PackedObservableShotBatch,
) -> NpResult<CorrectionMaskBatch> {
    batch.validate_shape()?;
    let mut masks = vec![Mask::zero(crate::word_count(batch.shots)); batch.observable_ids.len()];
    for shot in 0..batch.shots {
        let row = shot * batch.observable_byte_count;
        for (observable, mask) in masks.iter_mut().enumerate() {
            if batch.data[row + (observable >> 3)] & (1u8 << (observable & 7)) != 0 {
                mask.words[shot >> 6] |= 1u64 << (shot & 63);
            }
        }
    }
    CorrectionMaskBatch::new(batch.observable_ids, masks, batch.shots)
}

pub trait NativeDecoderFactory: Send + Sync {
    fn name(&self) -> &str;
    /// Ordered detector input layout. Detector ids must be unique.
    fn detector_ids(&self) -> &[i64];
    fn observable_ids(&self) -> &[i64];
    /// Stable, non-empty, duplicate-free format preference list.
    fn batch_formats(&self) -> &[DetectorBatchFormat];
    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>>;
}

pub trait NativeDecoderWorker: Send {
    fn name(&self) -> &str;
    /// Ordered detector input layout. Detector ids must be unique.
    fn detector_ids(&self) -> &[i64];
    fn observable_ids(&self) -> &[i64];
    /// Stable, non-empty, duplicate-free format preference list.
    fn batch_formats(&self) -> &[DetectorBatchFormat];
    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch>;

    fn decode_batch_checked(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        validate_decoder_batch_formats(self.batch_formats())?;
        detectors.validate_shape()?;
        if detectors.detector_ids() != self.detector_ids() {
            return Err(NpError::new(format!(
                "decoder `{}` received detector ids {:?}; expected {:?}",
                self.name(),
                detectors.detector_ids(),
                self.detector_ids()
            )));
        }
        if !self.batch_formats().contains(&detectors.format()) {
            return Err(NpError::new(format!(
                "decoder `{}` does not declare support for {:?} detector batches",
                self.name(),
                detectors.format()
            )));
        }
        let shots = detectors.shots();
        let expected_format = detectors.format().correction_format();
        let corrections = self.decode_batch(detectors)?;
        if corrections.format() != expected_format {
            return Err(NpError::new(format!(
                "decoder `{}` returned {:?} corrections for {:?} detector input; expected {:?} corrections",
                self.name(),
                corrections.format(),
                detectors.format(),
                expected_format
            )));
        }
        corrections.validate_against(self.observable_ids(), shots)?;
        Ok(corrections)
    }
}

/// Validate a decoder's ordered format preference metadata.
pub fn validate_decoder_batch_formats(formats: &[DetectorBatchFormat]) -> NpResult<()> {
    if formats.is_empty() {
        return Err(NpError::new(
            "native decoder batch format preference list must not be empty",
        ));
    }
    let mut seen = [false; 3];
    for &format in formats {
        let index = match format {
            DetectorBatchFormat::Masks => 0,
            DetectorBatchFormat::Packed => 1,
            DetectorBatchFormat::Events => 2,
        };
        if std::mem::replace(&mut seen[index], true) {
            return Err(NpError::new(format!(
                "native decoder batch format preference list contains duplicate format {format:?}"
            )));
        }
    }
    Ok(())
}

/// Select the first decoder preference a producer can emit without conversion.
/// If there is no direct intersection, the decoder's first preference wins.
pub fn select_detector_batch_format(
    decoder_formats: &[DetectorBatchFormat],
    producer_formats: &[DetectorBatchFormat],
) -> NpResult<DetectorBatchFormat> {
    validate_decoder_batch_formats(decoder_formats)?;
    if producer_formats.is_empty() {
        return Err(NpError::new(
            "detector batch producer format list must not be empty",
        ));
    }
    Ok(decoder_formats
        .iter()
        .copied()
        .find(|format| producer_formats.contains(format))
        .unwrap_or(decoder_formats[0]))
}

const MASKS_BATCH_FORMATS: [DetectorBatchFormat; 1] = [DetectorBatchFormat::Masks];
#[cfg(feature = "decoder-fusion-blossom")]
const PACKED_BATCH_FORMATS: [DetectorBatchFormat; 1] = [DetectorBatchFormat::Packed];
const ALL_BATCH_FORMATS: [DetectorBatchFormat; 3] = DetectorBatchFormat::STABLE_ORDER;

/// Validate the ordered detector input layout exposed by a decoder.
pub fn validate_decoder_detector_ids(detector_ids: &[i64]) -> NpResult<()> {
    let mut seen = HashSet::with_capacity(detector_ids.len());
    for &detector_id in detector_ids {
        if !seen.insert(detector_id) {
            return Err(NpError::new(format!(
                "decoder detector ids must be unique; duplicate detector id {detector_id}"
            )));
        }
    }
    Ok(())
}

pub struct NativeCompositeDecoder {
    children: Vec<Arc<dyn NativeDecoderFactory>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_indices: HashMap<i64, usize>,
    child_plans: Vec<CompositeChildPlan>,
    batch_formats: Vec<DetectorBatchFormat>,
}

#[derive(Clone)]
struct CompositeChildPlan {
    detector_indices: Vec<usize>,
    detector_ids: Vec<i64>,
    global_to_local: Vec<Option<usize>>,
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
    pub fn new(children: Vec<Arc<dyn NativeDecoderFactory>>) -> NpResult<Self> {
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
        for (child_index, child) in children.iter().enumerate() {
            validate_decoder_batch_formats(child.batch_formats()).map_err(|err| {
                NpError::new(format!(
                    "native composite decoder child {child_index} (`{}`) has invalid batch format metadata: {}",
                    child.name(),
                    err.message()
                ))
            })?;
            validate_decoder_detector_ids(child.detector_ids()).map_err(|err| {
                NpError::new(format!(
                    "native composite decoder child {child_index} (`{}`) has invalid detector metadata: {}",
                    child.name(),
                    err.message()
                ))
            })?;
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
        let observable_indices = observable_ids
            .iter()
            .copied()
            .enumerate()
            .map(|(index, id)| (id, index))
            .collect();
        let child_plans = child_detector_indices
            .into_iter()
            .map(|detector_indices| {
                let mut mapping = vec![None; detector_ids.len()];
                for (local, &global) in detector_indices.iter().enumerate() {
                    mapping[global] = Some(local);
                }
                let child_detector_ids = detector_indices
                    .iter()
                    .map(|&index| detector_ids[index])
                    .collect();
                CompositeChildPlan {
                    detector_indices,
                    detector_ids: child_detector_ids,
                    global_to_local: mapping,
                }
            })
            .collect();
        let mut batch_formats = DetectorBatchFormat::STABLE_ORDER.to_vec();
        batch_formats.sort_by_key(|format| {
            let conversion_count = children
                .iter()
                .filter(|child| !child.batch_formats().contains(format))
                .count();
            let preference_rank = children
                .iter()
                .map(|child| {
                    child
                        .batch_formats()
                        .iter()
                        .position(|candidate| candidate == format)
                        .unwrap_or(0)
                })
                .sum::<usize>();
            let stable_rank = DetectorBatchFormat::STABLE_ORDER
                .iter()
                .position(|candidate| candidate == format)
                .expect("candidate is in the stable format order");
            (conversion_count, preference_rank, stable_rank)
        });
        Ok(Self {
            children,
            detector_ids,
            observable_ids,
            observable_indices,
            child_plans,
            batch_formats,
        })
    }
}

impl NativeDecoderFactory for NativeCompositeDecoder {
    fn name(&self) -> &str {
        "composite"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &self.batch_formats
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        let children = self
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| {
                let worker = child.create_worker().map_err(|err| {
                    NpError::new(format!(
                        "composite decoder `{}` child {index} (`{}`) worker factory failed: {}",
                        self.name(),
                        child.name(),
                        err.message()
                    ))
                })?;
                if worker.name() != child.name() {
                    return Err(NpError::new(format!(
                        "composite decoder `{}` child {index} (`{}`) worker name `{}` does not match factory name",
                        self.name(),
                        child.name(),
                        worker.name()
                    )));
                }
                if worker.detector_ids() != child.detector_ids() {
                    return Err(NpError::new(format!(
                        "composite decoder `{}` child {index} (`{}`) worker detector ids {:?} do not match factory detector ids {:?}",
                        self.name(),
                        child.name(),
                        worker.detector_ids(),
                        child.detector_ids()
                    )));
                }
                if worker.observable_ids() != child.observable_ids() {
                    return Err(NpError::new(format!(
                        "composite decoder `{}` child {index} (`{}`) worker observable ids {:?} do not match factory observable ids {:?}",
                        self.name(),
                        child.name(),
                        worker.observable_ids(),
                        child.observable_ids()
                    )));
                }
                validate_decoder_batch_formats(worker.batch_formats())?;
                if worker.batch_formats() != child.batch_formats() {
                    return Err(NpError::new(format!(
                        "composite decoder `{}` child {index} (`{}`) worker batch formats {:?} do not match factory batch formats {:?}",
                        self.name(),
                        child.name(),
                        worker.batch_formats(),
                        child.batch_formats()
                    )));
                }
                Ok(worker)
            })
            .collect::<NpResult<Vec<_>>>()?;
        Ok(Box::new(NativeCompositeDecoderWorker {
            children,
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            observable_indices: self.observable_indices.clone(),
            child_plans: self.child_plans.clone(),
            batch_formats: self.batch_formats.clone(),
        }))
    }
}

struct NativeCompositeDecoderWorker {
    children: Vec<Box<dyn NativeDecoderWorker>>,
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_indices: HashMap<i64, usize>,
    child_plans: Vec<CompositeChildPlan>,
    batch_formats: Vec<DetectorBatchFormat>,
}

impl NativeCompositeDecoderWorker {
    fn validate_detector_order(&self, detector_ids: &[i64]) -> NpResult<()> {
        if detector_ids != self.detector_ids {
            return Err(NpError::new(
                "native composite decoder received detectors in an unexpected order",
            ));
        }
        Ok(())
    }

    fn merge_packed_child(
        observable_ids: &[i64],
        observable_indices: &HashMap<i64, usize>,
        output: &mut [u8],
        child: &PackedObservableShotBatch,
    ) -> NpResult<()> {
        let output_byte_count = observable_ids.len().div_ceil(8);
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

    fn project_batch(
        source: DetectorBatchView<'_>,
        plan: &CompositeChildPlan,
    ) -> NpResult<DetectorBatch> {
        match source {
            DetectorBatchView::Masks(view) => DetectorBatch::Masks {
                detector_ids: plan.detector_ids.clone(),
                masks: plan
                    .detector_indices
                    .iter()
                    .map(|index| view.masks[*index].clone())
                    .collect(),
                shots: view.shots,
            }
            .validated_shape(),
            DetectorBatchView::Packed(view) => {
                let child_byte_count = plan.detector_indices.len().div_ceil(8);
                let len = view.shots.checked_mul(child_byte_count).ok_or_else(|| {
                    NpError::new("composite packed detector projection length overflowed usize")
                })?;
                let mut data = vec![0u8; len];
                for shot in 0..view.shots {
                    for (child_detector, source_detector) in
                        plan.detector_indices.iter().copied().enumerate()
                    {
                        if view.data[shot * view.detector_byte_count + (source_detector >> 3)]
                            & (1u8 << (source_detector & 7))
                            != 0
                        {
                            data[shot * child_byte_count + (child_detector >> 3)] |=
                                1u8 << (child_detector & 7);
                        }
                    }
                }
                DetectorBatch::Packed {
                    detector_ids: plan.detector_ids.clone(),
                    data,
                    shots: view.shots,
                    detector_byte_count: child_byte_count,
                }
                .validated_shape()
            }
            DetectorBatchView::Events(view) => {
                let mut offsets = empty_detector_event_offsets(view.shots)?;
                let mut events = Vec::new();
                for shot in 0..view.shots {
                    for source_detector in &view.events[view.offsets[shot]..view.offsets[shot + 1]]
                    {
                        if let Some(child_detector) = plan.global_to_local[*source_detector] {
                            events.push(child_detector);
                        }
                    }
                    offsets.push(events.len());
                }
                DetectorBatch::Events {
                    detector_ids: plan.detector_ids.clone(),
                    offsets,
                    events,
                    shots: view.shots,
                }
                .validated_shape()
            }
        }
    }
}

trait ValidatedDetectorBatch {
    fn validated_shape(self) -> NpResult<DetectorBatch>;
}

impl ValidatedDetectorBatch for DetectorBatch {
    fn validated_shape(self) -> NpResult<DetectorBatch> {
        self.validate_shape()?;
        Ok(self)
    }
}

impl NativeDecoderWorker for NativeCompositeDecoderWorker {
    fn name(&self) -> &str {
        "composite"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &self.batch_formats
    }

    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        self.validate_detector_order(detectors.detector_ids())?;
        detectors.validate_shape()?;
        let input_format = detectors.format();
        let shots = detectors.shots();

        let child_targets = self
            .children
            .iter()
            .map(|child| {
                if child.batch_formats().contains(&input_format) {
                    input_format
                } else {
                    child.batch_formats()[0]
                }
            })
            .collect::<Vec<_>>();
        let mut shared = Vec::<(DetectorBatchFormat, DetectorBatch)>::new();
        for target in child_targets.iter().copied() {
            if !shared.iter().any(|(format, _)| *format == target) {
                shared.push((
                    target,
                    convert_detector_batch_prevalidated(detectors, target)?,
                ));
            }
        }

        let mut mask_output = (input_format == DetectorBatchFormat::Masks)
            .then(|| vec![Mask::zero(crate::word_count(shots)); self.observable_ids.len()]);
        let mut packed_output = (input_format != DetectorBatchFormat::Masks)
            .then(|| PackedObservableShotBatch::zero(self.observable_ids.clone(), shots))
            .transpose()?;

        for (((child, plan), target), child_index) in self
            .children
            .iter_mut()
            .zip(&self.child_plans)
            .zip(child_targets)
            .zip(0usize..)
        {
            let source = shared
                .iter()
                .find_map(|(format, batch)| (*format == target).then(|| batch.view()))
                .expect("shared conversion exists for every child target");
            let child_batch = Self::project_batch(source, plan)?;
            let corrections = child
                .decode_batch_checked(child_batch.view())
                .map_err(|err| {
                    NpError::new(format!(
                        "composite decoder child {child_index} (`{}`) failed: {}",
                        child.name(),
                        err.message()
                    ))
                })?;
            if let Some(output) = &mut mask_output {
                let corrections = corrections.into_masks()?;
                for (observable_id, mask) in corrections
                    .observable_ids
                    .into_iter()
                    .zip(corrections.masks)
                {
                    let output_index =
                        self.observable_indices.get(&observable_id).ok_or_else(|| {
                            NpError::new(format!(
                            "native composite child returned unknown observable id {observable_id}"
                        ))
                        })?;
                    output[*output_index] = mask;
                }
            } else if let Some(output) = &mut packed_output {
                let corrections = corrections.into_packed()?;
                Self::merge_packed_child(
                    &self.observable_ids,
                    &self.observable_indices,
                    &mut output.data,
                    &corrections,
                )?;
            }
        }

        if let Some(masks) = mask_output {
            Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                masks,
                shots,
            )?))
        } else {
            Ok(DecoderCorrectionBatch::Packed(
                packed_output.expect("non-mask inputs allocate packed output"),
            ))
        }
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

    pub fn with_detector_ids(detector_ids: Vec<i64>, observable_ids: Vec<i64>) -> NpResult<Self> {
        validate_decoder_detector_ids(&detector_ids)?;
        Ok(Self {
            detector_ids,
            observable_ids,
        })
    }
}

impl NativeDecoderFactory for NativeNoCorrectionDecoder {
    fn name(&self) -> &str {
        "no-correction"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &ALL_BATCH_FORMATS
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        Ok(Box::new(NativeNoCorrectionDecoderWorker {
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
    }
}

#[derive(Debug)]
struct NativeNoCorrectionDecoderWorker {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

impl NativeDecoderWorker for NativeNoCorrectionDecoderWorker {
    fn name(&self) -> &str {
        "no-correction"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &ALL_BATCH_FORMATS
    }

    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        match detectors {
            DetectorBatchView::Masks(view) => {
                Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                    self.observable_ids.clone(),
                    vec![Mask::zero(crate::word_count(view.shots)); self.observable_ids.len()],
                    view.shots,
                )?))
            }
            DetectorBatchView::Packed(view) => Ok(DecoderCorrectionBatch::Packed(
                PackedObservableShotBatch::zero(self.observable_ids.clone(), view.shots)?,
            )),
            DetectorBatchView::Events(view) => Ok(DecoderCorrectionBatch::Packed(
                PackedObservableShotBatch::zero(self.observable_ids.clone(), view.shots)?,
            )),
        }
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
        let detector_count = problem.detector_ids().len();
        let observable_count = problem.observable_ids().len();
        let mut observable_detector_indices = vec![None; observable_count];
        for edge in problem.iter_edges() {
            if edge.detector_count() != 1 || edge.fault_observables().len() != 1 {
                continue;
            }
            let detector_index = edge
                .detectors()
                .next()
                .expect("single-detector edge has one detector");
            let observable_index = edge
                .fault_observables()
                .next()
                .expect("single-observable edge has one observable");
            if detector_index >= detector_count {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy edge {} references detector index {} but only {} detectors exist",
                    edge.dem_edge_index(),
                    detector_index,
                    detector_count
                )));
            }
            if observable_index >= observable_count {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy edge {} references observable index {} but only {} observables exist",
                    edge.dem_edge_index(),
                    observable_index,
                    observable_count
                )));
            }
            if let Some(existing_detector_index) = observable_detector_indices[observable_index] {
                return Err(NpError::new(format!(
                    "graphlike-detector-copy found multiple single-detector candidate edges for observable id {}; detector indices {} and {}",
                    problem.observable_ids()[observable_index],
                    existing_detector_index,
                    detector_index
                )));
            }
            observable_detector_indices[observable_index] = Some(detector_index);
        }
        let (detector_ids, observable_ids) = problem.into_ids();
        Ok(Self {
            detector_ids,
            observable_ids,
            observable_detector_indices,
        })
    }

    pub fn observable_detector_indices(&self) -> &[Option<usize>] {
        &self.observable_detector_indices
    }
}

impl NativeDecoderFactory for NativeGraphlikeDetectorCopyDecoder {
    fn name(&self) -> &str {
        "graphlike-detector-copy"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &MASKS_BATCH_FORMATS
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        Ok(Box::new(NativeGraphlikeDetectorCopyDecoderWorker {
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
            observable_detector_indices: self.observable_detector_indices.clone(),
        }))
    }
}

#[derive(Debug)]
struct NativeGraphlikeDetectorCopyDecoderWorker {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
    observable_detector_indices: Vec<Option<usize>>,
}

impl NativeDecoderWorker for NativeGraphlikeDetectorCopyDecoderWorker {
    fn name(&self) -> &str {
        "graphlike-detector-copy"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &MASKS_BATCH_FORMATS
    }

    fn decode_batch(
        &mut self,
        detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        let DetectorBatchView::Masks(detectors) = detectors else {
            return Err(NpError::new(
                "graphlike-detector-copy only accepts Masks detector batches",
            ));
        };
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
        Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
            self.observable_ids.clone(),
            masks,
            detectors.shots,
        )?))
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
        let edge_count = problem.edge_count();
        let (detector_ids, observable_ids) = problem.into_ids();
        Ok(Self {
            detector_ids,
            observable_ids,
            edge_count,
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
impl NativeDecoderFactory for NativeFusionBlossomDecoder {
    fn name(&self) -> &str {
        "fusion-blossom"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &PACKED_BATCH_FORMATS
    }

    fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
        Ok(Box::new(NativeFusionBlossomDecoderWorker {
            detector_ids: self.detector_ids.clone(),
            observable_ids: self.observable_ids.clone(),
        }))
    }
}

#[cfg(feature = "decoder-fusion-blossom")]
#[derive(Debug)]
struct NativeFusionBlossomDecoderWorker {
    detector_ids: Vec<i64>,
    observable_ids: Vec<i64>,
}

#[cfg(feature = "decoder-fusion-blossom")]
impl NativeDecoderWorker for NativeFusionBlossomDecoderWorker {
    fn name(&self) -> &str {
        "fusion-blossom"
    }

    fn detector_ids(&self) -> &[i64] {
        &self.detector_ids
    }

    fn observable_ids(&self) -> &[i64] {
        &self.observable_ids
    }

    fn batch_formats(&self) -> &[DetectorBatchFormat] {
        &PACKED_BATCH_FORMATS
    }

    fn decode_batch(
        &mut self,
        _detectors: DetectorBatchView<'_>,
    ) -> NpResult<DecoderCorrectionBatch> {
        Err(NativeFusionBlossomDecoder::unavailable_error())
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

    struct FixedCorrectionDecoderWorker {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        correction: Mask,
    }

    struct FastCopyDecoder {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
    }

    struct FastCopyDecoderWorker {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
    }

    struct SingleFormatCopyDecoder {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        formats: Vec<DetectorBatchFormat>,
    }

    struct SingleFormatCopyDecoderWorker {
        detector_ids: Vec<i64>,
        observable_ids: Vec<i64>,
        formats: Vec<DetectorBatchFormat>,
    }

    fn graphlike_edges() -> Vec<GraphlikeEdge> {
        vec![
            GraphlikeEdge {
                detectors: vec![1],
                fault_observables: vec![0],
                probability: 0.1,
                dem_edge_index: 0,
            },
            GraphlikeEdge {
                detectors: vec![0, 1],
                fault_observables: vec![1],
                probability: 0.2,
                dem_edge_index: 1,
            },
        ]
    }

    fn graphlike_problem() -> GraphlikeDecodingProblem {
        GraphlikeDecodingProblem::new(
            vec![10, 20],
            vec![Vec::new(), Vec::new()],
            vec![0, 1],
            graphlike_edges(),
        )
        .unwrap()
    }

    #[test]
    fn packed_residual_count_uses_the_canonical_layout_path() {
        let empty_corrections = PackedObservableShotBatch::zero(vec![], 3).unwrap();
        assert_eq!(
            packed_residual_failure_count(&[], &[], 0, &empty_corrections, 3).unwrap(),
            0
        );

        let corrections =
            PackedObservableShotBatch::new(vec![10, 20], vec![0b00, 0b01, 0b00, 0b01], 4).unwrap();

        let failures =
            packed_residual_failure_count(&[10, 20], &[0b00, 0b01, 0b10, 0b11], 1, &corrections, 4)
                .unwrap();

        assert_eq!(failures, 2);
    }

    #[test]
    fn packed_residual_count_rejects_noncanonical_layouts() {
        let reordered =
            PackedObservableShotBatch::new(vec![20, 10], vec![0b10, 0b01, 0b11], 3).unwrap();
        let err = packed_residual_failure_count(&[10, 20], &[0b01, 0b10, 0b11], 1, &reordered, 3)
            .unwrap_err();
        assert!(err.to_string().contains("observable layout mismatch"));

        for ids in [vec![10], vec![10, 20, 30], vec![20, 30]] {
            let corrections = PackedObservableShotBatch::zero(ids, 1).unwrap();
            let err =
                packed_residual_failure_count(&[10, 20], &[0], 1, &corrections, 1).unwrap_err();
            assert!(err.to_string().contains("observable layout mismatch"));
        }
    }

    #[test]
    fn packed_observable_batches_require_zero_padding() {
        for (ids, data) in [
            (vec![0], vec![0b1000_0000]),
            ((0..7).collect(), vec![0b1000_0000]),
            ((0..9).collect(), vec![0, 0b0000_0010]),
        ] {
            let err = PackedObservableShotBatch::new(ids, data, 1).unwrap_err();
            assert!(err.to_string().contains("non-zero padding bits"));
        }

        assert!(PackedObservableShotBatch::new((0..8).collect(), vec![0xff], 1).is_ok());
        assert!(
            PackedObservableShotBatch::new((0..9).collect(), vec![0xff, 0b0000_0001], 1,).is_ok()
        );

        let corrections = PackedObservableShotBatch::zero(vec![0], 1).unwrap();
        let err =
            packed_residual_failure_count(&[0], &[0b1000_0000], 1, &corrections, 1).unwrap_err();
        assert!(err
            .to_string()
            .contains("packed observable batch shot 0 has non-zero padding bits"));
    }

    #[test]
    fn packed_residual_count_rejects_malformed_rows() {
        let corrections = PackedObservableShotBatch::zero(vec![10], 1).unwrap();

        let err = packed_residual_failure_count(&[10], &[], 0, &corrections, 1).unwrap_err();

        assert!(err.to_string().contains("observable byte count"));
    }

    fn patterned_detector_masks(detector_count: usize, shots: usize) -> Vec<Mask> {
        let mut masks = vec![Mask::zero(crate::word_count(shots)); detector_count];
        for (detector, mask) in masks.iter_mut().enumerate() {
            for shot in 0..shots {
                if (shot * 5 + detector * 3 + 1) % 7 < 3 {
                    mask.words[shot >> 6] |= 1u64 << (shot & 63);
                }
            }
        }
        masks
    }

    #[test]
    fn detector_formats_round_trip_across_byte_and_word_boundaries() {
        let detector_ids = (100..109).collect::<Vec<_>>();
        for shots in [0, 1, 7, 8, 63, 64, 65, 129] {
            let expected = patterned_detector_masks(detector_ids.len(), shots);
            let source = DetectorMaskBatchView::new(&detector_ids, &expected, shots).unwrap();
            for first_format in DetectorBatchFormat::STABLE_ORDER {
                let first = convert_detector_batch(source.into(), first_format).unwrap();
                assert_eq!(first.format(), first_format);
                for second_format in DetectorBatchFormat::STABLE_ORDER {
                    let second = convert_detector_batch(first.view(), second_format).unwrap();
                    let round_trip =
                        convert_detector_batch(second.view(), DetectorBatchFormat::Masks).unwrap();
                    let DetectorBatch::Masks {
                        detector_ids: actual_ids,
                        masks: actual_masks,
                        shots: actual_shots,
                    } = round_trip
                    else {
                        unreachable!()
                    };
                    assert_eq!(actual_ids, detector_ids);
                    assert_eq!(actual_masks, expected);
                    assert_eq!(actual_shots, shots);
                }
            }
        }
    }

    #[test]
    fn detector_events_combine_duplicates_by_gf2_xor_and_canonicalize_output() {
        let detector_ids = [10, 20, 30];
        let offsets = [0, 4, 7, 7];
        let events = [0, 0, 1, 2, 1, 1, 1];
        let source = DetectorEventShotBatchView::new(&detector_ids, &offsets, &events, 3).unwrap();

        let masks = convert_detector_batch(source.into(), DetectorBatchFormat::Masks).unwrap();
        let DetectorBatch::Masks { masks, .. } = masks else {
            unreachable!()
        };
        assert_eq!(masks[0].words, [0]);
        assert_eq!(masks[1].words, [0b011]);
        assert_eq!(masks[2].words, [0b001]);

        let packed = convert_detector_batch(source.into(), DetectorBatchFormat::Packed).unwrap();
        let DetectorBatch::Packed { data, .. } = packed else {
            unreachable!()
        };
        assert_eq!(data, [0b110, 0b010, 0]);

        let canonical = convert_detector_batch(source.into(), DetectorBatchFormat::Events).unwrap();
        let DetectorBatch::Events {
            offsets, events, ..
        } = canonical
        else {
            unreachable!()
        };
        assert_eq!(offsets, [0, 2, 3, 3]);
        assert_eq!(events, [1, 2, 1]);
    }

    #[test]
    fn detector_event_offset_allocation_overflow_returns_an_error() {
        let masks = DetectorMaskBatchView::new(&[], &[], usize::MAX).unwrap();
        let error = convert_detector_batch(masks.into(), DetectorBatchFormat::Events).unwrap_err();
        assert_eq!(
            error.message(),
            "detector event offset length overflowed usize"
        );

        let packed = PackedDetectorShotBatchView::new(&[], &[], usize::MAX - 1).unwrap();
        let error = convert_detector_batch(packed.into(), DetectorBatchFormat::Events).unwrap_err();
        assert!(error
            .message()
            .contains("detector event offset storage for"));
        assert!(error.message().contains("could not be allocated"));
    }

    #[test]
    fn packed_zero_and_no_correction_report_length_overflow() {
        let observable_ids = (0..9).collect::<Vec<_>>();
        let error =
            PackedObservableShotBatch::zero(observable_ids.clone(), usize::MAX).unwrap_err();
        assert_eq!(
            error.message(),
            "packed correction byte length overflowed usize"
        );

        let factory = NativeNoCorrectionDecoder::new(observable_ids);
        let mut worker = factory.create_worker().unwrap();
        let packed = PackedDetectorShotBatchView::new(&[], &[], usize::MAX).unwrap();
        let error = worker.decode_batch_checked(packed.into()).unwrap_err();
        assert_eq!(
            error.message(),
            "packed correction byte length overflowed usize"
        );

        let events = DetectorEventShotBatchView {
            detector_ids: &[],
            offsets: &[],
            events: &[],
            shots: usize::MAX,
        };
        let error = worker.decode_batch(events.into()).unwrap_err();
        assert_eq!(
            error.message(),
            "packed correction byte length overflowed usize"
        );

        let child: Arc<dyn NativeDecoderFactory> =
            Arc::new(NativeNoCorrectionDecoder::new(vec![0]));
        let factory = NativeCompositeDecoder::new(vec![child]).unwrap();
        let mut worker = factory.create_worker().unwrap();
        let capacity_overflow_shots = (isize::MAX as usize).checked_add(1).unwrap();
        let packed = PackedDetectorShotBatchView::new(&[], &[], capacity_overflow_shots).unwrap();
        let error = worker.decode_batch_checked(packed.into()).unwrap_err();
        assert!(error.message().contains("packed correction storage for"));
        assert!(error.message().contains("could not be allocated"));
    }

    #[test]
    fn public_detector_views_reject_duplicate_ids() {
        let ids = [7, 7];
        let masks = [Mask::zero(1), Mask::zero(1)];
        assert!(DetectorMaskBatchView::new(&ids, &masks, 1).is_err());
        assert!(PackedDetectorShotBatchView::new(&ids, &[0], 1).is_err());
        assert!(DetectorEventShotBatchView::new(&ids, &[0, 0], &[], 1).is_err());
    }

    #[test]
    fn checked_dispatch_trusts_prevalidated_detector_id_metadata() {
        struct TrustedMetadataWorker {
            calls: usize,
        }

        impl NativeDecoderWorker for TrustedMetadataWorker {
            fn name(&self) -> &str {
                "trusted-metadata"
            }

            fn detector_ids(&self) -> &[i64] {
                &[7, 7]
            }

            fn observable_ids(&self) -> &[i64] {
                &[]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &MASKS_BATCH_FORMATS
            }

            fn decode_batch(
                &mut self,
                detectors: DetectorBatchView<'_>,
            ) -> NpResult<DecoderCorrectionBatch> {
                self.calls += 1;
                Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::empty(
                    detectors.shots(),
                )))
            }
        }

        let ids = [7, 7];
        let masks = [Mask::zero(1), Mask::zero(1)];
        let raw_view = DetectorMaskBatchView {
            detector_ids: &ids,
            masks: &masks,
            shots: 1,
        };
        let mut worker = TrustedMetadataWorker { calls: 0 };

        worker.decode_batch_checked(raw_view.into()).unwrap();
        worker.decode_batch_checked(raw_view.into()).unwrap();

        assert_eq!(worker.calls, 2);
    }

    #[test]
    fn decoder_format_metadata_and_selection_are_strict_and_ordered() {
        assert!(validate_decoder_batch_formats(&[]).is_err());
        assert!(validate_decoder_batch_formats(&[
            DetectorBatchFormat::Packed,
            DetectorBatchFormat::Packed,
        ])
        .is_err());
        assert_eq!(
            select_detector_batch_format(
                &[DetectorBatchFormat::Events, DetectorBatchFormat::Packed],
                &[DetectorBatchFormat::Masks, DetectorBatchFormat::Packed],
            )
            .unwrap(),
            DetectorBatchFormat::Packed
        );
        assert_eq!(
            select_detector_batch_format(
                &[DetectorBatchFormat::Events, DetectorBatchFormat::Packed],
                &[DetectorBatchFormat::Masks],
            )
            .unwrap(),
            DetectorBatchFormat::Events
        );
    }

    #[test]
    fn checked_decode_enforces_the_fixed_input_to_correction_mapping() {
        struct WrongVariantWorker;

        impl NativeDecoderWorker for WrongVariantWorker {
            fn name(&self) -> &str {
                "wrong-variant"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &ALL_BATCH_FORMATS
            }

            fn decode_batch(
                &mut self,
                detectors: DetectorBatchView<'_>,
            ) -> NpResult<DecoderCorrectionBatch> {
                match detectors {
                    DetectorBatchView::Masks(view) => Ok(DecoderCorrectionBatch::Packed(
                        PackedObservableShotBatch::zero(vec![], view.shots)?,
                    )),
                    DetectorBatchView::Packed(view) => Ok(DecoderCorrectionBatch::Masks(
                        CorrectionMaskBatch::empty(view.shots),
                    )),
                    DetectorBatchView::Events(view) => Ok(DecoderCorrectionBatch::Masks(
                        CorrectionMaskBatch::empty(view.shots),
                    )),
                }
            }
        }

        let mut worker = WrongVariantWorker;
        let masks = DetectorMaskBatchView::new(&[], &[], 1).unwrap();
        let packed = PackedDetectorShotBatchView::new(&[], &[], 1).unwrap();
        let offsets = [0, 0];
        let events = DetectorEventShotBatchView::new(&[], &offsets, &[], 1).unwrap();
        for input in [masks.into(), packed.into(), events.into()] {
            let error = worker.decode_batch_checked(input).unwrap_err();
            assert!(error.message().contains("expected"));
            assert!(error.message().contains("corrections"));
        }
    }

    impl NativeDecoderFactory for FixedCorrectionDecoder {
        fn name(&self) -> &str {
            "fixed-correction"
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn batch_formats(&self) -> &[DetectorBatchFormat] {
            &MASKS_BATCH_FORMATS
        }

        fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
            Ok(Box::new(FixedCorrectionDecoderWorker {
                detector_ids: self.detector_ids.clone(),
                observable_ids: self.observable_ids.clone(),
                correction: self.correction.clone(),
            }))
        }
    }

    impl NativeDecoderWorker for FixedCorrectionDecoderWorker {
        fn name(&self) -> &str {
            "fixed-correction"
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn batch_formats(&self) -> &[DetectorBatchFormat] {
            &MASKS_BATCH_FORMATS
        }

        fn decode_batch(
            &mut self,
            detectors: DetectorBatchView<'_>,
        ) -> NpResult<DecoderCorrectionBatch> {
            let DetectorBatchView::Masks(detectors) = detectors else {
                return Err(NpError::new("fixed-correction only accepts Masks"));
            };
            Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                vec![self.correction.clone()],
                detectors.shots,
            )?))
        }
    }

    impl NativeDecoderFactory for FastCopyDecoder {
        fn name(&self) -> &str {
            "fast-copy"
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn batch_formats(&self) -> &[DetectorBatchFormat] {
            &ALL_BATCH_FORMATS
        }

        fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
            Ok(Box::new(FastCopyDecoderWorker {
                detector_ids: self.detector_ids.clone(),
                observable_ids: self.observable_ids.clone(),
            }))
        }
    }

    impl NativeDecoderWorker for FastCopyDecoderWorker {
        fn name(&self) -> &str {
            "fast-copy"
        }

        fn detector_ids(&self) -> &[i64] {
            &self.detector_ids
        }

        fn observable_ids(&self) -> &[i64] {
            &self.observable_ids
        }

        fn batch_formats(&self) -> &[DetectorBatchFormat] {
            &ALL_BATCH_FORMATS
        }

        fn decode_batch(
            &mut self,
            detectors: DetectorBatchView<'_>,
        ) -> NpResult<DecoderCorrectionBatch> {
            match detectors {
                DetectorBatchView::Masks(detectors) => {
                    Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                        self.observable_ids.clone(),
                        vec![detectors.masks[0].clone()],
                        detectors.shots,
                    )?))
                }
                DetectorBatchView::Packed(detectors) => {
                    let data = (0..detectors.shots)
                        .map(|shot| detectors.data[shot * detectors.detector_byte_count] & 1)
                        .collect();
                    Ok(DecoderCorrectionBatch::Packed(
                        PackedObservableShotBatch::new(
                            self.observable_ids.clone(),
                            data,
                            detectors.shots,
                        )?,
                    ))
                }
                DetectorBatchView::Events(detectors) => {
                    let data = (0..detectors.shots)
                        .map(|shot| {
                            (detectors.events[detectors.offsets[shot]..detectors.offsets[shot + 1]]
                                .iter()
                                .filter(|&&event| event == 0)
                                .count()
                                & 1) as u8
                        })
                        .collect();
                    Ok(DecoderCorrectionBatch::Packed(
                        PackedObservableShotBatch::new(
                            self.observable_ids.clone(),
                            data,
                            detectors.shots,
                        )?,
                    ))
                }
            }
        }
    }

    impl NativeDecoderFactory for SingleFormatCopyDecoder {
        fn name(&self) -> &str {
            "single-format-copy"
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

        fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
            Ok(Box::new(SingleFormatCopyDecoderWorker {
                detector_ids: self.detector_ids.clone(),
                observable_ids: self.observable_ids.clone(),
                formats: self.formats.clone(),
            }))
        }
    }

    impl NativeDecoderWorker for SingleFormatCopyDecoderWorker {
        fn name(&self) -> &str {
            "single-format-copy"
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
            detectors: DetectorBatchView<'_>,
        ) -> NpResult<DecoderCorrectionBatch> {
            let input_format = detectors.format();
            let shots = detectors.shots();
            let masks = convert_detector_batch(detectors, DetectorBatchFormat::Masks)?;
            let DetectorBatch::Masks { masks, .. } = masks else {
                unreachable!()
            };
            let corrections = CorrectionMaskBatch::new(
                self.observable_ids.clone(),
                vec![masks[0].clone()],
                shots,
            )?;
            if input_format == DetectorBatchFormat::Masks {
                Ok(DecoderCorrectionBatch::Masks(corrections))
            } else {
                Ok(DecoderCorrectionBatch::Packed(mask_corrections_to_packed(
                    corrections,
                )?))
            }
        }
    }

    #[test]
    fn fixed_decoder_correction_controls_native_residual_loss() {
        let factory = FixedCorrectionDecoder {
            detector_ids: vec![],
            observable_ids: vec![0],
            correction: Mask {
                words: vec![0b0011],
            },
        };
        let mut decoder = factory.create_worker().unwrap();
        let detector_ids = vec![];
        let detector_masks = vec![];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();
        let corrections = decoder
            .decode_batch(view.into())
            .unwrap()
            .into_masks()
            .unwrap();
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
        let first: Arc<dyn NativeDecoderFactory> = Arc::new(FixedCorrectionDecoder {
            detector_ids: vec![10, 20],
            observable_ids: vec![2],
            correction: Mask {
                words: vec![0b0011],
            },
        });
        let second: Arc<dyn NativeDecoderFactory> = Arc::new(FixedCorrectionDecoder {
            detector_ids: vec![20, 30],
            observable_ids: vec![5],
            correction: Mask {
                words: vec![0b1100],
            },
        });
        let factory = NativeCompositeDecoder::new(vec![first, second]).unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_masks = vec![Mask::zero(1); 3];
        let detector_ids = decoder.detector_ids().to_vec();
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let corrections = decoder
            .decode_batch_checked(view.into())
            .unwrap()
            .into_masks()
            .unwrap();

        assert_eq!(factory.detector_ids(), &[10, 20, 30]);
        assert_eq!(factory.observable_ids(), &[2, 5]);
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
        let children: Vec<Arc<dyn NativeDecoderFactory>> = vec![
            Arc::new(NativeNoCorrectionDecoder::new(vec![1])),
            Arc::new(NativeNoCorrectionDecoder::new(vec![1])),
        ];
        let err = NativeCompositeDecoder::new(children).unwrap_err();
        assert!(err.to_string().contains("duplicate observable id 1"));
    }

    #[test]
    fn composite_decoder_rejects_duplicate_child_detector_ids() {
        let child: Arc<dyn NativeDecoderFactory> = Arc::new(FixedCorrectionDecoder {
            detector_ids: vec![10, 10],
            observable_ids: vec![2],
            correction: Mask::zero(1),
        });

        let error = NativeCompositeDecoder::new(vec![child]).unwrap_err();

        assert_eq!(
            error.message(),
            "native composite decoder child 0 (`fixed-correction`) has invalid detector metadata: decoder detector ids must be unique; duplicate detector id 10"
        );
    }

    #[test]
    fn composite_decoder_dispatches_packed_and_event_fast_paths() {
        let children: Vec<Arc<dyn NativeDecoderFactory>> = vec![
            Arc::new(FastCopyDecoder {
                detector_ids: vec![10],
                observable_ids: vec![2],
            }),
            Arc::new(FastCopyDecoder {
                detector_ids: vec![20],
                observable_ids: vec![5],
            }),
        ];
        let factory = NativeCompositeDecoder::new(children).unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let packed_data = [0b01, 0b10, 0b11, 0b00];
        let detector_ids = decoder.detector_ids().to_vec();
        let packed_view = PackedDetectorShotBatchView::new(&detector_ids, &packed_data, 4).unwrap();
        let packed = decoder
            .decode_batch_checked(packed_view.into())
            .unwrap()
            .into_packed()
            .unwrap();
        assert_eq!(packed.observable_ids, vec![2, 5]);
        assert_eq!(packed.data, vec![0b01, 0b10, 0b11, 0b00]);

        let offsets = [0, 1, 2, 4, 4];
        let events = [0, 1, 0, 1];
        let event_view =
            DetectorEventShotBatchView::new(&detector_ids, &offsets, &events, 4).unwrap();
        let event = decoder.decode_batch_checked(event_view.into()).unwrap();
        let event = event.into_packed().unwrap();
        assert_eq!(event.observable_ids, vec![2, 5]);
        assert_eq!(event.data, vec![0b01, 0b10, 0b11, 0b00]);
    }

    #[test]
    fn heterogeneous_composite_ranks_formats_converts_once_and_merges() {
        let children: Vec<Arc<dyn NativeDecoderFactory>> = vec![
            Arc::new(SingleFormatCopyDecoder {
                detector_ids: vec![10],
                observable_ids: vec![2],
                formats: vec![DetectorBatchFormat::Masks],
            }),
            Arc::new(SingleFormatCopyDecoder {
                detector_ids: vec![20],
                observable_ids: vec![5],
                formats: vec![DetectorBatchFormat::Packed],
            }),
            Arc::new(SingleFormatCopyDecoder {
                detector_ids: vec![30],
                observable_ids: vec![7],
                formats: vec![DetectorBatchFormat::Events],
            }),
        ];
        let factory = NativeCompositeDecoder::new(children).unwrap();
        assert_eq!(factory.detector_ids(), [10, 20, 30]);
        assert_eq!(factory.observable_ids(), [2, 5, 7]);
        assert_eq!(factory.batch_formats(), DetectorBatchFormat::STABLE_ORDER);
        let mut worker = factory.create_worker().unwrap();

        let offsets = [0, 1, 2, 3, 6];
        let events = [0, 1, 2, 0, 1, 2];
        let view =
            DetectorEventShotBatchView::new(factory.detector_ids(), &offsets, &events, 4).unwrap();
        let corrections = worker
            .decode_batch_checked(view.into())
            .unwrap()
            .into_packed()
            .unwrap();

        assert_eq!(corrections.observable_ids, [2, 5, 7]);
        assert_eq!(corrections.data, [0b001, 0b010, 0b100, 0b111]);
    }

    #[test]
    fn composite_decoder_reuses_scratch_across_batch_shapes() {
        let children: Vec<Arc<dyn NativeDecoderFactory>> = vec![
            Arc::new(FastCopyDecoder {
                detector_ids: vec![10],
                observable_ids: vec![2],
            }),
            Arc::new(FastCopyDecoder {
                detector_ids: vec![20],
                observable_ids: vec![5],
            }),
        ];
        let factory = NativeCompositeDecoder::new(children).unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_ids = decoder.detector_ids().to_vec();

        for (shots, words) in [(4, vec![0b1010]), (65, vec![u64::MAX, 1])] {
            let first = Mask { words };
            let mut second = Mask::all(shots);
            second.xor_assign(&first);
            let masks = vec![first, second];
            let view = DetectorMaskBatchView::new(&detector_ids, &masks, shots).unwrap();
            let corrections = decoder
                .decode_batch_checked(view.into())
                .unwrap()
                .into_masks()
                .unwrap();
            assert_eq!(corrections.masks, masks);
        }

        for data in [vec![0b01, 0b10, 0b11, 0b00], vec![0b11; 9]] {
            let view = PackedDetectorShotBatchView::new(&detector_ids, &data, data.len()).unwrap();
            let corrections = decoder
                .decode_batch_checked(view.into())
                .unwrap()
                .into_packed()
                .unwrap();
            assert_eq!(corrections.data, data);
        }

        let offsets = [0, 2, 2, 3];
        let events = [0, 1, 1];
        let view = DetectorEventShotBatchView::new(&detector_ids, &offsets, &events, 3).unwrap();
        let corrections = decoder
            .decode_batch_checked(view.into())
            .unwrap()
            .into_packed()
            .unwrap();
        assert_eq!(corrections.data, vec![0b11, 0b00, 0b10]);
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
    fn correction_batch_validation_rejects_mismatched_public_field_lengths() {
        let corrections = CorrectionMaskBatch {
            observable_ids: vec![0],
            masks: Vec::new(),
            shots: 1,
        };

        let err = corrections.validate_against(&[0], 1).unwrap_err();

        assert_eq!(
            err.message(),
            "observable ids and correction masks must have the same length"
        );
    }

    #[test]
    fn correction_batches_require_the_complete_declared_layout() {
        for observable_ids in [vec![10], vec![20, 10], vec![10, 20, 30]] {
            let masks = vec![Mask::zero(1); observable_ids.len()];
            let corrections = CorrectionMaskBatch::new(observable_ids, masks, 1).unwrap();
            let err = corrections.validate_against(&[10, 20], 1).unwrap_err();
            assert!(err.to_string().contains("observable layout mismatch"));
        }
    }

    #[test]
    fn all_checked_decode_paths_require_the_complete_declared_layout() {
        struct LayoutOutputWorker {
            declared: Vec<i64>,
            actual: Vec<i64>,
        }

        impl NativeDecoderWorker for LayoutOutputWorker {
            fn name(&self) -> &str {
                "layout-output"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &self.declared
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &ALL_BATCH_FORMATS
            }

            fn decode_batch(
                &mut self,
                detectors: DetectorBatchView<'_>,
            ) -> NpResult<DecoderCorrectionBatch> {
                match detectors {
                    DetectorBatchView::Masks(view) => {
                        Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                            self.actual.clone(),
                            vec![Mask::zero(crate::word_count(view.shots)); self.actual.len()],
                            view.shots,
                        )?))
                    }
                    DetectorBatchView::Packed(view) => Ok(DecoderCorrectionBatch::Packed(
                        PackedObservableShotBatch::zero(self.actual.clone(), view.shots)?,
                    )),
                    DetectorBatchView::Events(view) => Ok(DecoderCorrectionBatch::Packed(
                        PackedObservableShotBatch::zero(self.actual.clone(), view.shots)?,
                    )),
                }
            }
        }

        for actual in [vec![10, 20], vec![10], vec![20, 10], vec![10, 20, 30]] {
            let succeeds = actual == [10, 20];
            let mut worker = LayoutOutputWorker {
                declared: vec![10, 20],
                actual,
            };

            let mask_view = DetectorMaskBatchView::new(&[], &[], 1).unwrap();
            assert_eq!(
                worker.decode_batch_checked(mask_view.into()).is_ok(),
                succeeds
            );

            let packed_view = PackedDetectorShotBatchView::new(&[], &[], 1).unwrap();
            assert_eq!(
                worker.decode_batch_checked(packed_view.into()).is_ok(),
                succeeds
            );

            let offsets = [0, 0];
            let event_view = DetectorEventShotBatchView::new(&[], &offsets, &[], 1).unwrap();
            assert_eq!(
                worker.decode_batch_checked(event_view.into()).is_ok(),
                succeeds
            );
        }
    }

    #[test]
    fn checked_decode_rejects_unknown_observable_id() {
        struct UnknownObservableFactory;
        struct UnknownObservableWorker;

        impl NativeDecoderFactory for UnknownObservableFactory {
            fn name(&self) -> &str {
                "unknown-observable"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &MASKS_BATCH_FORMATS
            }

            fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
                Ok(Box::new(UnknownObservableWorker))
            }
        }

        impl NativeDecoderWorker for UnknownObservableWorker {
            fn name(&self) -> &str {
                "unknown-observable"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &MASKS_BATCH_FORMATS
            }

            fn decode_batch(
                &mut self,
                detectors: DetectorBatchView<'_>,
            ) -> NpResult<DecoderCorrectionBatch> {
                let DetectorBatchView::Masks(detectors) = detectors else {
                    return Err(NpError::new("expected Masks"));
                };
                Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                    vec![1],
                    vec![Mask::zero(crate::word_count(detectors.shots))],
                    detectors.shots,
                )?))
            }
        }

        let factory = UnknownObservableFactory;
        let mut decoder = factory.create_worker().unwrap();
        let view = DetectorMaskBatchView::new(&[], &[], 4).unwrap();
        let err = decoder.decode_batch_checked(view.into()).unwrap_err();

        assert!(err
            .to_string()
            .contains("correction observable layout mismatch: expected [0], got [1]"));
    }

    #[test]
    fn checked_decode_rejects_shots_mismatch() {
        struct MismatchedShotsFactory;
        struct MismatchedShotsWorker;

        impl NativeDecoderFactory for MismatchedShotsFactory {
            fn name(&self) -> &str {
                "mismatched-shots"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &MASKS_BATCH_FORMATS
            }

            fn create_worker(&self) -> NpResult<Box<dyn NativeDecoderWorker>> {
                Ok(Box::new(MismatchedShotsWorker))
            }
        }

        impl NativeDecoderWorker for MismatchedShotsWorker {
            fn name(&self) -> &str {
                "mismatched-shots"
            }

            fn detector_ids(&self) -> &[i64] {
                &[]
            }

            fn observable_ids(&self) -> &[i64] {
                &[0]
            }

            fn batch_formats(&self) -> &[DetectorBatchFormat] {
                &MASKS_BATCH_FORMATS
            }

            fn decode_batch(
                &mut self,
                _detectors: DetectorBatchView<'_>,
            ) -> NpResult<DecoderCorrectionBatch> {
                Ok(DecoderCorrectionBatch::Masks(CorrectionMaskBatch::new(
                    vec![0],
                    vec![Mask::zero(crate::word_count(5))],
                    5,
                )?))
            }
        }

        let factory = MismatchedShotsFactory;
        let mut decoder = factory.create_worker().unwrap();
        let view = DetectorMaskBatchView::new(&[], &[], 4).unwrap();
        let err = decoder.decode_batch_checked(view.into()).unwrap_err();

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
        let factory = NativeNoCorrectionDecoder::new(vec![0, 2]);
        let mut decoder = factory.create_worker().unwrap();
        let detector_ids = vec![1];
        let detector_masks = vec![Mask {
            words: vec![0b1010],
        }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let corrections = decoder
            .decode_batch(view.into())
            .unwrap()
            .into_masks()
            .unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 2]);
        assert_eq!(corrections.masks, vec![Mask { words: vec![0] }; 2]);
    }

    #[test]
    fn no_correction_decoder_returns_zero_packed_rows_for_observables() {
        let factory = NativeNoCorrectionDecoder::with_detector_ids(vec![1, 2], vec![0, 3]).unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_data = vec![0b11, 0b01, 0b10, 0b00];
        let detector_ids = decoder.detector_ids().to_vec();
        let view = PackedDetectorShotBatchView::new(&detector_ids, &detector_data, 4).unwrap();

        let corrections = decoder
            .decode_batch_checked(view.into())
            .unwrap()
            .into_packed()
            .unwrap();

        assert_eq!(corrections.observable_ids, vec![0, 3]);
        assert_eq!(corrections.shots, 4);
        assert_eq!(corrections.observable_byte_count, 1);
        assert_eq!(corrections.data, vec![0; 4]);
    }

    #[test]
    fn no_correction_decoder_rejects_duplicate_detector_ids() {
        let err = NativeNoCorrectionDecoder::with_detector_ids(vec![1, 2, 1], vec![0]).unwrap_err();

        assert_eq!(
            err.message(),
            "decoder detector ids must be unique; duplicate detector id 1"
        );
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
        let mut edges = graphlike_edges();
        edges.push(GraphlikeEdge {
            detectors: vec![0],
            fault_observables: vec![0],
            probability: 0.3,
            dem_edge_index: 2,
        });
        let problem = GraphlikeDecodingProblem::new(
            vec![10, 20],
            vec![Vec::new(), Vec::new()],
            vec![0, 1],
            edges,
        )
        .unwrap();

        let err = NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(problem).unwrap_err();

        assert!(err
            .to_string()
            .contains("multiple single-detector candidate edges for observable id 0"));
    }

    #[test]
    fn graphlike_detector_copy_decodes_by_copying_detector_masks() {
        let factory =
            NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(graphlike_problem())
                .unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_masks = vec![
            Mask {
                words: vec![0b0011],
            },
            Mask {
                words: vec![0b1010],
            },
        ];
        let detector_ids = decoder.detector_ids().to_vec();
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let corrections = decoder
            .decode_batch_checked(view.into())
            .unwrap()
            .into_masks()
            .unwrap();

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
        let factory =
            NativeGraphlikeDetectorCopyDecoder::from_graphlike_problem(graphlike_problem())
                .unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_ids = vec![20, 10];
        let detector_masks = vec![Mask { words: vec![0] }, Mask { words: vec![0] }];
        let view = DetectorMaskBatchView::new(&detector_ids, &detector_masks, 4).unwrap();

        let err = decoder.decode_batch(view.into()).unwrap_err();

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
        let factory =
            NativeFusionBlossomDecoder::from_graphlike_problem(graphlike_problem()).unwrap();
        let mut decoder = factory.create_worker().unwrap();
        let detector_data = vec![0; 4];
        let detector_ids = decoder.detector_ids().to_vec();
        let view = PackedDetectorShotBatchView::new(&detector_ids, &detector_data, 4).unwrap();

        let err = decoder.decode_batch_checked(view.into()).unwrap_err();

        assert!(err
            .to_string()
            .contains("no stable Rust dependency is linked yet"));
    }
}
