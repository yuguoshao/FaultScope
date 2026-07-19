from __future__ import annotations

import ctypes
from concurrent.futures import ThreadPoolExecutor


CAPSULE_NAME = b"faultscope.native_decoder_plugin.v4"
MASKS = 1
PACKED = 2
EVENTS = 3


class StringView(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_size_t)]


class I64Slice(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_int64)), ("len", ctypes.c_size_t)]


class U32Slice(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint32)), ("len", ctypes.c_size_t)]


class Status(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int32), ("message", StringView)]


class MaskView(ctypes.Structure):
    _fields_ = [("words", ctypes.POINTER(ctypes.c_uint64)), ("word_count", ctypes.c_size_t)]


class MaskMutView(ctypes.Structure):
    _fields_ = [("words", ctypes.POINTER(ctypes.c_uint64)), ("word_count", ctypes.c_size_t)]


class MaskBatch(ctypes.Structure):
    _fields_ = [
        ("detector_ids", ctypes.POINTER(ctypes.c_int64)),
        ("detector_count", ctypes.c_size_t),
        ("masks", ctypes.POINTER(MaskView)),
        ("shots", ctypes.c_size_t),
        ("word_count", ctypes.c_size_t),
    ]


class CorrectionBatch(ctypes.Structure):
    _fields_ = [
        ("observable_ids", ctypes.POINTER(ctypes.c_int64)),
        ("observable_count", ctypes.c_size_t),
        ("masks", ctypes.POINTER(MaskMutView)),
        ("shots", ctypes.c_size_t),
        ("word_count", ctypes.c_size_t),
    ]


class PackedDetectorBatch(ctypes.Structure):
    _fields_ = [
        ("detector_ids", ctypes.POINTER(ctypes.c_int64)),
        ("detector_count", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("shots", ctypes.c_size_t),
        ("detector_byte_count", ctypes.c_size_t),
    ]


class DetectorEventBatch(ctypes.Structure):
    _fields_ = [
        ("detector_ids", ctypes.POINTER(ctypes.c_int64)),
        ("detector_count", ctypes.c_size_t),
        ("offsets", ctypes.POINTER(ctypes.c_size_t)),
        ("offsets_len", ctypes.c_size_t),
        ("events", ctypes.POINTER(ctypes.c_size_t)),
        ("event_count", ctypes.c_size_t),
        ("shots", ctypes.c_size_t),
    ]


class PackedObservableBatch(ctypes.Structure):
    _fields_ = [
        ("observable_ids", ctypes.POINTER(ctypes.c_int64)),
        ("observable_count", ctypes.c_size_t),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("shots", ctypes.c_size_t),
        ("observable_byte_count", ctypes.c_size_t),
    ]


class DetectorPayload(ctypes.Union):
    _fields_ = [
        ("masks", MaskBatch),
        ("packed", PackedDetectorBatch),
        ("events", DetectorEventBatch),
    ]


class TaggedDetectorBatch(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("payload", DetectorPayload),
    ]


class CorrectionPayload(ctypes.Union):
    _fields_ = [("masks", CorrectionBatch), ("packed", PackedObservableBatch)]


class TaggedCorrectionBatch(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("payload", CorrectionPayload),
    ]


class WorkerV4(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_size_t),
        ("worker_state", ctypes.c_void_p),
        ("drop_worker_state", ctypes.c_void_p),
        ("decode_batch", ctypes.c_void_p),
    ]


class FactoryV4(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("struct_size", ctypes.c_size_t),
        ("flags", ctypes.c_uint64),
        ("factory_state", ctypes.c_void_p),
        ("drop_factory_state", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
        ("detector_ids", ctypes.c_void_p),
        ("observable_ids", ctypes.c_void_p),
        ("batch_formats", ctypes.c_void_p),
        ("create_worker", ctypes.c_void_p),
    ]


CREATE_WORKER = ctypes.CFUNCTYPE(
    Status,
    ctypes.c_void_p,
    ctypes.POINTER(WorkerV4),
    ctypes.c_size_t,
)
DROP_WORKER = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
GET_FORMATS = ctypes.CFUNCTYPE(Status, ctypes.c_void_p, ctypes.POINTER(U32Slice))
DECODE = ctypes.CFUNCTYPE(
    Status,
    ctypes.c_void_p,
    ctypes.POINTER(TaggedDetectorBatch),
    ctypes.POINTER(TaggedCorrectionBatch),
)


_py_capsule_get_name = ctypes.pythonapi.PyCapsule_GetName
_py_capsule_get_name.argtypes = [ctypes.py_object]
_py_capsule_get_name.restype = ctypes.c_char_p
_py_capsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_py_capsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_py_capsule_get_pointer.restype = ctypes.c_void_p


def _bytes(value: ctypes.Structure) -> bytes:
    return ctypes.string_at(ctypes.byref(value), ctypes.sizeof(value))


def _filled_worker() -> WorkerV4:
    worker = WorkerV4()
    ctypes.memset(ctypes.byref(worker), 0xA5, ctypes.sizeof(worker))
    return worker


def assert_v4_worker_contract(test, decoder):
    test.assertEqual(ctypes.sizeof(U32Slice), 16)
    test.assertEqual(ctypes.sizeof(TaggedDetectorBatch), 64)
    test.assertEqual(ctypes.sizeof(TaggedCorrectionBatch), 48)
    test.assertEqual(ctypes.sizeof(FactoryV4), 80)
    test.assertEqual(ctypes.sizeof(WorkerV4), 32)

    test_stats = decoder._inner._test_stats_for_test()
    capsule = decoder.__faultscope_native_decoder_capsule__()
    test.assertEqual(_py_capsule_get_name(capsule), CAPSULE_NAME)
    pointer = _py_capsule_get_pointer(capsule, CAPSULE_NAME)
    test.assertTrue(pointer)
    factory = ctypes.cast(pointer, ctypes.POINTER(FactoryV4)).contents
    test.assertEqual(factory.abi_version, 4)
    test.assertEqual(factory.struct_size, ctypes.sizeof(FactoryV4))
    test.assertEqual(factory.flags, 1)
    test.assertTrue(factory.factory_state)
    test.assertTrue(factory.drop_factory_state)
    test.assertTrue(factory.name)
    test.assertTrue(factory.detector_ids)
    test.assertTrue(factory.observable_ids)
    test.assertTrue(factory.batch_formats)
    test.assertTrue(factory.create_worker)

    formats = U32Slice()
    status = GET_FORMATS(factory.batch_formats)(factory.factory_state, ctypes.byref(formats))
    test.assertEqual(status.code, 0)
    test.assertTrue(formats.ptr)
    test.assertEqual(tuple(formats.ptr[index] for index in range(formats.len)), (PACKED,))

    create_worker = CREATE_WORKER(factory.create_worker)
    sentinel = _filled_worker()
    before = _bytes(sentinel)
    status = create_worker(
        factory.factory_state,
        ctypes.byref(sentinel),
        ctypes.sizeof(WorkerV4) - 1,
    )
    test.assertNotEqual(status.code, 0)
    test.assertEqual(_bytes(sentinel), before)

    status = create_worker(factory.factory_state, None, ctypes.sizeof(WorkerV4))
    test.assertNotEqual(status.code, 0)

    null_factory_out = _filled_worker()
    before = _bytes(null_factory_out)
    status = create_worker(None, ctypes.byref(null_factory_out), ctypes.sizeof(WorkerV4))
    test.assertNotEqual(status.code, 0)
    test.assertEqual(_bytes(null_factory_out), before)

    first = WorkerV4()
    second = WorkerV4()
    test.assertEqual(
        create_worker(factory.factory_state, ctypes.byref(first), ctypes.sizeof(first)).code,
        0,
    )
    test.assertEqual(
        create_worker(factory.factory_state, ctypes.byref(second), ctypes.sizeof(second)).code,
        0,
    )
    try:
        test.assertEqual(first.struct_size, ctypes.sizeof(WorkerV4))
        test.assertEqual(second.struct_size, ctypes.sizeof(WorkerV4))
        test.assertTrue(first.worker_state)
        test.assertTrue(second.worker_state)
        test.assertNotEqual(first.worker_state, second.worker_state)
        for worker in (first, second):
            test.assertTrue(worker.drop_worker_state)
            test.assertTrue(worker.decode_batch)
            test.assertEqual(_decode_packed(worker, decoder), 1)
            test.assertNotEqual(_decode_mask(worker, decoder)[0].code, 0)
            test.assertNotEqual(_decode_events(worker, decoder)[0].code, 0)

            first_error = DECODE(worker.decode_batch)(worker.worker_state, None, None)
            test.assertNotEqual(first_error.code, 0)
            test.assertTrue(first_error.message.ptr)
            retained_message = ctypes.string_at(
                first_error.message.ptr,
                first_error.message.len,
            )
            second_error = DECODE(worker.decode_batch)(worker.worker_state, None, None)
            test.assertNotEqual(second_error.code, 0)
            test.assertEqual(
                ctypes.string_at(first_error.message.ptr, first_error.message.len),
                retained_message,
            )
    finally:
        for worker in (first, second):
            if worker.worker_state and worker.drop_worker_state:
                DROP_WORKER(worker.drop_worker_state)(worker.worker_state)
                worker.worker_state = None
    test.assertEqual(test_stats.worker_creates, 2)
    test.assertEqual(test_stats.worker_drops, 2)
    test.assertEqual(len(set(test_stats.worker_addresses)), 2)
    return test_stats


def assert_factory_failure_lifetimes(test, invalid_decoder_type) -> None:
    for kind, expected in (
        ("create-error", b"worker creation failed"),
        ("create-panic", b"worker creation panicked"),
    ):
        decoder = invalid_decoder_type(kind)
        capsule = decoder.__faultscope_native_decoder_capsule__()
        pointer = _py_capsule_get_pointer(capsule, CAPSULE_NAME)
        factory = ctypes.cast(pointer, ctypes.POINTER(FactoryV4)).contents
        create_worker = CREATE_WORKER(factory.create_worker)

        def fail_once(_index):
            output = _filled_worker()
            before = _bytes(output)
            status = create_worker(
                factory.factory_state,
                ctypes.byref(output),
                ctypes.sizeof(output),
            )
            return status, before, _bytes(output)

        with ThreadPoolExecutor(max_workers=4) as executor:
            failures = list(executor.map(fail_once, range(8)))

        for status, before, after in failures:
            test.assertNotEqual(status.code, 0)
            test.assertEqual(after, before)
            test.assertTrue(status.message.ptr)
            test.assertIn(expected, ctypes.string_at(status.message.ptr, status.message.len))
        retained = [
            ctypes.string_at(status.message.ptr, status.message.len)
            for status, _before, _after in failures
        ]
        test.assertTrue(all(expected in message for message in retained))


def _ids(decoder):
    return (
        (ctypes.c_int64 * 1)(decoder.detector_ids[0]),
        (ctypes.c_int64 * 1)(decoder.observable_ids[0]),
    )


def _decode_mask(worker: WorkerV4, decoder):
    detector_ids, observable_ids = _ids(decoder)
    detector_words = (ctypes.c_uint64 * 1)(1)
    observable_words = (ctypes.c_uint64 * 1)(0)
    masks = (MaskView * 1)(MaskView(detector_words, 1))
    corrections = (MaskMutView * 1)(MaskMutView(observable_words, 1))
    input_payload = DetectorPayload()
    input_payload.masks = MaskBatch(detector_ids, 1, masks, 1, 1)
    output_payload = CorrectionPayload()
    output_payload.masks = CorrectionBatch(observable_ids, 1, corrections, 1, 1)
    inputs = TaggedDetectorBatch(MASKS, 0, input_payload)
    outputs = TaggedCorrectionBatch(MASKS, 0, output_payload)
    status = DECODE(worker.decode_batch)(
        worker.worker_state,
        ctypes.byref(inputs),
        ctypes.byref(outputs),
    )
    return status, observable_words[0]


def _decode_packed(worker: WorkerV4, decoder) -> int:
    status, value = _decode_packed_raw(worker, decoder)
    if status.code:
        raise AssertionError(f"packed callback failed with status {status.code}")
    return value


def _decode_packed_raw(
    worker: WorkerV4,
    decoder,
    *,
    input_reserved: int = 0,
    output_reserved: int = 0,
):
    detector_ids, observable_ids = _ids(decoder)
    detector_data = (ctypes.c_uint8 * 1)(1)
    observable_data = (ctypes.c_uint8 * 1)(0)
    input_payload = DetectorPayload()
    input_payload.packed = PackedDetectorBatch(detector_ids, 1, detector_data, 1, 1)
    output_payload = CorrectionPayload()
    output_payload.packed = PackedObservableBatch(observable_ids, 1, observable_data, 1, 1)
    inputs = TaggedDetectorBatch(PACKED, input_reserved, input_payload)
    outputs = TaggedCorrectionBatch(PACKED, output_reserved, output_payload)
    status = DECODE(worker.decode_batch)(
        worker.worker_state,
        ctypes.byref(inputs),
        ctypes.byref(outputs),
    )
    return status, observable_data[0]


def _decode_events(worker: WorkerV4, decoder):
    detector_ids, observable_ids = _ids(decoder)
    offsets = (ctypes.c_size_t * 2)(0, 1)
    events = (ctypes.c_size_t * 1)(0)
    observable_data = (ctypes.c_uint8 * 1)(0)
    input_payload = DetectorPayload()
    input_payload.events = DetectorEventBatch(detector_ids, 1, offsets, 2, events, 1, 1)
    output_payload = CorrectionPayload()
    output_payload.packed = PackedObservableBatch(observable_ids, 1, observable_data, 1, 1)
    inputs = TaggedDetectorBatch(EVENTS, 0, input_payload)
    outputs = TaggedCorrectionBatch(PACKED, 0, output_payload)
    status = DECODE(worker.decode_batch)(
        worker.worker_state,
        ctypes.byref(inputs),
        ctypes.byref(outputs),
    )
    return status, observable_data[0]
