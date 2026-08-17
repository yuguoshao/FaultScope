import ctypes
import gc
import pathlib
import subprocess
import tempfile
import unittest
from types import SimpleNamespace

from faultscope import _native
from faultscope.core import BernoulliPauliNoise, Circuit, NoiseLocation, Operation
from faultscope.decoders import (
    NativeBatchDecoder,
    NativeCompositeDecoder,
    NativeGraphlikeDetectorCopyDecoder,
    NativeNoCorrectionDecoder,
)
from faultscope.dem import (
    Detector,
    DetectorErrorEdge,
    DetectorErrorModel,
    LogicalObservable,
)
from faultscope.runtime import compile_native_dem_sampler, compile_native_sampler
from tests.native_backend_v4_helpers import (
    CREATE_WORKER,
    DROP_WORKER,
    FactoryV4,
    TaggedCorrectionBatch,
    TaggedDetectorBatch,
    U32Slice,
    WorkerV4,
    _decode_events,
    _decode_mask,
    _decode_packed,
    _decode_packed_raw,
)


CAPSULE_V4 = b"faultscope.native_decoder_plugin.v4"
CAPSULE_V3 = b"faultscope.native_decoder_plugin.v3"
CAPSULE_V2 = b"faultscope.native_decoder_plugin.v2"
CAPSULE_V1 = b"faultscope.native_decoder_plugin.v1"


class _FixtureDecoder:
    def __init__(
        self,
        library,
        mode,
        capsule_name=CAPSULE_V4,
        *,
        install_capsule_destructor=True,
    ):
        self._library = library
        self._pointer = library.fs_fixture_new(mode)
        self._capsule_name = ctypes.create_string_buffer(capsule_name)
        self._install_capsule_destructor = install_capsule_destructor
        pointer = self._pointer
        drop_factory = library.fs_fixture_drop_factory

        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def capsule_destructor(_capsule):
            drop_factory(pointer)

        self._capsule_destructor = capsule_destructor
        destructor = (
            ctypes.cast(self._capsule_destructor, ctypes.c_void_p)
            if install_capsule_destructor
            else None
        )
        self._capsule = self._new_capsule(
            library.fs_fixture_descriptor(self._pointer),
            ctypes.cast(self._capsule_name, ctypes.c_char_p),
            destructor,
        )

    def __faultscope_native_decoder_capsule__(self):
        return self._capsule

    @property
    def creates(self):
        return self._library.fs_fixture_creates(self._pointer)

    @property
    def worker_drops(self):
        return self._library.fs_fixture_worker_drops(self._pointer)

    @property
    def factory_drops(self):
        return self._library.fs_fixture_factory_drops(self._pointer)

    @property
    def callback_calls(self):
        return (
            self._library.fs_fixture_mask_decode_calls(self._pointer),
            self._library.fs_fixture_packed_decode_calls(self._pointer),
            self._library.fs_fixture_event_decode_calls(self._pointer),
        )

    @property
    def last_capacity(self):
        return self._library.fs_fixture_last_capacity(self._pointer)

    @property
    def worker_size(self):
        return self._library.fs_fixture_worker_size()

    def worker_address(self, index):
        return self._library.fs_fixture_worker_address(self._pointer, index)

    def close(self):
        if self._pointer:
            self._capsule = None
            gc.collect()
            if not self._install_capsule_destructor and self.factory_drops == 0:
                self._library.fs_fixture_drop_factory(self._pointer)
            if self.factory_drops != 1:
                raise AssertionError(
                    f"fixture capsule dropped factory state {self.factory_drops} times"
                )
            self._library.fs_fixture_free(self._pointer)
            self._pointer = None


class NativeDecoderAbiV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tempdir = tempfile.TemporaryDirectory(prefix="faultscope-abi-v4-")
        source = pathlib.Path(__file__).with_name("native_decoder_v4_fixture.c")
        library_path = pathlib.Path(cls._tempdir.name) / "native_decoder_v4_fixture.so"
        subprocess.run(
            ["cc", "-shared", "-fPIC", "-O0", str(source), "-o", str(library_path)],
            check=True,
        )
        cls.library = ctypes.CDLL(str(library_path))
        cls.library.fs_fixture_new.argtypes = [ctypes.c_int]
        cls.library.fs_fixture_new.restype = ctypes.c_void_p
        cls.library.fs_fixture_free.argtypes = [ctypes.c_void_p]
        cls.library.fs_fixture_drop_factory.argtypes = [ctypes.c_void_p]
        cls.library.fs_fixture_descriptor.argtypes = [ctypes.c_void_p]
        cls.library.fs_fixture_descriptor.restype = ctypes.c_void_p
        for name in (
            "fs_fixture_creates",
            "fs_fixture_worker_drops",
            "fs_fixture_factory_drops",
            "fs_fixture_mask_decode_calls",
            "fs_fixture_packed_decode_calls",
            "fs_fixture_event_decode_calls",
        ):
            function = getattr(cls.library, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_uint64
        cls.library.fs_fixture_last_capacity.argtypes = [ctypes.c_void_p]
        cls.library.fs_fixture_last_capacity.restype = ctypes.c_size_t
        cls.library.fs_fixture_worker_size.restype = ctypes.c_size_t
        cls.library.fs_fixture_worker_address.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        cls.library.fs_fixture_worker_address.restype = ctypes.c_size_t
        cls._new_capsule = ctypes.pythonapi.PyCapsule_New
        cls._new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        cls._new_capsule.restype = ctypes.py_object
        _FixtureDecoder._new_capsule = cls._new_capsule

    @classmethod
    def tearDownClass(cls):
        cls._tempdir.cleanup()

    def fixture(
        self,
        mode,
        capsule_name=CAPSULE_V4,
        *,
        install_capsule_destructor=True,
    ):
        fixture = _FixtureDecoder(
            self.library,
            mode,
            capsule_name,
            install_capsule_destructor=install_capsule_destructor,
        )
        self.addCleanup(fixture.close)
        return fixture

    @staticmethod
    def batch(mask=0b1011, shots=4):
        return SimpleNamespace(shots=shots, detectors={0: mask})

    @staticmethod
    def dem():
        return DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=0),),
            edges=(DetectorErrorEdge(1.0, (0,), (0,), "edge0", "X"),),
        )

    def test_v4_ctypes_layouts_are_frozen(self):
        self.assertEqual(ctypes.sizeof(U32Slice), 16)
        self.assertEqual(ctypes.alignment(U32Slice), 8)
        self.assertEqual((U32Slice.ptr.offset, U32Slice.len.offset), (0, 8))
        self.assertEqual(ctypes.sizeof(TaggedDetectorBatch), 64)
        self.assertEqual(ctypes.alignment(TaggedDetectorBatch), 8)
        self.assertEqual(
            (
                TaggedDetectorBatch.format.offset,
                TaggedDetectorBatch.reserved.offset,
                TaggedDetectorBatch.payload.offset,
            ),
            (0, 4, 8),
        )
        self.assertEqual(ctypes.sizeof(TaggedCorrectionBatch), 48)
        self.assertEqual(ctypes.alignment(TaggedCorrectionBatch), 8)
        self.assertEqual(
            (
                TaggedCorrectionBatch.format.offset,
                TaggedCorrectionBatch.reserved.offset,
                TaggedCorrectionBatch.payload.offset,
            ),
            (0, 4, 8),
        )
        self.assertEqual(ctypes.sizeof(FactoryV4), 80)
        self.assertEqual(ctypes.alignment(FactoryV4), 8)
        self.assertEqual(
            tuple(getattr(FactoryV4, field).offset for field, _ in FactoryV4._fields_),
            (0, 8, 16, 24, 32, 40, 48, 56, 64, 72),
        )
        self.assertEqual(ctypes.sizeof(WorkerV4), 32)
        self.assertEqual(ctypes.alignment(WorkerV4), 8)
        self.assertEqual(
            tuple(getattr(WorkerV4, field).offset for field, _ in WorkerV4._fields_),
            (0, 8, 16, 24),
        )

    def test_python_public_decoder_class_names_remain_unchanged(self):
        self.assertEqual(NativeBatchDecoder.__name__, "NativeBatchDecoder")
        self.assertEqual(NativeCompositeDecoder.__name__, "NativeCompositeDecoder")
        self.assertEqual(NativeNoCorrectionDecoder.__name__, "NativeNoCorrectionDecoder")
        self.assertEqual(
            NativeGraphlikeDetectorCopyDecoder.__name__,
            "NativeGraphlikeDetectorCopyDecoder",
        )

    def test_builtin_debug_mask_helpers_preserve_corrections(self):
        empty_factories = (
            NativeBatchDecoder.no_correction(observable_ids=(0,), detector_ids=(0,)),
            NativeNoCorrectionDecoder(observable_ids=(0,), detector_ids=(0,)),
        )
        for decoder in empty_factories:
            with self.subTest(decoder=decoder.__class__.__name__):
                self.assertEqual(decoder.decode_batch_masks(self.batch()), {0: 0})

        copying = NativeGraphlikeDetectorCopyDecoder.from_dem(self.dem())
        self.assertEqual(copying.decode_batch_masks(self.batch()), {0: 0b1011})

    def test_exact_v4_capsule_creates_distinct_temporary_workers(self):
        fixture = self.fixture(0)
        decoder = NativeCompositeDecoder((fixture,))

        first = decoder.decode_batch_masks(self.batch())
        second = decoder.decode_batch_masks(self.batch())

        self.assertEqual(first, {0: 0b1011})
        self.assertEqual(second, first)
        self.assertEqual(fixture.creates, 2)
        self.assertEqual(fixture.worker_drops, 2)
        self.assertNotEqual(fixture.worker_address(0), fixture.worker_address(1))
        self.assertEqual(fixture.last_capacity, fixture.worker_size)
        del decoder
        gc.collect()
        self.assertEqual(fixture.factory_drops, 0)
        fixture._capsule = None
        gc.collect()
        self.assertEqual(fixture.factory_drops, 1)

    def test_repeated_same_capsule_conversions_share_factory_ownership(self):
        fixture = self.fixture(0)
        sampler = compile_native_dem_sampler(self.dem())

        first = sampler.estimate(shots=4, seed=11, decoder=fixture)
        second = sampler.estimate(shots=4, seed=12, decoder=fixture)

        self.assertEqual(first.mean_loss, 0.0)
        self.assertEqual(second.mean_loss, 0.0)
        self.assertEqual(fixture.creates, 2)
        self.assertEqual(fixture.worker_drops, 2)
        self.assertEqual(fixture.factory_drops, 0)
        fixture._capsule = None
        gc.collect()
        self.assertEqual(fixture.factory_drops, 1)

    def test_v1_through_v3_capsule_names_are_rejected_with_v4_context(self):
        for capsule_name in (CAPSULE_V1, CAPSULE_V2, CAPSULE_V3):
            with self.subTest(capsule_name=capsule_name):
                fixture = self.fixture(0, capsule_name)
                with self.assertRaisesRegex(ValueError, r"expected ABI v4"):
                    NativeCompositeDecoder((fixture,))

    def test_v1_through_v3_numeric_versions_are_rejected(self):
        for mode in (24, 25, 10):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, r"expected ABI v4"):
                    NativeCompositeDecoder((fixture,))

    def test_capsule_without_destructor_is_rejected_before_factory_use(self):
        fixture = self.fixture(0, install_capsule_destructor=False)

        with self.assertRaisesRegex(
            ValueError,
            r"capsule must provide a destructor.*drop_factory_state exactly once",
        ):
            NativeCompositeDecoder((fixture,))

        self.assertEqual(fixture.creates, 0)
        self.assertEqual(fixture.factory_drops, 0)

    def test_invalid_v4_factory_descriptors_and_formats_are_rejected(self):
        cases = {
            11: r"ABI v4.*size",
            12: r"ABI v4.*thread-safe",
            13: r"ABI v4.*null factory state",
            14: r"ABI v4.*required callbacks",
            15: r"ABI v4.*required callbacks",
            16: r"ABI v4.*required callbacks",
            17: r"ABI v4.*required callbacks",
            26: r"ABI v4.*required callbacks",
            27: r"format preference list must not be empty",
            28: r"duplicate format",
            29: r"unknown native decoder batch format tag 99",
            30: r"null batch format pointer",
            23: r"decoder detector ids must be unique; duplicate detector id 0",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, message):
                    NativeCompositeDecoder((fixture,))

    def test_invalid_worker_descriptors_and_metadata_are_rejected(self):
        cases = {
            5: (r"ABI v4.*worker descriptor size", 1),
            6: (r"ABI v4.*unsafe non-null partial worker state", 0),
            7: (r"ABI v4.*missing required callbacks", 1),
            8: (r"ABI v4.*null worker state", 0),
            9: (r"modified tagged mask metadata", 1),
            20: (r"inconsistent or unstable decoder metadata", 1),
            31: (r"inconsistent or unstable decoder metadata", 1),
        }
        for mode, (message, expected_drops) in cases.items():
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                decoder = NativeCompositeDecoder((fixture,))
                with self.assertRaisesRegex(ValueError, message):
                    decoder.decode_batch_masks(self.batch())
                self.assertEqual(fixture.worker_drops, expected_drops)

    def test_create_failure_empty_and_partial_cleanup(self):
        empty = self.fixture(3)
        empty_decoder = NativeCompositeDecoder((empty,))
        with self.assertRaisesRegex(ValueError, "test create failure"):
            empty_decoder.decode_batch_masks(self.batch())
        self.assertEqual(empty.worker_drops, 0)

        partial = self.fixture(4)
        partial_decoder = NativeCompositeDecoder((partial,))
        with self.assertRaisesRegex(ValueError, "test create failure"):
            partial_decoder.decode_batch_masks(self.batch())
        self.assertEqual(partial.worker_drops, 1)

    def test_all_three_tags_enter_the_single_callback_and_reserved_must_be_zero(self):
        fixture = self.fixture(2)
        factory = ctypes.cast(
            self.library.fs_fixture_descriptor(fixture._pointer),
            ctypes.POINTER(FactoryV4),
        ).contents
        worker = WorkerV4()
        status = CREATE_WORKER(factory.create_worker)(
            factory.factory_state,
            ctypes.byref(worker),
            ctypes.sizeof(worker),
        )
        self.assertEqual(status.code, 0)
        decoder = SimpleNamespace(detector_ids=(0,), observable_ids=(0,))
        try:
            mask_status, mask_value = _decode_mask(worker, decoder)
            self.assertEqual(mask_status.code, 0)
            self.assertEqual(mask_value, 1)
            self.assertEqual(_decode_packed(worker, decoder), 1)
            event_status, event_value = _decode_events(worker, decoder)
            self.assertEqual(event_status.code, 0)
            self.assertEqual(event_value, 1)
            input_reserved, _ = _decode_packed_raw(
                worker,
                decoder,
                input_reserved=1,
            )
            output_reserved, _ = _decode_packed_raw(
                worker,
                decoder,
                output_reserved=1,
            )
            self.assertNotEqual(input_reserved.code, 0)
            self.assertNotEqual(output_reserved.code, 0)
        finally:
            DROP_WORKER(worker.drop_worker_state)(worker.worker_state)
        self.assertEqual(fixture.callback_calls, (1, 1, 1))

    def test_dem_estimate_honors_masks_packed_and_events_preference_with_hotspots(self):
        cases = {
            "masks": (0, (1, 0, 0)),
            "packed": (2, (0, 1, 0)),
            "events": (1, (0, 0, 1)),
        }
        for aggregate_hotspots in (False, True):
            for path, (mode, expected_calls) in cases.items():
                with self.subTest(path=path, aggregate_hotspots=aggregate_hotspots):
                    fixture = self.fixture(mode)
                    result = compile_native_dem_sampler(self.dem()).estimate(
                        shots=16,
                        seed=123,
                        decoder=fixture,
                        aggregate_hotspots=aggregate_hotspots,
                    )
                    self.assertEqual(result.mean_loss, 0.0)
                    self.assertEqual(fixture.callback_calls, expected_calls)
                    self.assertEqual(fixture.creates, 1)
                    self.assertEqual(fixture.worker_drops, 1)

    def test_tag_pointer_dimension_and_nested_view_tampering_is_rejected(self):
        for mode in range(32, 40):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                decoder = NativeCompositeDecoder((fixture,))
                with self.assertRaisesRegex(ValueError, r"modified tagged .*metadata"):
                    decoder.decode_batch_masks(self.batch())
                self.assertEqual(fixture.worker_drops, 1)

    def test_packed_and_event_invalid_output_shapes_are_rejected(self):
        for mode in (18, 19):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, r"modified tagged batch metadata"):
                    compile_native_dem_sampler(self.dem()).estimate(
                        shots=4,
                        seed=321,
                        decoder=fixture,
                        aggregate_hotspots=False,
                    )
                self.assertEqual(fixture.worker_drops, 1)

    def test_packed_and_event_nonzero_padding_is_rejected(self):
        for mode in (21, 22):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, r"non-zero padding bits"):
                    compile_native_dem_sampler(self.dem()).estimate(
                        shots=4,
                        seed=654,
                        decoder=fixture,
                        aggregate_hotspots=False,
                    )
                self.assertEqual(fixture.worker_drops, 1)

    def test_sampler_layout_mismatch_is_rejected_before_worker_creation(self):
        dem = DetectorErrorModel(
            detectors=(Detector(id=0, measurement_keys=()),),
            observables=(LogicalObservable(id=1),),
            edges=(DetectorErrorEdge(1.0, (0,), (1,), "edge1", "X"),),
        )
        fixture = self.fixture(0)

        with self.assertRaisesRegex(
            ValueError,
            r"expected sampler canonical ids \[1\], got \[0\]",
        ):
            compile_native_dem_sampler(dem).estimate(
                shots=4,
                seed=987,
                decoder=fixture,
            )

        self.assertEqual(fixture.creates, 0)
        self.assertEqual(fixture.callback_calls, (0, 0, 0))

    def test_circuit_estimate_converts_to_each_decoder_preference(self):
        location = NoiseLocation("x0", BernoulliPauliNoise("X"), 1.0, (0,))
        circuit = Circuit(
            1,
            (
                Operation.noise(location),
                Operation.measure(0, key="m", basis="Z"),
                Operation.detector(("m",), detector_id=0),
                Operation.observable_include(0, ("m",)),
            ),
        )
        sampler = compile_native_sampler(
            circuit,
            observables=(LogicalObservable(id=0, measurement_keys=("m",)),),
        )
        for mode, expected_calls in ((0, (1, 0, 0)), (2, (0, 1, 0)), (1, (0, 0, 1))):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                result = sampler.estimate(shots=16, seed=456, decoder=fixture)
                self.assertEqual(result.mean_loss, 0.0)
                self.assertEqual(fixture.callback_calls, expected_calls)
                self.assertEqual(fixture.worker_drops, 1)

    def test_detailed_collection_and_hotspot_use_each_v4_preference(self):
        sampler = compile_native_dem_sampler(self.dem())
        for mode, selected_index in ((0, 0), (2, 1), (1, 2)):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                tasks = (
                    {
                        "task_id": f"abi-v4-{mode}",
                        "strong_id": f"abi-v4-{mode}-strong",
                        "sampling_id": f"abi-v4-{mode}-sampling",
                        "sampler": sampler,
                        "decoder": fixture,
                        "metadata_json": "{}",
                        "max_shots": 8,
                        "min_shots": 8,
                        "batch_size": 4,
                    },
                )

                (stats,) = _native._collect_dem_logical_error_stats_many(
                    tasks,
                    num_workers=1,
                    seed=789,
                    count_observable_error_combos=True,
                    count_detection_events=True,
                )
                (hotspot,) = _native._collect_dem_hotspots_many(
                    tasks,
                    num_workers=1,
                    seed=789,
                    count_observable_error_combos=True,
                    count_detection_events=True,
                )

                self.assertEqual(stats["errors"], 0)
                self.assertEqual(stats["custom_counts"]["detection_events"], 8)
                self.assertEqual(stats["custom_counts"]["detectors_checked"], 8)
                self.assertEqual(hotspot["stats"]["errors"], 0)
                self.assertEqual(hotspot["edge_sensitivities"], (0.0,))
                callback_calls = fixture.callback_calls
                self.assertGreater(callback_calls[selected_index], 0)
                self.assertEqual(sum(callback_calls), callback_calls[selected_index])
                self.assertGreaterEqual(fixture.creates, 2)
                self.assertEqual(fixture.worker_drops, fixture.creates)


if __name__ == "__main__":
    unittest.main()
