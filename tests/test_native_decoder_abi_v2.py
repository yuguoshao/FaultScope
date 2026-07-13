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


CAPSULE_V2 = b"faultscope.native_decoder_plugin.v2"
CAPSULE_V1 = b"faultscope.native_decoder_plugin.v1"


class _FixtureDecoder:
    def __init__(self, library, mode, capsule_name=CAPSULE_V2):
        self._library = library
        self._pointer = library.fs_fixture_new(mode)
        self._capsule_name = ctypes.create_string_buffer(capsule_name)
        pointer = self._pointer
        drop_factory = library.fs_fixture_drop_factory

        @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
        def capsule_destructor(_capsule):
            drop_factory(pointer)

        self._capsule_destructor = capsule_destructor
        self._capsule = self._new_capsule(
            library.fs_fixture_descriptor(self._pointer),
            ctypes.cast(self._capsule_name, ctypes.c_char_p),
            ctypes.cast(self._capsule_destructor, ctypes.c_void_p),
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
            if self.factory_drops != 1:
                raise AssertionError(
                    f"fixture capsule dropped factory state {self.factory_drops} times"
                )
            self._library.fs_fixture_free(self._pointer)
            self._pointer = None


class NativeDecoderAbiV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tempdir = tempfile.TemporaryDirectory(prefix="faultscope-abi-v2-")
        source = pathlib.Path(__file__).with_name("native_decoder_v2_fixture.c")
        library_path = pathlib.Path(cls._tempdir.name) / "native_decoder_v2_fixture.so"
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

    def fixture(self, mode, capsule_name=CAPSULE_V2):
        fixture = _FixtureDecoder(self.library, mode, capsule_name)
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

    def test_exact_v2_capsule_creates_distinct_temporary_workers(self):
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

    def test_v1_capsule_name_is_rejected_with_v2_context(self):
        fixture = self.fixture(0, CAPSULE_V1)
        with self.assertRaisesRegex(ValueError, r"v2"):
            NativeCompositeDecoder((fixture,))

    def test_invalid_v2_factory_descriptors_are_rejected(self):
        cases = {
            10: r"expected ABI v2",
            11: r"ABI v2.*size",
            12: r"ABI v2.*thread-safe",
            13: r"ABI v2.*null factory state",
            14: r"ABI v2.*required callbacks",
            15: r"ABI v2.*required callbacks",
            16: r"ABI v2.*required callbacks",
            17: r"ABI v2.*required callbacks",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, message):
                    NativeCompositeDecoder((fixture,))

    def test_invalid_worker_descriptors_and_output_shape_are_rejected(self):
        cases = {
            5: (r"ABI v2.*worker descriptor size", 1),
            6: (r"ABI v2.*unsafe non-null partial worker state", 0),
            7: (r"ABI v2.*missing required callbacks", 1),
            8: (r"ABI v2.*null worker state", 0),
            9: (r"ABI v2.*invalid mask output shape", 1),
            20: (r"ABI v2.*inconsistent decoder metadata", 1),
        }
        for mode, (message, expected_drops) in cases.items():
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                decoder = NativeCompositeDecoder((fixture,))
                with self.assertRaisesRegex(ValueError, message):
                    decoder.decode_batch_masks(self.batch())
                self.assertEqual(fixture.worker_drops, expected_drops)
                del decoder
                gc.collect()

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

    def test_dem_estimate_uses_event_packed_and_mask_workers(self):
        for mode in (1, 2, 0):
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                result = compile_native_dem_sampler(self.dem()).estimate(
                    shots=16,
                    seed=123,
                    decoder=fixture,
                )
                self.assertEqual(result.mean_loss, 0.0)
                self.assertEqual(fixture.creates, 1)
                self.assertEqual(fixture.worker_drops, 1)

    def test_packed_and_event_invalid_output_shapes_are_rejected(self):
        cases = {
            18: r"invalid packed-row output shape",
            19: r"invalid detector-event output shape",
        }
        for mode, message in cases.items():
            with self.subTest(mode=mode):
                fixture = self.fixture(mode)
                with self.assertRaisesRegex(ValueError, message):
                    compile_native_dem_sampler(self.dem()).estimate(
                        shots=4,
                        seed=321,
                        decoder=fixture,
                        aggregate_hotspots=False,
                    )
                self.assertEqual(fixture.worker_drops, 1)

    def test_packed_estimate_uses_temporary_mask_worker(self):
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
        fixture = self.fixture(0)

        result = compile_native_sampler(circuit).estimate(
            shots=16,
            seed=456,
            decoder=fixture,
        )

        self.assertEqual(result.mean_loss, 0.0)
        self.assertEqual(fixture.creates, 1)
        self.assertEqual(fixture.worker_drops, 1)

    def test_collection_task_uses_factory_worker(self):
        fixture = self.fixture(0)
        sampler = compile_native_dem_sampler(self.dem())
        tasks = ({
            "task_id": "abi-v2",
            "strong_id": "abi-v2-strong",
            "sampler": sampler,
            "decoder": fixture,
            "metadata_json": "{}",
            "max_shots": 8,
            "min_shots": 8,
            "batch_size": 4,
        },)

        stats = _native._collect_dem_logical_error_stats_many(tasks, num_workers=1, seed=789)

        self.assertEqual(len(stats), 1)
        self.assertGreaterEqual(fixture.creates, 1)
        self.assertEqual(fixture.worker_drops, fixture.creates)


if __name__ == "__main__":
    unittest.main()
