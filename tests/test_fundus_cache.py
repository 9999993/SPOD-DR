"""Cache validation and reuse tests using synthetic images only."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


SCRIPT = (Path(__file__).resolve().parents[1] / 'data/preprocessing/generic/scripts'
          / 'build_diabetic_retinopathy_fundus_cache.py')
spec = importlib.util.spec_from_file_location('fundus_cache_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class FundusCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.image = self.root / 'image.png'
        self.cache = self.root / 'cache.npy'
        Image.new('RGB', (12, 8), (128, 128, 128)).save(self.image)

    def process(self):
        return builder._process_single_job(
            image_path=str(self.image), cache_path=str(self.cache),
            input_size=8, normalization='imagenet', overwrite=False)[0]

    def assert_rebuilt(self):
        self.assertEqual(self.process(), 'written')
        result = np.load(self.cache, allow_pickle=False)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(result.shape, (3, 8, 8))
        self.assertTrue(np.isfinite(result).all())
        # A uniform gray image has these independently calculated channel values.
        for channel, mean, std in [(0, .485, .229), (1, .456, .224), (2, .406, .225)]:
            np.testing.assert_allclose(result[channel], (128 / 255 - mean) / std, atol=1e-6)

    def test_valid_float32_cache_is_reused_without_rewriting(self):
        np.save(self.cache, np.full((3, 8, 8), 0.25, dtype=np.float32))
        before = self.cache.read_bytes()
        self.assertEqual(self.process(), 'skipped')
        self.assertEqual(self.cache.read_bytes(), before)

    def test_wrong_dtype_is_rebuilt_instead_of_cast_for_validation(self):
        for dtype in [np.float64, np.int16]:
            with self.subTest(dtype=dtype):
                np.save(self.cache, np.zeros((3, 8, 8), dtype=dtype))
                self.assert_rebuilt()

    def test_nonfinite_values_are_rebuilt(self):
        for value in [np.nan, np.inf, -np.inf]:
            with self.subTest(value=value):
                array = np.zeros((3, 8, 8), dtype=np.float32)
                array[0, 0, 0] = value
                np.save(self.cache, array)
                self.assert_rebuilt()

    def test_wrong_shape_is_rebuilt(self):
        np.save(self.cache, np.zeros((8, 8, 3), dtype=np.float32))
        self.assert_rebuilt()

    def test_corrupt_file_is_rebuilt(self):
        self.cache.write_bytes(b'not an NPY file')
        self.assert_rebuilt()

    def test_object_array_is_rebuilt_without_loading_pickle(self):
        np.save(self.cache, np.full((3, 8, 8), None, dtype=object))
        self.assert_rebuilt()

    def test_missing_cache_is_created(self):
        self.assert_rebuilt()


if __name__ == '__main__':
    unittest.main()
