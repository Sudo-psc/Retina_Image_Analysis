"""
Unit tests for the data module.

This module contains tests for dataset loading, preprocessing,
and data validation functionality.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Mock imports for testing when packages aren't installed
try:
    import cv2
    import torch

    TORCH_AVAILABLE = True
    CV2_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    CV2_AVAILABLE = False


@pytest.fixture
def temp_image():
    """Create a temporary test image."""
    # Create a simple RGB image
    image_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        image.save(f.name)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def temp_dataset_dir():
    """Create a temporary dataset directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create class directories
        class_dirs = ["normal", "diabetic_retinopathy", "glaucoma"]
        for class_name in class_dirs:
            class_dir = temp_path / class_name
            class_dir.mkdir()

            # Create sample images in each class
            for i in range(3):
                image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                image = Image.fromarray(image_array)
                image.save(class_dir / f"image_{i}.jpg")

        yield temp_path


class TestImagePreprocessor:
    """Test cases for ImagePreprocessor class."""

    def test_init(self):
        """Test ImagePreprocessor initialization."""
        # Skip if OpenCV not available
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        assert preprocessor.target_size == (512, 512)
        assert preprocessor.normalize is True
        assert preprocessor.enhance_contrast is True

    def test_load_image(self, temp_image):
        """Test image loading functionality."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        image = ImagePreprocessor.load_image(temp_image)
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3
        assert image.shape[2] == 3  # RGB channels

    def test_load_nonexistent_image(self):
        """Test loading non-existent image raises error."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        with pytest.raises(FileNotFoundError):
            ImagePreprocessor.load_image("nonexistent.jpg")

    def test_assess_image_quality(self, temp_image):
        """Test image quality assessment."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        image = ImagePreprocessor.load_image(temp_image)

        metrics = preprocessor.assess_image_quality(image)

        expected_keys = ["sharpness", "contrast", "brightness", "snr", "dynamic_range"]
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (int, float, np.integer, np.floating))

    def test_resize_image(self):
        """Test image resizing functionality."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        # Create test image
        image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)

        resized = ImagePreprocessor.resize_image(image, (64, 64))

        assert resized.shape == (64, 64, 3)

    def test_normalize_pixels(self):
        """Test pixel normalization."""
        if not CV2_AVAILABLE:
            pytest.skip("OpenCV not available")

        from src.data.preprocessing import ImagePreprocessor

        # Create test image with known values
        image = np.array([[[0, 127, 255]]], dtype=np.uint8)

        normalized = ImagePreprocessor.normalize_pixels(image)

        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
        np.testing.assert_almost_equal(normalized[0, 0], [0.0, 127 / 255, 1.0])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
class TestRetinaDataset:
    """Test cases for RetinaDataset class."""

    def test_init_from_directory(self, temp_dataset_dir):
        """Test dataset initialization from directory structure."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir)

        assert len(dataset) > 0
        assert hasattr(dataset, "annotations")
        assert "image_path" in dataset.annotations.columns
        assert "label" in dataset.annotations.columns

    def test_getitem(self, temp_dataset_dir):
        """Test dataset item retrieval."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir)

        sample = dataset[0]

        assert isinstance(sample, dict)
        assert "image" in sample
        assert "label" in sample
        assert "image_path" in sample
        assert "index" in sample

    def test_len(self, temp_dataset_dir):
        """Test dataset length."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir)

        assert len(dataset) == len(dataset.annotations)

    def test_get_class_weights(self, temp_dataset_dir):
        """Test class weights calculation."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir)
        weights = dataset.get_class_weights()

        assert torch.is_tensor(weights)
        assert len(weights) > 0

    def test_get_dataset_stats(self, temp_dataset_dir):
        """Test dataset statistics."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir)
        stats = dataset.get_dataset_stats()

        expected_keys = ["total_samples", "mode", "dataset_type", "image_size"]
        for key in expected_keys:
            assert key in stats


class TestConfigValidation:
    """Test configuration validation."""

    def test_config_file_exists(self):
        """Test that configuration file exists."""
        config_path = Path("configs/config.yaml")
        assert config_path.exists(), "Configuration file not found"

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists."""
        pyproject_path = Path("pyproject.toml")
        assert pyproject_path.exists(), "pyproject.toml not found"


class TestProjectStructure:
    """Test project structure and file organization."""

    def test_source_directory_structure(self):
        """Test that source directory has expected structure."""
        src_path = Path("src")
        assert src_path.exists(), "Source directory not found"

        expected_modules = ["data", "models", "training", "inference", "api", "utils"]
        for module in expected_modules:
            module_path = src_path / module
            assert module_path.exists(), f"Module {module} not found"

    def test_data_directory_structure(self):
        """Test that data directory has expected structure."""
        data_path = Path("data")
        assert data_path.exists(), "Data directory not found"

        expected_subdirs = ["raw", "processed", "annotations"]
        for subdir in expected_subdirs:
            subdir_path = data_path / subdir
            assert subdir_path.exists(), f"Data subdirectory {subdir} not found"

    def test_tests_directory_exists(self):
        """Test that tests directory exists."""
        tests_path = Path("tests")
        assert tests_path.exists(), "Tests directory not found"

    def test_docs_directory_exists(self):
        """Test that docs directory exists."""
        docs_path = Path("docs")
        assert docs_path.exists(), "Docs directory not found"


class TestDocumentation:
    """Test documentation files."""

    def test_readme_exists(self):
        """Test that README file exists."""
        readme_path = Path("README.md")
        assert readme_path.exists(), "README.md not found"

    def test_contributing_guide_exists(self):
        """Test that contributing guide exists."""
        contributing_path = Path("CONTRIBUTING.md")
        assert contributing_path.exists(), "CONTRIBUTING.md not found"

    def test_license_exists(self):
        """Test that license file exists."""
        license_path = Path("LICENSE")
        assert license_path.exists(), "LICENSE file not found"


# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.skipif(
        not (TORCH_AVAILABLE and CV2_AVAILABLE),
        reason="Required packages not available",
    )
    def test_end_to_end_preprocessing(self, temp_image):
        """Test end-to-end preprocessing pipeline."""
        from src.data.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor(target_size=(128, 128))
        result = preprocessor.preprocess(temp_image)

        assert "processed_image" in result
        assert "quality_metrics" in result
        assert result["processed_image"].shape == (128, 128, 3)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_dataset_preprocessing_integration(self, temp_dataset_dir):
        """Test integration between dataset and preprocessing."""
        from src.data.dataset import RetinaDataset

        dataset = RetinaDataset(temp_dataset_dir, image_size=(64, 64))

        # Test that we can load samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            assert sample["image"].shape[1:] == (64, 64)  # Exclude batch dimension
