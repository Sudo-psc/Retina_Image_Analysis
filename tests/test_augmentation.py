"""
Tests for data augmentation module.
"""

import pytest
import numpy as np
from PIL import Image

from src.data.augmentation import (
    RetinaAugmentation,
    MedicalAugmentation,
    get_retina_transforms,
    create_augmentation_pipeline,
    AUGMENTATION_CONFIGS,
)


class TestRetinaAugmentation:
    """Test cases for RetinaAugmentation class."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        aug = RetinaAugmentation()
        assert aug.rotation_range == 15.0
        assert aug.brightness_range == (0.8, 1.2)
        assert aug.horizontal_flip is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        aug = RetinaAugmentation(
            rotation_range=30.0,
            brightness_range=(0.5, 1.5),
            horizontal_flip=False,
        )
        assert aug.rotation_range == 30.0
        assert aug.brightness_range == (0.5, 1.5)
        assert aug.horizontal_flip is False

    def test_augment_pil_image(self):
        """Test augmentation with PIL Image input."""
        # Create test image
        image = Image.new("RGB", (512, 512), color="red")
        
        aug = RetinaAugmentation(
            rotation_range=0,  # Disable rotation for predictable test
            horizontal_flip=False,
            vertical_flip=False,
            gaussian_noise=False,
            gaussian_blur=False,
        )
        
        result = aug(image)
        assert isinstance(result, Image.Image)
        assert result.size == image.size

    def test_augment_numpy_array(self):
        """Test augmentation with numpy array input."""
        # Create test image
        image = np.random.rand(512, 512, 3).astype(np.float32)
        
        aug = RetinaAugmentation(
            rotation_range=0,
            horizontal_flip=False,
            vertical_flip=False,
            gaussian_noise=False,
            gaussian_blur=False,
        )
        
        result = aug(image)
        assert isinstance(result, np.ndarray)
        assert result.shape == image.shape

    def test_geometric_transforms(self):
        """Test geometric transformations."""
        image = Image.new("RGB", (256, 256), color="blue")
        
        aug = RetinaAugmentation(
            rotation_range=45.0,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=(0.8, 1.2),
        )
        
        # Test multiple times to ensure transforms are applied
        results = [aug(image) for _ in range(10)]
        
        # At least some results should be different (due to randomness)
        unique_results = set(str(np.array(img).tobytes()) for img in results)
        assert len(unique_results) > 1  # Should have some variation

    def test_color_transforms(self):
        """Test color transformations."""
        image = Image.new("RGB", (128, 128), color=(128, 128, 128))
        
        aug = RetinaAugmentation(
            rotation_range=0,
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_range=0.2,
        )
        
        result = aug(image)
        assert isinstance(result, Image.Image)
        assert result.size == image.size


class TestMedicalAugmentation:
    """Test cases for MedicalAugmentation class."""

    def test_simulate_camera_shift(self):
        """Test camera shift simulation."""
        image = np.random.rand(256, 256, 3).astype(np.uint8) * 255
        
        result = MedicalAugmentation.simulate_camera_shift(image, max_shift=10)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_simulate_illumination_variation(self):
        """Test illumination variation simulation."""
        image = np.random.rand(256, 256, 3).astype(np.uint8) * 255
        
        result = MedicalAugmentation.simulate_illumination_variation(image, intensity=0.3)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype

    def test_simulate_acquisition_artifacts(self):
        """Test acquisition artifacts simulation."""
        image = np.random.rand(256, 256, 3).astype(np.uint8) * 255
        
        # Test with high probability to ensure artifacts are added
        result = MedicalAugmentation.simulate_acquisition_artifacts(image, artifact_prob=1.0)
        
        assert result.shape == image.shape
        assert result.dtype == image.dtype


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_get_retina_transforms_train(self):
        """Test getting training transforms."""
        transforms = get_retina_transforms(mode="train", image_size=(512, 512))
        
        assert transforms is not None
        # Should include augmentation for training mode

    def test_get_retina_transforms_val(self):
        """Test getting validation transforms."""
        transforms = get_retina_transforms(mode="val", image_size=(384, 384))
        
        assert transforms is not None
        # Should not include augmentation for validation mode

    def test_get_retina_transforms_test(self):
        """Test getting test transforms."""
        transforms = get_retina_transforms(mode="test", image_size=(256, 256))
        
        assert transforms is not None

    def test_create_augmentation_pipeline(self):
        """Test creating augmentation pipeline from config."""
        config = {
            "rotation_range": 20.0,
            "brightness_range": (0.7, 1.3),
            "horizontal_flip": True,
        }
        
        pipeline = create_augmentation_pipeline(config)
        
        assert isinstance(pipeline, RetinaAugmentation)
        assert pipeline.rotation_range == 20.0
        assert pipeline.brightness_range == (0.7, 1.3)
        assert pipeline.horizontal_flip is True

    def test_augmentation_configs(self):
        """Test predefined augmentation configurations."""
        assert "baseline" in AUGMENTATION_CONFIGS
        assert "aggressive" in AUGMENTATION_CONFIGS
        assert "medical_focused" in AUGMENTATION_CONFIGS
        
        # Test that configs can create valid pipelines
        for config_name, config in AUGMENTATION_CONFIGS.items():
            pipeline = create_augmentation_pipeline(config)
            assert isinstance(pipeline, RetinaAugmentation)


class TestAugmentationIntegration:
    """Integration tests for augmentation pipeline."""

    def test_end_to_end_pipeline(self):
        """Test complete augmentation pipeline."""
        # Create test image
        image = Image.new("RGB", (512, 512), color=(100, 150, 200))
        
        # Create augmentation pipeline
        aug = RetinaAugmentation()
        
        # Apply augmentation multiple times
        results = []
        for _ in range(5):
            result = aug(image)
            assert isinstance(result, Image.Image)
            assert result.size == image.size
            results.append(result)
        
        # Results should show some variation due to randomness
        # (This is a statistical test and might rarely fail)
        pixel_arrays = [np.array(img) for img in results]
        variations = [np.std(arr) for arr in pixel_arrays]
        
        # At least some images should show variation
        assert any(var > 0 for var in variations)

    def test_augmentation_preserves_image_properties(self):
        """Test that augmentation preserves important image properties."""
        # Create test image with specific properties
        image_array = np.random.rand(256, 256, 3) * 255
        image = Image.fromarray(image_array.astype(np.uint8))
        
        aug = RetinaAugmentation()
        
        for _ in range(10):
            result = aug(image)
            
            # Should preserve basic properties
            assert isinstance(result, Image.Image)
            assert result.size == image.size
            assert result.mode == image.mode
            
            # Pixel values should be in valid range
            result_array = np.array(result)
            assert np.all(result_array >= 0)
            assert np.all(result_array <= 255)

    def test_augmentation_reproducibility(self):
        """Test augmentation reproducibility with fixed seed."""
        import random
        
        image = Image.new("RGB", (128, 128), color="green")
        aug = RetinaAugmentation()
        
        # Set seed and apply augmentation
        random.seed(42)
        np.random.seed(42)
        result1 = aug(image)
        
        # Reset seed and apply again
        random.seed(42)
        np.random.seed(42)
        result2 = aug(image)
        
        # Results should be identical with same seed
        assert np.array_equal(np.array(result1), np.array(result2))
