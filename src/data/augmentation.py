"""
Data augmentation module for retinal image analysis.

This module provides various data augmentation techniques specifically
designed for retinal fundus images to improve model generalization.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as F

logger = logging.getLogger(__name__)


class RetinaAugmentation:
    """
    Data augmentation pipeline for retinal images.
    
    This class provides medical imaging specific augmentations that preserve
    the clinical relevance of retinal fundus images.
    """

    def __init__(
        self,
        rotation_range: float = 15.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: float = 0.1,
        zoom_range: Tuple[float, float] = (0.9, 1.1),
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        gaussian_noise: bool = True,
        gaussian_blur: bool = True,
        elastic_transform: bool = True,
        preserve_optic_disc: bool = True,
    ) -> None:
        """
        Initialize the retinal augmentation pipeline.

        Args:
            rotation_range: Maximum rotation angle in degrees
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            saturation_range: Range for saturation adjustment
            hue_range: Range for hue adjustment
            zoom_range: Range for zoom/scale adjustment
            horizontal_flip: Whether to apply horizontal flipping
            vertical_flip: Whether to apply vertical flipping
            gaussian_noise: Whether to add gaussian noise
            gaussian_blur: Whether to apply gaussian blur
            elastic_transform: Whether to apply elastic deformation
            preserve_optic_disc: Whether to preserve optic disc region
        """
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.gaussian_noise = gaussian_noise
        self.gaussian_blur = gaussian_blur
        self.elastic_transform = elastic_transform
        self.preserve_optic_disc = preserve_optic_disc

    def __call__(self, image: Union[Image.Image, np.ndarray]) -> Union[Image.Image, np.ndarray]:
        """
        Apply augmentation pipeline to an image.

        Args:
            image: Input image (PIL Image or numpy array)

        Returns:
            Augmented image in the same format as input
        """
        # Convert to PIL Image if numpy array
        was_numpy = isinstance(image, np.ndarray)
        if was_numpy:
            if image.dtype == np.uint8:
                image = Image.fromarray(image)
            else:
                image = Image.fromarray((image * 255).astype(np.uint8))

        # Apply augmentations
        image = self._apply_geometric_transforms(image)
        image = self._apply_color_transforms(image)
        image = self._apply_noise_and_blur(image)

        # Convert back to numpy if needed
        if was_numpy:
            image = np.array(image)

        return image

    def _apply_geometric_transforms(self, image: Image.Image) -> Image.Image:
        """Apply geometric transformations."""
        # Random rotation
        if self.rotation_range > 0:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            image = F.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)

        # Random horizontal flip
        if self.horizontal_flip and random.random() > 0.5:
            image = F.hflip(image)

        # Random vertical flip (less common for retinal images)
        if self.vertical_flip and random.random() > 0.5:
            image = F.vflip(image)

        # Random zoom/scale
        if self.zoom_range != (1.0, 1.0):
            scale_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            width, height = image.size
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize and center crop back to original size
            image = F.resize(image, (new_height, new_width), interpolation=Image.BILINEAR)
            if scale_factor > 1.0:
                # Crop from center
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                image = F.crop(image, top, left, height, width)
            else:
                # Pad to original size
                padding = [
                    (width - new_width) // 2,
                    (height - new_height) // 2,
                    (width - new_width) - (width - new_width) // 2,
                    (height - new_height) - (height - new_height) // 2,
                ]
                image = F.pad(image, padding, fill=0)

        return image

    def _apply_color_transforms(self, image: Image.Image) -> Image.Image:
        """Apply color transformations."""
        # Random brightness
        if self.brightness_range != (1.0, 1.0):
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness_factor)

        # Random contrast
        if self.contrast_range != (1.0, 1.0):
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast_factor)

        # Random saturation
        if self.saturation_range != (1.0, 1.0):
            saturation_factor = random.uniform(self.saturation_range[0], self.saturation_range[1])
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(saturation_factor)

        # Random hue adjustment
        if self.hue_range > 0:
            hue_factor = random.uniform(-self.hue_range, self.hue_range)
            image = F.adjust_hue(image, hue_factor)

        return image

    def _apply_noise_and_blur(self, image: Image.Image) -> Image.Image:
        """Apply noise and blur effects."""
        # Random Gaussian blur
        if self.gaussian_blur and random.random() > 0.7:
            radius = random.uniform(0.5, 2.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        # Random Gaussian noise
        if self.gaussian_noise and random.random() > 0.7:
            image = self._add_gaussian_noise(image)

        return image

    def _add_gaussian_noise(self, image: Image.Image, mean: float = 0, std: float = 0.1) -> Image.Image:
        """Add Gaussian noise to image."""
        image_array = np.array(image).astype(np.float32) / 255.0
        noise = np.random.normal(mean, std, image_array.shape)
        noisy_image = image_array + noise
        noisy_image = np.clip(noisy_image, 0, 1)
        noisy_image = (noisy_image * 255).astype(np.uint8)
        return Image.fromarray(noisy_image)

    def _apply_elastic_transform(self, image: np.ndarray, alpha: float = 1, sigma: float = 50) -> np.ndarray:
        """
        Apply elastic deformation to image.
        
        Args:
            image: Input image as numpy array
            alpha: Intensity of deformation
            sigma: Smoothness of deformation
            
        Returns:
            Deformed image
        """
        shape = image.shape[:2]
        dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
        dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        mapx = np.float32(x + dx)
        mapy = np.float32(y + dy)

        return cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class MedicalAugmentation:
    """
    Medical imaging specific augmentation techniques.
    
    These augmentations are designed to simulate real-world variations
    in medical imaging while preserving diagnostic information.
    """

    @staticmethod
    def simulate_camera_shift(image: np.ndarray, max_shift: int = 20) -> np.ndarray:
        """Simulate camera position shift."""
        h, w = image.shape[:2]
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)

    @staticmethod
    def simulate_illumination_variation(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
        """Simulate illumination variations common in fundus imaging."""
        h, w = image.shape[:2]
        
        # Create radial gradient
        center_x, center_y = w // 2, h // 2
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distance
        max_dist = np.sqrt(center_x**2 + center_y**2)
        normalized_dist = dist_from_center / max_dist
        
        # Create illumination map
        illumination = 1 + intensity * (2 * normalized_dist - 1)
        illumination = np.clip(illumination, 0.5, 1.5)
        
        # Apply to image
        if len(image.shape) == 3:
            illumination = illumination[:, :, np.newaxis]
        
        result = image.astype(np.float32) * illumination
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def simulate_acquisition_artifacts(image: np.ndarray, artifact_prob: float = 0.1) -> np.ndarray:
        """Simulate common acquisition artifacts."""
        if random.random() > artifact_prob:
            return image
            
        h, w = image.shape[:2]
        result = image.copy()
        
        # Add random dark spots (simulating dust or scratches)
        num_spots = random.randint(1, 5)
        for _ in range(num_spots):
            spot_x = random.randint(0, w - 1)
            spot_y = random.randint(0, h - 1)
            spot_radius = random.randint(5, 20)
            cv2.circle(result, (spot_x, spot_y), spot_radius, (0, 0, 0), -1)
        
        return result


def get_retina_transforms(
    mode: str = "train",
    image_size: Tuple[int, int] = (512, 512),
    augmentation_strength: str = "medium",
) -> transforms.Compose:
    """
    Get standard transforms for retinal images.

    Args:
        mode: 'train', 'val', or 'test'
        image_size: Target image size
        augmentation_strength: 'light', 'medium', or 'heavy'

    Returns:
        Composed transforms
    """
    if mode == "train":
        augmentation_params = {
            "light": {
                "rotation_range": 5.0,
                "brightness_range": (0.9, 1.1),
                "contrast_range": (0.9, 1.1),
                "gaussian_noise": False,
                "elastic_transform": False,
            },
            "medium": {
                "rotation_range": 15.0,
                "brightness_range": (0.8, 1.2),
                "contrast_range": (0.8, 1.2),
                "gaussian_noise": True,
                "elastic_transform": False,
            },
            "heavy": {
                "rotation_range": 30.0,
                "brightness_range": (0.7, 1.3),
                "contrast_range": (0.7, 1.3),
                "gaussian_noise": True,
                "elastic_transform": True,
            },
        }
        
        params = augmentation_params.get(augmentation_strength, augmentation_params["medium"])
        
        transform_list = [
            transforms.Resize(image_size),
            RetinaAugmentation(**params),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        # Validation and test transforms (no augmentation)
        transform_list = [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    return transforms.Compose(transform_list)


def create_augmentation_pipeline(
    augmentation_config: Dict[str, Any]
) -> RetinaAugmentation:
    """
    Create augmentation pipeline from configuration.

    Args:
        augmentation_config: Configuration dictionary

    Returns:
        Configured augmentation pipeline
    """
    return RetinaAugmentation(**augmentation_config)


# Example configurations
AUGMENTATION_CONFIGS = {
    "baseline": {
        "rotation_range": 10.0,
        "brightness_range": (0.9, 1.1),
        "contrast_range": (0.9, 1.1),
        "horizontal_flip": True,
        "gaussian_noise": False,
        "elastic_transform": False,
    },
    "aggressive": {
        "rotation_range": 25.0,
        "brightness_range": (0.7, 1.3),
        "contrast_range": (0.7, 1.3),
        "saturation_range": (0.7, 1.3),
        "horizontal_flip": True,
        "vertical_flip": True,
        "gaussian_noise": True,
        "gaussian_blur": True,
        "elastic_transform": True,
    },
    "medical_focused": {
        "rotation_range": 15.0,
        "brightness_range": (0.8, 1.2),
        "contrast_range": (0.8, 1.2),
        "horizontal_flip": True,
        "gaussian_noise": True,
        "preserve_optic_disc": True,
    },
}
