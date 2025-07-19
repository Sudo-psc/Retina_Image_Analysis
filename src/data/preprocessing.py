"""
Image preprocessing utilities for retinal image analysis.

This module provides functions for preprocessing retinal images including
normalization, enhancement, and quality assessment.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImagePreprocessor:
    """
    Image preprocessing pipeline for retinal images.

    This class provides various preprocessing techniques specifically
    designed for retinal fundus images.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        normalize: bool = True,
        enhance_contrast: bool = True,
        remove_artifacts: bool = True,
    ) -> None:
        """
        Initialize the image preprocessor.

        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize pixel values
            enhance_contrast: Whether to enhance image contrast
            remove_artifacts: Whether to remove imaging artifacts
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.remove_artifacts = remove_artifacts

    def preprocess(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        mask_optic_disc: bool = False,
    ) -> Dict[str, Any]:
        """
        Apply complete preprocessing pipeline to an image.

        Args:
            image: Input image (path, array, or PIL Image)
            mask_optic_disc: Whether to detect and mask optic disc

        Returns:
            Dictionary containing processed image and metadata
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = self.load_image(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)

        original_shape = image.shape

        # Store original for comparison
        original_image = image.copy()

        # Preprocessing steps
        processed_image = image.copy()

        # 1. Quality assessment
        quality_metrics = self.assess_image_quality(processed_image)

        # 2. Remove artifacts and noise
        if self.remove_artifacts:
            processed_image = self.remove_artifacts_func(processed_image)

        # 3. Enhance contrast
        if self.enhance_contrast:
            processed_image = self.enhance_contrast_func(processed_image)

        # 4. Normalize illumination
        processed_image = self.normalize_illumination(processed_image)

        # 5. Resize to target size
        processed_image = self.resize_image(processed_image, self.target_size)

        # 6. Normalize pixel values
        if self.normalize:
            processed_image = self.normalize_pixels(processed_image)

        # 7. Optional optic disc masking
        optic_disc_mask = None
        if mask_optic_disc:
            optic_disc_mask = self.detect_optic_disc(processed_image)

        return {
            "processed_image": processed_image,
            "original_image": original_image,
            "original_shape": original_shape,
            "quality_metrics": quality_metrics,
            "optic_disc_mask": optic_disc_mask,
            "preprocessing_config": {
                "target_size": self.target_size,
                "normalize": self.normalize,
                "enhance_contrast": self.enhance_contrast,
                "remove_artifacts": self.remove_artifacts,
            },
        }

    @staticmethod
    def load_image(image_path: Union[str, Path]) -> np.ndarray:
        """Load image from file path."""
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load with OpenCV for better color handling
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Unable to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """
        Assess image quality metrics.

        Args:
            image: Input image

        Returns:
            Dictionary of quality metrics
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Calculate metrics
        metrics = {}

        # Sharpness (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        metrics["sharpness"] = laplacian.var()

        # Contrast (standard deviation)
        metrics["contrast"] = gray.std()

        # Brightness (mean pixel value)
        metrics["brightness"] = gray.mean()

        # Signal-to-noise ratio estimate
        kernel = np.ones((5, 5), np.float32) / 25
        filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        noise = gray.astype(np.float32) - filtered
        signal = filtered.mean()
        noise_level = noise.std()
        metrics["snr"] = signal / noise_level if noise_level > 0 else float("inf")

        # Dynamic range
        metrics["dynamic_range"] = gray.max() - gray.min()

        return metrics

    def remove_artifacts_func(self, image: np.ndarray) -> np.ndarray:
        """
        Remove common imaging artifacts from retinal images.

        Args:
            image: Input image

        Returns:
            Image with artifacts removed
        """
        # Median filtering to remove salt-and-pepper noise
        denoised = cv2.medianBlur(image, 3)

        # Gaussian blur to reduce high-frequency noise
        denoised = cv2.GaussianBlur(denoised, (3, 3), 0.5)

        # Morphological opening to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        denoised = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

        return denoised

    def enhance_contrast_func(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.

        Args:
            image: Input image

        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])

        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return enhanced

    def normalize_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize illumination across the image.

        Args:
            image: Input image

        Returns:
            Illumination-normalized image
        """
        # Convert to grayscale for illumination estimation
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Estimate background illumination using morphological opening
        kernel_size = max(image.shape) // 10
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        # Smooth the background
        background = cv2.GaussianBlur(
            background, (kernel_size // 2 * 2 + 1, kernel_size // 2 * 2 + 1), 0
        )

        # Normalize each channel
        normalized = np.zeros_like(image, dtype=np.float32)

        for i in range(image.shape[2]):
            channel = image[:, :, i].astype(np.float32)
            # Avoid division by zero
            background_norm = background.astype(np.float32) + 1e-7
            normalized[:, :, i] = channel / background_norm * background.mean()

        # Clip to valid range and convert back to uint8
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        return normalized

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
    ) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.

        Args:
            image: Input image
            target_size: Target size (height, width)
            interpolation: Interpolation method

        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_h, target_w = target_size

        # Calculate scaling factor
        scale = min(target_h / h, target_w / w)

        # Calculate new dimensions
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        # Pad to target size if necessary
        if new_h != target_h or new_w != target_w:
            # Calculate padding
            pad_h = target_h - new_h
            pad_w = target_w - new_w

            # Pad symmetrically
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            # Pad with black pixels
            resized = cv2.copyMakeBorder(
                resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

        return resized

    @staticmethod
    def normalize_pixels(image: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.

        Args:
            image: Input image

        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0

    def detect_optic_disc(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect optic disc location in retinal image.

        Args:
            image: Input retinal image

        Returns:
            Binary mask of optic disc region
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Enhance bright regions (optic disc is typically bright)
        enhanced = cv2.equalizeHist(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (21, 21), 0)

        # Threshold to find bright regions
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Find largest contour (likely optic disc)
        largest_contour = max(contours, key=cv2.contourArea)

        # Create mask
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [largest_contour], 255)

        return mask

    def batch_preprocess(
        self,
        image_paths: List[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        save_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Preprocess a batch of images.

        Args:
            image_paths: List of image paths
            output_dir: Directory to save processed images
            save_metadata: Whether to save preprocessing metadata

        Returns:
            List of preprocessing results
        """
        results = []

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for i, image_path in enumerate(image_paths):
            try:
                # Preprocess image
                result = self.preprocess(image_path)

                # Add path info
                result["input_path"] = str(image_path)

                # Save processed image if output directory provided
                if output_dir:
                    output_path = output_dir / f"processed_{Path(image_path).name}"
                    processed_image = (result["processed_image"] * 255).astype(np.uint8)
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR),
                    )
                    result["output_path"] = str(output_path)

                results.append(result)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({"input_path": str(image_path), "error": str(e)})

        return results
