"""
Data validation module for retinal image analysis.

This module provides comprehensive validation and quality control
for retinal imaging datasets.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


class ImageQualityValidator:
    """
    Validator for retinal image quality assessment.
    
    This class provides methods to assess and validate the quality
    of retinal fundus images before they are used for training.
    """

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (256, 256),
        max_resolution: Tuple[int, int] = (4096, 4096),
        min_brightness: float = 10.0,
        max_brightness: float = 245.0,
        min_contrast: float = 20.0,
        blur_threshold: float = 100.0,
        noise_threshold: float = 50.0,
    ) -> None:
        """
        Initialize the image quality validator.

        Args:
            min_resolution: Minimum acceptable image resolution
            max_resolution: Maximum acceptable image resolution
            min_brightness: Minimum acceptable mean brightness
            max_brightness: Maximum acceptable mean brightness
            min_contrast: Minimum acceptable contrast (std of pixel values)
            blur_threshold: Threshold for blur detection (Laplacian variance)
            noise_threshold: Threshold for noise detection
        """
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast
        self.blur_threshold = blur_threshold
        self.noise_threshold = noise_threshold

    def validate_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Validate a single image and return quality metrics.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing validation results and quality metrics
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {
                    "valid": False,
                    "error": "Could not load image",
                    "path": str(image_path),
                }

            # Convert to grayscale for some analyses
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform validation checks
            results = {
                "valid": True,
                "path": str(image_path),
                "warnings": [],
                "metrics": {},
            }

            # Check resolution
            height, width = image.shape[:2]
            results["metrics"]["resolution"] = (width, height)
            
            if width < self.min_resolution[0] or height < self.min_resolution[1]:
                results["valid"] = False
                results["warnings"].append(f"Resolution too low: {width}x{height}")
            
            if width > self.max_resolution[0] or height > self.max_resolution[1]:
                results["warnings"].append(f"Resolution very high: {width}x{height}")

            # Check brightness
            mean_brightness = np.mean(gray)
            results["metrics"]["brightness"] = float(mean_brightness)
            
            if mean_brightness < self.min_brightness:
                results["warnings"].append(f"Image too dark: {mean_brightness:.1f}")
            elif mean_brightness > self.max_brightness:
                results["warnings"].append(f"Image too bright: {mean_brightness:.1f}")

            # Check contrast
            contrast = np.std(gray)
            results["metrics"]["contrast"] = float(contrast)
            
            if contrast < self.min_contrast:
                results["warnings"].append(f"Low contrast: {contrast:.1f}")

            # Check for blur
            blur_metric = self._calculate_blur_metric(gray)
            results["metrics"]["blur"] = float(blur_metric)
            
            if blur_metric < self.blur_threshold:
                results["warnings"].append(f"Image may be blurry: {blur_metric:.1f}")

            # Check for noise
            noise_metric = self._calculate_noise_metric(gray)
            results["metrics"]["noise"] = float(noise_metric)
            
            if noise_metric > self.noise_threshold:
                results["warnings"].append(f"High noise detected: {noise_metric:.1f}")

            # Check for artifacts
            artifact_score = self._detect_artifacts(image)
            results["metrics"]["artifacts"] = float(artifact_score)

            # Check for proper retinal structure
            structure_score = self._validate_retinal_structure(gray)
            results["metrics"]["structure"] = float(structure_score)

            return results

        except Exception as e:
            return {
                "valid": False,
                "error": str(e),
                "path": str(image_path),
            }

    def _calculate_blur_metric(self, gray_image: np.ndarray) -> float:
        """Calculate blur metric using Laplacian variance."""
        return cv2.Laplacian(gray_image, cv2.CV_64F).var()

    def _calculate_noise_metric(self, gray_image: np.ndarray) -> float:
        """Calculate noise metric using median filtering."""
        median_filtered = cv2.medianBlur(gray_image, 5)
        noise = np.mean(np.abs(gray_image.astype(np.float32) - median_filtered.astype(np.float32)))
        return noise

    def _detect_artifacts(self, image: np.ndarray) -> float:
        """Detect imaging artifacts and return artifact score."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect extreme dark or bright regions
        very_dark = np.sum(gray < 10) / gray.size
        very_bright = np.sum(gray > 245) / gray.size
        
        # Detect sudden intensity changes (possible artifacts)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Combine metrics
        artifact_score = (very_dark + very_bright) * 100 + edge_density * 50
        return artifact_score

    def _validate_retinal_structure(self, gray_image: np.ndarray) -> float:
        """Validate presence of retinal structures."""
        # Simple circular structure detection (approximate)
        height, width = gray_image.shape
        center_x, center_y = width // 2, height // 2
        
        # Check for circular intensity patterns typical of fundus images
        y, x = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Calculate radial intensity profile
        max_radius = min(center_x, center_y) * 0.8
        mask = dist_from_center <= max_radius
        
        if np.sum(mask) == 0:
            return 0.0
        
        center_intensity = np.mean(gray_image[mask])
        edge_intensity = np.mean(gray_image[~mask])
        
        # Good retinal images typically have higher intensity in center
        structure_score = max(0, center_intensity - edge_intensity)
        return structure_score


class DatasetValidator:
    """
    Validator for entire retinal image datasets.
    
    This class provides methods to validate dataset structure,
    annotations, and overall data quality.
    """

    def __init__(self, quality_validator: Optional[ImageQualityValidator] = None) -> None:
        """
        Initialize the dataset validator.

        Args:
            quality_validator: Image quality validator instance
        """
        self.quality_validator = quality_validator or ImageQualityValidator()

    def validate_dataset(
        self,
        data_dir: Union[str, Path],
        annotations_file: Optional[Union[str, Path]] = None,
        sample_fraction: float = 0.1,
        output_report: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Validate an entire dataset.

        Args:
            data_dir: Directory containing the images
            annotations_file: Path to annotations CSV file
            sample_fraction: Fraction of images to validate (for large datasets)
            output_report: Path to save validation report

        Returns:
            Dictionary containing validation results
        """
        data_dir = Path(data_dir)
        logger.info(f"Validating dataset in {data_dir}")

        results = {
            "dataset_path": str(data_dir),
            "total_images": 0,
            "validated_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "warnings": [],
            "errors": [],
            "quality_metrics": {},
            "annotation_validation": {},
        }

        try:
            # Find all image files
            image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(data_dir.rglob(f"*{ext}"))
                image_files.extend(data_dir.rglob(f"*{ext.upper()}"))

            results["total_images"] = len(image_files)
            
            if len(image_files) == 0:
                results["errors"].append("No image files found")
                return results

            # Sample images for validation
            if sample_fraction < 1.0:
                import random
                random.seed(42)
                num_samples = max(1, int(len(image_files) * sample_fraction))
                image_files = random.sample(image_files, num_samples)

            # Validate images
            valid_count = 0
            invalid_count = 0
            all_metrics = []

            for image_file in image_files:
                validation_result = self.quality_validator.validate_image(image_file)
                
                if validation_result.get("valid", False):
                    valid_count += 1
                    if "metrics" in validation_result:
                        all_metrics.append(validation_result["metrics"])
                else:
                    invalid_count += 1
                    results["errors"].append(
                        f"Invalid image {image_file}: {validation_result.get('error', 'Unknown error')}"
                    )

                # Collect warnings
                if "warnings" in validation_result:
                    for warning in validation_result["warnings"]:
                        results["warnings"].append(f"{image_file}: {warning}")

            results["validated_images"] = len(image_files)
            results["valid_images"] = valid_count
            results["invalid_images"] = invalid_count

            # Calculate aggregate quality metrics
            if all_metrics:
                results["quality_metrics"] = self._calculate_aggregate_metrics(all_metrics)

            # Validate annotations if provided
            if annotations_file:
                annotation_results = self._validate_annotations(
                    annotations_file, [f.name for f in image_files]
                )
                results["annotation_validation"] = annotation_results

            # Save report if requested
            if output_report:
                self._save_validation_report(results, output_report)

            logger.info(f"Dataset validation complete: {valid_count}/{len(image_files)} images valid")
            return results

        except Exception as e:
            results["errors"].append(f"Dataset validation failed: {str(e)}")
            logger.error(f"Dataset validation error: {e}")
            return results

    def _calculate_aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate statistics from individual image metrics."""
        if not metrics_list:
            return {}

        # Collect all numeric metrics
        numeric_metrics = {}
        for metrics in metrics_list:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_metrics:
                        numeric_metrics[key] = []
                    numeric_metrics[key].append(value)

        # Calculate statistics
        aggregate = {}
        for metric_name, values in numeric_metrics.items():
            aggregate[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values)),
            }

        return aggregate

    def _validate_annotations(
        self, annotations_file: Union[str, Path], image_filenames: List[str]
    ) -> Dict[str, Any]:
        """Validate annotation file."""
        try:
            # Load annotations
            df = pd.read_csv(annotations_file)
            
            results = {
                "annotation_file": str(annotations_file),
                "total_annotations": len(df),
                "missing_images": [],
                "missing_annotations": [],
                "duplicate_annotations": [],
                "invalid_labels": [],
                "column_validation": {},
            }

            # Check for required columns
            required_columns = ["image_id", "label"]  # Basic requirements
            for col in required_columns:
                if col not in df.columns:
                    results["column_validation"][col] = "Missing"
                else:
                    results["column_validation"][col] = "Present"

            if "image_id" in df.columns:
                # Check for missing images
                image_set = set(image_filenames)
                annotation_images = set(df["image_id"].astype(str))
                
                results["missing_images"] = list(annotation_images - image_set)
                results["missing_annotations"] = list(image_set - annotation_images)

                # Check for duplicate annotations
                duplicates = df[df.duplicated("image_id", keep=False)]
                if not duplicates.empty:
                    results["duplicate_annotations"] = duplicates["image_id"].tolist()

            # Validate labels if present
            if "label" in df.columns:
                # Check for invalid/missing labels
                invalid_labels = df[df["label"].isna() | (df["label"] == "")]
                if not invalid_labels.empty:
                    results["invalid_labels"] = invalid_labels["image_id"].tolist()

            return results

        except Exception as e:
            return {"error": f"Annotation validation failed: {str(e)}"}

    def _save_validation_report(
        self, results: Dict[str, Any], output_path: Union[str, Path]
    ) -> None:
        """Save validation report to file."""
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {output_path}")


def validate_retina_dataset(
    data_dir: Union[str, Path],
    annotations_file: Optional[Union[str, Path]] = None,
    quality_config: Optional[Dict[str, Any]] = None,
    sample_fraction: float = 0.1,
    output_report: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to validate a retinal image dataset.

    Args:
        data_dir: Directory containing the images
        annotations_file: Path to annotations CSV file
        quality_config: Configuration for quality validator
        sample_fraction: Fraction of images to validate
        output_report: Path to save validation report

    Returns:
        Dictionary containing validation results
    """
    # Create quality validator
    if quality_config:
        quality_validator = ImageQualityValidator(**quality_config)
    else:
        quality_validator = ImageQualityValidator()

    # Create dataset validator
    dataset_validator = DatasetValidator(quality_validator)

    # Validate dataset
    return dataset_validator.validate_dataset(
        data_dir=data_dir,
        annotations_file=annotations_file,
        sample_fraction=sample_fraction,
        output_report=output_report,
    )


# Default configurations for different validation scenarios
VALIDATION_CONFIGS = {
    "strict": {
        "min_resolution": (512, 512),
        "min_brightness": 30.0,
        "max_brightness": 220.0,
        "min_contrast": 30.0,
        "blur_threshold": 150.0,
        "noise_threshold": 30.0,
    },
    "lenient": {
        "min_resolution": (256, 256),
        "min_brightness": 10.0,
        "max_brightness": 245.0,
        "min_contrast": 15.0,
        "blur_threshold": 50.0,
        "noise_threshold": 70.0,
    },
    "research": {
        "min_resolution": (384, 384),
        "min_brightness": 20.0,
        "max_brightness": 235.0,
        "min_contrast": 25.0,
        "blur_threshold": 100.0,
        "noise_threshold": 50.0,
    },
}
