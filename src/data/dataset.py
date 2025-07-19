"""
Dataset module for retinal image analysis.

This module provides PyTorch Dataset classes for loading and preprocessing
retinal images for machine learning tasks.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import lru_cache

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Constants
DEFAULT_IMAGE_SIZE = (512, 512)
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".tif"})
DEFAULT_TRANSFORM_PARAMS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "brightness": 0.2,
    "contrast": 0.2,
    "saturation": 0.2,
    "hue": 0.1,
    "rotation_degrees": 15
}


class RetinaDataset(Dataset):
    """
    PyTorch Dataset for retinal images.

    This dataset supports multiple retinal imaging datasets including:
    - DRIVE
    - STARE
    - Messidor
    - Kaggle Diabetic Retinopathy
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_file: Optional[Union[str, Path]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
        mode: str = "train",
        dataset_type: str = "classification",
    ) -> None:
        """
        Initialize the RetinaDataset.

        Args:
            data_dir: Directory containing the images
            annotations_file: Path to annotations CSV file
            transform: Optional transform to be applied on images
            target_transform: Optional transform to be applied on labels
            image_size: Target size for images (height, width)
            mode: Dataset mode ('train', 'val', 'test')
            dataset_type: Type of task ('classification', 'segmentation', 'detection')
        """
        self.data_dir = Path(data_dir)
        self.annotations_file = annotations_file
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.mode = mode
        self.dataset_type = dataset_type

        # Load annotations if provided
        if annotations_file and Path(annotations_file).exists():
            try:
                self.annotations = pd.read_csv(annotations_file)
                logger.info(f"Loaded {len(self.annotations)} annotations from {annotations_file}")
            except Exception as e:
                logger.error(f"Failed to load annotations from {annotations_file}: {e}")
                raise
        else:
            # If no annotations file, create from directory structure
            self.annotations = self._create_annotations_from_directory()
            logger.info(f"Created {len(self.annotations)} annotations from directory structure")

        # Filter annotations by mode if split column exists
        if "split" in self.annotations.columns:
            self.annotations = self.annotations[
                self.annotations["split"] == mode
            ].reset_index(drop=True)

        # Define default transforms if none provided
        if self.transform is None:
            self.transform = self._get_default_transform()

    def _create_annotations_from_directory(self) -> pd.DataFrame:
        """Create annotations DataFrame from directory structure."""
        # Use predefined image extensions
        image_extensions: Set[str] = IMAGE_EXTENSIONS

        images = []
        labels = []

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory does not exist: {self.data_dir}")

        # If directory has subdirectories (class-based organization)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Class-based directory structure
            for class_dir in subdirs:
                class_name = class_dir.name
                class_images = [
                    img_path for img_path in class_dir.iterdir()
                    if img_path.suffix.lower() in image_extensions
                ]
                
                for img_path in class_images:
                    images.append(str(img_path.relative_to(self.data_dir)))
                    labels.append(class_name)
                
                logger.debug(f"Found {len(class_images)} images in class '{class_name}'")
        else:
            # Flat directory structure
            flat_images = [
                img_path for img_path in self.data_dir.iterdir()
                if img_path.suffix.lower() in image_extensions
            ]
            
            for img_path in flat_images:
                images.append(str(img_path.relative_to(self.data_dir)))
                labels.append("unknown")  # More descriptive default label

        if not images:
            raise ValueError(f"No valid images found in {self.data_dir}")

        return pd.DataFrame({"image_path": images, "label": labels})

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms."""
        if self.mode == "train":
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=DEFAULT_TRANSFORM_PARAMS["rotation_degrees"]),
                    transforms.ColorJitter(
                        brightness=DEFAULT_TRANSFORM_PARAMS["brightness"], 
                        contrast=DEFAULT_TRANSFORM_PARAMS["contrast"], 
                        saturation=DEFAULT_TRANSFORM_PARAMS["saturation"], 
                        hue=DEFAULT_TRANSFORM_PARAMS["hue"]
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=DEFAULT_TRANSFORM_PARAMS["mean"], 
                        std=DEFAULT_TRANSFORM_PARAMS["std"]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=DEFAULT_TRANSFORM_PARAMS["mean"], 
                        std=DEFAULT_TRANSFORM_PARAMS["std"]
                    ),
                ]
            )

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing image and label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image path and label
        row = self.annotations.iloc[idx]
        img_path = self.data_dir / row["image_path"]

        # Load image with validation
        try:
            if not img_path.exists():
                raise FileNotFoundError(f"Image file not found: {img_path}")
            
            image = Image.open(img_path).convert("RGB")
            
            # Validate image
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError(f"Invalid image dimensions: {image.size}")
                
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            raise RuntimeError(f"Error loading image {img_path}: {e}") from e

        # Get label
        label = row["label"]

        # Convert string labels to numeric if needed
        if isinstance(label, str):
            label = self._encode_label(label)

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        sample = {
            "image": image,
            "label": label,
            "image_path": str(img_path),
            "index": idx,
        }

        # Add additional annotations if available
        for col in self.annotations.columns:
            if col not in ["image_path", "label"]:
                sample[col] = row[col]

        return sample

    @lru_cache(maxsize=None)
    def _get_label_encoder(self) -> Dict[str, int]:
        """Get or create label encoder mapping."""
        unique_labels = self.annotations["label"].unique()
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
    def _encode_label(self, label: str) -> int:
        """Encode string label to integer."""
        if not hasattr(self, "_label_encoder_cache"):
            self._label_encoder_cache = self._get_label_encoder()
        return self._label_encoder_cache[label]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets using sklearn-like formula."""
        if "label" not in self.annotations.columns or len(self.annotations) == 0:
            return torch.ones(1)

        # Convert string labels to numeric
        labels = self.annotations["label"].values
        if len(labels) > 0 and isinstance(labels[0], str):
            labels = [self._encode_label(label) for label in labels]

        # Calculate class frequencies
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)
        n_classes = len(unique_labels)

        # Calculate balanced weights (sklearn formula)
        weights = total_samples / (n_classes * counts)

        # Create tensor with weights for all classes
        class_weights = torch.zeros(n_classes, dtype=torch.float32)
        for i, label in enumerate(unique_labels):
            class_weights[label] = weights[i]

        logger.info(f"Calculated class weights for {n_classes} classes: {class_weights.tolist()}")
        return class_weights

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "total_samples": len(self.annotations),
            "mode": self.mode,
            "dataset_type": self.dataset_type,
            "image_size": self.image_size,
        }

        if "label" in self.annotations.columns:
            label_counts = self.annotations["label"].value_counts()
            stats["label_distribution"] = label_counts.to_dict()
            stats["num_classes"] = len(label_counts)

        return stats


class MultiTaskRetinaDataset(RetinaDataset):
    """
    Dataset for multi-task learning on retinal images.

    Supports multiple simultaneous tasks like:
    - Disease classification
    - Severity grading
    - Vessel segmentation
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Identify task columns (excluding metadata columns)
        excluded_columns = {"image_path", "split", "index", "label"}
        self.task_columns = [
            col for col in self.annotations.columns
            if col not in excluded_columns
        ]
        
        logger.info(f"Initialized MultiTaskRetinaDataset with {len(self.task_columns)} task columns: {self.task_columns}")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with multiple tasks."""
        sample = super().__getitem__(idx)

        # Add all task labels
        tasks = {}
        row = self.annotations.iloc[idx]

        for task_col in self.task_columns:
            if task_col in row:
                task_value = row[task_col]
                if isinstance(task_value, str):
                    task_value = self._encode_task_label(task_col, task_value)
                tasks[task_col] = task_value

        sample["tasks"] = tasks
        return sample

    def _encode_task_label(self, task: str, label: str) -> int:
        """Encode task-specific labels with caching."""
        if not hasattr(self, "_task_encoders"):
            self._task_encoders = {}
            
        if task not in self._task_encoders:
            unique_labels = self.annotations[task].unique()
            self._task_encoders[task] = {
                label: idx for idx, label in enumerate(sorted(unique_labels))
            }
            logger.debug(f"Created encoder for task '{task}' with {len(unique_labels)} classes")

        return self._task_encoders[task][label]
