"""
Dataset module for retinal image analysis.

This module provides PyTorch Dataset classes for loading and preprocessing
retinal images for machine learning tasks.
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


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
        image_size: Tuple[int, int] = (512, 512),
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
        if annotations_file and os.path.exists(annotations_file):
            self.annotations = pd.read_csv(annotations_file)
        else:
            # If no annotations file, create from directory structure
            self.annotations = self._create_annotations_from_directory()

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
        image_extensions = {".jpg", ".jpeg", ".png", ".tiff", ".bmp"}

        images = []
        labels = []

        # If directory has subdirectories (class-based organization)
        subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        if subdirs:
            # Class-based directory structure
            for class_dir in subdirs:
                class_name = class_dir.name
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        images.append(str(img_path.relative_to(self.data_dir)))
                        labels.append(class_name)
        else:
            # Flat directory structure
            for img_path in self.data_dir.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    images.append(str(img_path.relative_to(self.data_dir)))
                    labels.append(0)  # Default label

        return pd.DataFrame({"image_path": images, "label": labels})

    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transforms."""
        if self.mode == "train":
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error loading image {img_path}: {e}")

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

    def _encode_label(self, label: str) -> int:
        """Encode string label to integer."""
        if not hasattr(self, "label_encoder"):
            unique_labels = self.annotations["label"].unique()
            self.label_encoder = {
                label: idx for idx, label in enumerate(sorted(unique_labels))
            }

        return self.label_encoder[label]

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced datasets."""
        if "label" not in self.annotations.columns:
            return torch.ones(1)

        # Convert string labels to numeric
        labels = self.annotations["label"].values
        if isinstance(labels[0], str):
            labels = [self._encode_label(label) for label in labels]

        # Calculate class frequencies
        unique_labels, counts = np.unique(labels, return_counts=True)
        total_samples = len(labels)

        # Calculate weights (inverse frequency)
        weights = total_samples / (len(unique_labels) * counts)

        # Create tensor with weights for all classes
        class_weights = torch.zeros(len(unique_labels))
        for i, label in enumerate(unique_labels):
            class_weights[label] = weights[i]

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Identify task columns
        self.task_columns = [
            col
            for col in self.annotations.columns
            if col not in ["image_path", "split", "index"]
        ]

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
        """Encode task-specific labels."""
        encoder_attr = f"{task}_encoder"

        if not hasattr(self, encoder_attr):
            unique_labels = self.annotations[task].unique()
            encoder = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            setattr(self, encoder_attr, encoder)

        encoder = getattr(self, encoder_attr)
        return encoder[label]
