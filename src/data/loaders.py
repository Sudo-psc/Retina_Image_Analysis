"""
Data loaders for retinal image analysis.

This module provides optimized data loaders for efficient training
and inference with retinal imaging datasets.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.sampler import WeightedRandomSampler

from .augmentation import get_retina_transforms
from .dataset import MultiTaskRetinaDataset, RetinaDataset

logger = logging.getLogger(__name__)


class DataLoaderConfig:
    """Configuration class for data loaders."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """
        Initialize data loader configuration.

        Args:
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch per worker
            shuffle: Whether to shuffle data
            drop_last: Whether to drop last incomplete batch
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle = shuffle
        self.drop_last = drop_last


class RetinaDataModule:
    """
    Data module for retinal image analysis.
    
    This class handles the creation and management of data loaders
    for training, validation, and testing.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        annotations_file: Optional[Union[str, Path]] = None,
        image_size: Tuple[int, int] = (512, 512),
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        val_test_split: float = 0.5,
        augmentation_strength: str = "medium",
        dataset_type: str = "classification",
        use_weighted_sampling: bool = False,
        seed: int = 42,
    ) -> None:
        """
        Initialize the data module.

        Args:
            data_dir: Directory containing the images
            annotations_file: Path to annotations CSV file
            image_size: Target image size
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            train_val_split: Fraction of data for training
            val_test_split: Fraction of remaining data for validation
            augmentation_strength: Strength of data augmentation
            dataset_type: Type of dataset ('classification', 'segmentation', 'multitask')
            use_weighted_sampling: Whether to use weighted sampling for imbalanced datasets
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.annotations_file = annotations_file
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.val_test_split = val_test_split
        self.augmentation_strength = augmentation_strength
        self.dataset_type = dataset_type
        self.use_weighted_sampling = use_weighted_sampling
        self.seed = seed

        # Data loaders will be created in setup()
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.test_loader: Optional[DataLoader] = None

        # Datasets
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self) -> None:
        """Set up the datasets and data loaders."""
        logger.info("Setting up data module...")

        # Create full dataset
        full_dataset = self._create_dataset(mode="full")

        # Split dataset
        train_dataset, val_dataset, test_dataset = self._split_dataset(full_dataset)

        # Store datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, mode="train")
        self.val_loader = self._create_dataloader(val_dataset, mode="val")
        self.test_loader = self._create_dataloader(test_dataset, mode="test")

        logger.info(f"Data module setup complete:")
        logger.info(f"  Train samples: {len(train_dataset)}")
        logger.info(f"  Validation samples: {len(val_dataset)}")
        logger.info(f"  Test samples: {len(test_dataset)}")

    def _create_dataset(self, mode: str) -> Dataset:
        """Create dataset based on configuration."""
        # Get transforms for the mode
        transform = get_retina_transforms(
            mode=mode if mode != "full" else "train",
            image_size=self.image_size,
            augmentation_strength=self.augmentation_strength,
        )

        # Choose dataset class based on type
        if self.dataset_type == "multitask":
            dataset_class = MultiTaskRetinaDataset
        else:
            dataset_class = RetinaDataset

        # Create dataset
        dataset = dataset_class(
            data_dir=self.data_dir,
            annotations_file=self.annotations_file,
            transform=transform,
            mode=mode,
            dataset_type=self.dataset_type,
        )

        return dataset

    def _split_dataset(self, dataset: Dataset) -> Tuple[Subset, Subset, Subset]:
        """Split dataset into train, validation, and test sets."""
        total_size = len(dataset)
        train_size = int(total_size * self.train_val_split)
        remaining_size = total_size - train_size
        val_size = int(remaining_size * self.val_test_split)
        test_size = remaining_size - val_size

        # Set random seed for reproducibility
        torch.manual_seed(self.seed)

        # Split dataset
        train_dataset, temp_dataset = random_split(
            dataset, [train_size, remaining_size]
        )
        val_dataset, test_dataset = random_split(
            temp_dataset, [val_size, test_size]
        )

        return train_dataset, val_dataset, test_dataset

    def _create_dataloader(self, dataset: Dataset, mode: str) -> DataLoader:
        """Create data loader for a dataset."""
        # Configure data loader based on mode
        if mode == "train":
            shuffle = True
            drop_last = True
            sampler = self._create_sampler(dataset) if self.use_weighted_sampling else None
            if sampler is not None:
                shuffle = False  # Can't use shuffle with custom sampler
        else:
            shuffle = False
            drop_last = False
            sampler = None

        # Create data loader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else 2,
            drop_last=drop_last,
            sampler=sampler,
        )

        return dataloader

    def _create_sampler(self, dataset: Dataset) -> Optional[WeightedRandomSampler]:
        """Create weighted sampler for imbalanced datasets."""
        try:
            # Get labels from dataset
            if hasattr(dataset, 'dataset'):
                # This is a Subset
                labels = [dataset.dataset.get_label(dataset.indices[i]) for i in range(len(dataset))]
            else:
                # This is the full dataset
                labels = [dataset.get_label(i) for i in range(len(dataset))]

            # Calculate class weights
            from collections import Counter
            label_counts = Counter(labels)
            total_samples = len(labels)
            num_classes = len(label_counts)

            # Calculate weights inversely proportional to class frequency
            class_weights = {
                label: total_samples / (num_classes * count)
                for label, count in label_counts.items()
            }

            # Create sample weights
            sample_weights = [class_weights[label] for label in labels]

            # Create sampler
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )

            logger.info(f"Created weighted sampler with class weights: {class_weights}")
            return sampler

        except Exception as e:
            logger.warning(f"Could not create weighted sampler: {e}")
            return None

    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        if self.train_loader is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        return self.train_loader

    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_loader is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        return self.val_loader

    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        if self.test_loader is None:
            raise RuntimeError("Data module not set up. Call setup() first.")
        return self.test_loader

    def get_class_weights(self) -> Optional[torch.Tensor]:
        """Get class weights for loss function."""
        if self.train_dataset is None:
            return None

        try:
            # Get labels from training dataset
            if hasattr(self.train_dataset, 'dataset'):
                labels = [
                    self.train_dataset.dataset.get_label(self.train_dataset.indices[i])
                    for i in range(len(self.train_dataset))
                ]
            else:
                labels = [self.train_dataset.get_label(i) for i in range(len(self.train_dataset))]

            # Calculate class weights
            from collections import Counter
            label_counts = Counter(labels)
            num_classes = len(label_counts)
            total_samples = len(labels)

            # Create weight tensor
            weights = torch.zeros(num_classes)
            for label, count in label_counts.items():
                weights[label] = total_samples / (num_classes * count)

            return weights

        except Exception as e:
            logger.warning(f"Could not calculate class weights: {e}")
            return None


def create_retina_dataloaders(
    data_dir: Union[str, Path],
    annotations_file: Optional[Union[str, Path]] = None,
    config: Optional[DataLoaderConfig] = None,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create retinal image data loaders.

    Args:
        data_dir: Directory containing the images
        annotations_file: Path to annotations CSV file
        config: Data loader configuration
        **kwargs: Additional arguments for RetinaDataModule

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if config is None:
        config = DataLoaderConfig()

    # Create data module
    data_module = RetinaDataModule(
        data_dir=data_dir,
        annotations_file=annotations_file,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        **kwargs,
    )

    # Setup data module
    data_module.setup()

    return (
        data_module.get_train_loader(),
        data_module.get_val_loader(),
        data_module.get_test_loader(),
    )


def create_inference_dataloader(
    data_dir: Union[str, Path],
    image_size: Tuple[int, int] = (512, 512),
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create data loader for inference.

    Args:
        data_dir: Directory containing the images
        image_size: Target image size
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Data loader for inference
    """
    # Create dataset with test transforms (no augmentation)
    transform = get_retina_transforms(mode="test", image_size=image_size)
    dataset = RetinaDataset(
        data_dir=data_dir,
        transform=transform,
        mode="inference",
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return dataloader


# Example usage configurations
DATALOADER_CONFIGS = {
    "development": DataLoaderConfig(
        batch_size=16,
        num_workers=2,
        persistent_workers=False,
    ),
    "training": DataLoaderConfig(
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
    ),
    "production": DataLoaderConfig(
        batch_size=64,
        num_workers=8,
        persistent_workers=True,
        prefetch_factor=4,
    ),
}
