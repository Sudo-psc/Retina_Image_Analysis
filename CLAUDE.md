# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI/ML project for retinal image analysis using PyTorch and computer vision techniques. The system is designed to detect and classify retinal diseases including diabetic retinopathy, glaucoma, and macular degeneration from fundus images.

## Development Environment Setup

### Poetry-based Dependency Management
```bash
# Install dependencies
poetry install

# Install with optional ML dependencies
poetry install --extras all

# Install development dependencies only
poetry install --only dev

# Install ML-specific dependencies
poetry install --only ml
```

### Docker Development
```bash
# Development environment with Jupyter
docker-compose up app

# Full stack with MLflow tracking
docker-compose up app mlflow postgres

# Production deployment
docker-compose --profile production up

# With monitoring stack
docker-compose --profile monitoring up
```

## Common Development Commands

### Testing
```bash
# Run all tests with coverage
pytest tests/

# Run specific test file
pytest tests/test_data.py

# Run with coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/test_data.py::TestRetinaDataset
```

### Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Data Processing
```bash
# Preprocess images
python -m src.data.preprocessing --input data/raw --output data/processed

# Create dataset splits
python -m src.data.dataset --config configs/config.yaml
```

### Training
```bash
# Train model with default config
python -m src.training.train

# Train with custom config
python -m src.training.train --config configs/custom_config.yaml

# Resume training from checkpoint
python -m src.training.train --resume checkpoints/latest.pth
```

### API Server
```bash
# Start development API server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Start production server
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

### MLflow Tracking
```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

# View experiments in browser
# Navigate to http://localhost:5000
```

## Architecture Overview

### Core Module Structure
- **`src/data/`**: Dataset loading, preprocessing, and augmentation
  - `dataset.py`: PyTorch Dataset classes for retinal images
  - `preprocessing.py`: Image preprocessing pipeline with quality assessment
- **`src/models/`**: Neural network architectures and model definitions
- **`src/training/`**: Training loops, optimization, and experiment management
- **`src/inference/`**: Model inference, prediction, and evaluation
- **`src/api/`**: FastAPI REST API for model serving
- **`src/utils/`**: Shared utilities and helper functions

### Data Pipeline
1. **Raw images** stored in `data/raw/` organized by dataset (DRIVE, STARE, Messidor, etc.)
2. **Preprocessing** applies CLAHE contrast enhancement, illumination normalization, and artifact removal
3. **Processed images** saved to `data/processed/` with metadata
4. **Annotations** in `data/annotations/` contain ground truth labels and medical metadata

### Configuration System
- Central config in `configs/config.yaml` with hierarchical settings
- Supports model architectures: ResNet50, EfficientNet, DenseNet, custom CNN
- Training hyperparameters, data augmentation, and evaluation metrics
- Hardware configuration for CPU/CUDA/MPS with mixed precision

### Multi-Task Learning Support
- `RetinaDataset`: Single-task classification
- `MultiTaskRetinaDataset`: Simultaneous disease classification, severity grading, vessel segmentation
- Flexible label encoding for string-based class names

## Key Implementation Details

### Image Preprocessing Pipeline
The `ImagePreprocessor` class implements medical imaging best practices:
- Quality assessment with sharpness, contrast, SNR metrics
- Artifact removal using morphological operations
- CLAHE contrast enhancement in LAB color space
- Illumination normalization to handle uneven lighting
- Optic disc detection for region-specific analysis

### Dataset Flexibility
- Supports multiple dataset formats (directory structure, CSV annotations)
- Automatic train/val/test splitting with configurable ratios
- Class weight calculation for imbalanced datasets
- Comprehensive data augmentation for medical images

### Development Workflow
- Bilingual documentation (English/Portuguese) throughout codebase
- Comprehensive test suite with fixtures for temporary datasets
- Docker multi-stage builds for development and production
- MLflow integration for experiment tracking
- Poetry for reproducible dependency management

### Testing Strategy
- Unit tests for all data processing components
- Integration tests for end-to-end preprocessing pipeline
- Mock imports for optional dependencies (torch, cv2)
- Temporary dataset fixtures for isolated testing
- Project structure validation tests

## Hardware Requirements

- **Development**: 8GB+ RAM, CPU sufficient for preprocessing
- **Training**: 16GB+ RAM, CUDA-compatible GPU recommended
- **Production**: Configurable workers, supports CPU/GPU inference
- **Docker**: 4GB+ available memory for full stack