# Retinal Image Datasets Inventory

This document provides an overview of all datasets available in this project.

## Available Datasets

### 1. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Location**: `data/raw/drive/`
- **Size**: 40 images (20 train, 20 test)
- **Resolution**: 565 × 584 pixels
- **Purpose**: Vessel segmentation
- **Status**: Manual download required
- **URL**: https://drive.grand-challenge.org/

### 2. STARE (STructured Analysis of the Retina)
- **Location**: `data/raw/stare/`
- **Size**: 397 images
- **Purpose**: Vessel extraction and pathology detection
- **Status**: Auto-downloadable
- **URL**: http://cecas.clemson.edu/~ahoover/stare/

### 3. Messidor
- **Location**: `data/raw/messidor/`
- **Size**: 1,200 images
- **Purpose**: Diabetic retinopathy detection
- **Status**: Manual download required
- **URL**: http://www.adcis.net/en/third-party/messidor/

### 4. Kaggle Diabetic Retinopathy Detection
- **Location**: `data/raw/kaggle_dr/`
- **Size**: 35,000+ images (~88GB)
- **Purpose**: DR classification
- **Status**: Kaggle CLI required
- **URL**: https://www.kaggle.com/c/diabetic-retinopathy-detection

## Usage Instructions

1. **Download datasets**: Run `scripts/download_datasets.sh`
2. **Validate data**: Use `src/data/validation.py`
3. **Preprocess**: Use `src/data/preprocessing.py`
4. **Load data**: Use `src/data/dataset.py` and `src/data/loaders.py`

## Data Organization

```
data/
├── raw/                    # Original downloaded datasets
│   ├── drive/
│   ├── stare/
│   ├── messidor/
│   └── kaggle_dr/
├── processed/              # Preprocessed images
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/            # Processed annotations
└── cache/                  # Cached data for faster loading
```

## Quality Metrics

After downloading, run validation to ensure data quality:

```bash
python -m src.data.validation --data-dir data/raw/stare --output-report data/validation_report.json
```

## License Information

Each dataset has its own license terms. Please review and comply with:
- DRIVE: Research use only
- STARE: Academic research use
- Messidor: Research purposes with registration
- Kaggle DR: Competition rules apply

## Contributing

When adding new datasets:
1. Add download instructions to this script
2. Update this inventory
3. Add validation support
4. Document preprocessing requirements
