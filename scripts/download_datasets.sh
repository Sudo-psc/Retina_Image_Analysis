#!/bin/bash

# Download script for retinal image datasets
# This script downloads and organizes the main datasets used for retinal image analysis

set -e

# Configuration
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${BASE_DIR}/data"
RAW_DATA_DIR="${DATA_DIR}/raw"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Create directory structure
create_directories() {
    log "Creating directory structure..."
    
    mkdir -p "${RAW_DATA_DIR}"/{drive,stare,messidor,kaggle_dr}
    mkdir -p "${DATA_DIR}"/{processed,annotations}
    
    # Create dataset-specific directories
    mkdir -p "${RAW_DATA_DIR}/drive"/{training,test}
    mkdir -p "${RAW_DATA_DIR}/stare"/{images,labels}
    mkdir -p "${RAW_DATA_DIR}/messidor"/{images,annotations}
    mkdir -p "${RAW_DATA_DIR}/kaggle_dr"/{train,test,sample}
    
    success "Directory structure created"
}

# Download DRIVE dataset
download_drive() {
    log "Downloading DRIVE dataset..."
    
    DRIVE_DIR="${RAW_DATA_DIR}/drive"
    
    if [ -f "${DRIVE_DIR}/DRIVE.zip" ]; then
        warning "DRIVE dataset already exists, skipping download"
        return 0
    fi
    
    # Note: DRIVE dataset requires registration
    cat > "${DRIVE_DIR}/README.md" << EOF
# DRIVE Dataset

## Download Instructions

The DRIVE dataset requires registration and manual download.

1. Visit: https://drive.grand-challenge.org/
2. Register for an account
3. Download the dataset
4. Extract to this directory

## Expected Structure
\`\`\`
drive/
├── training/
│   ├── images/
│   ├── 1st_manual/
│   └── mask/
└── test/
    ├── images/
    ├── 1st_manual/
    ├── 2nd_manual/
    └── mask/
\`\`\`

## Dataset Information
- 40 color fundus images (20 training, 20 test)
- Resolution: 565 × 584 pixels
- Manual vessel segmentations
- FOV masks included
- Purpose: Vessel segmentation and analysis
EOF
    
    warning "DRIVE dataset requires manual download - see ${DRIVE_DIR}/README.md"
}

# Download STARE dataset
download_stare() {
    log "Downloading STARE dataset..."
    
    STARE_DIR="${RAW_DATA_DIR}/stare"
    
    if [ -d "${STARE_DIR}/images" ] && [ "$(ls -A ${STARE_DIR}/images)" ]; then
        warning "STARE dataset already exists, skipping download"
        return 0
    fi
    
    # STARE dataset is freely available
    cd "${STARE_DIR}"
    
    # Download images
    log "Downloading STARE images..."
    for i in $(seq -f "%02g" 1 20); do
        if [ ! -f "images/im00${i}.ppm" ]; then
            wget -q "http://cecas.clemson.edu/~ahoover/stare/probing/im00${i}.ppm" -O "images/im00${i}.ppm" || true
        fi
    done
    
    # Download labels
    log "Downloading STARE labels..."
    for i in $(seq -f "%02g" 1 20); do
        if [ ! -f "labels/im00${i}.ah.ppm" ]; then
            wget -q "http://cecas.clemson.edu/~ahoover/stare/probing/im00${i}.ah.ppm" -O "labels/im00${i}.ah.ppm" || true
        fi
    done
    
    # Create README
    cat > "${STARE_DIR}/README.md" << EOF
# STARE Dataset

## Dataset Information
- 397 images with manual annotations
- Purpose: Vessel extraction and pathology detection
- Website: http://cecas.clemson.edu/~ahoover/stare/

## Directory Structure
\`\`\`
stare/
├── images/     # Original retinal images
└── labels/     # Manual vessel segmentations
\`\`\`

## Notes
- Images are in PPM format
- Manual segmentations by A. Hoover
- Includes pathological cases
EOF
    
    success "STARE dataset download completed"
}

# Download Messidor dataset
download_messidor() {
    log "Downloading Messidor dataset..."
    
    MESSIDOR_DIR="${RAW_DATA_DIR}/messidor"
    
    # Messidor requires registration
    cat > "${MESSIDOR_DIR}/README.md" << EOF
# Messidor Dataset

## Download Instructions

The Messidor dataset requires registration and manual download.

1. Visit: http://www.adcis.net/en/third-party/messidor/
2. Fill out the request form
3. Download the dataset files
4. Extract to this directory

## Expected Structure
\`\`\`
messidor/
├── images/
│   ├── Base11/
│   ├── Base12/
│   ├── Base13/
│   ├── Base21/
│   ├── Base22/
│   ├── Base23/
│   ├── Base31/
│   ├── Base32/
│   └── Base33/
└── annotations/
    └── messidor.csv
\`\`\`

## Dataset Information
- 1,200 eye fundus color images
- Resolution: 1440 × 960, 2240 × 1488, or 2304 × 1536 pixels
- Diabetic retinopathy grades (0-3)
- Macular edema risk levels
- Purpose: Diabetic retinopathy detection and grading
EOF
    
    warning "Messidor dataset requires manual download - see ${MESSIDOR_DIR}/README.md"
}

# Setup Kaggle DR dataset
setup_kaggle_dr() {
    log "Setting up Kaggle Diabetic Retinopathy dataset..."
    
    KAGGLE_DIR="${RAW_DATA_DIR}/kaggle_dr"
    
    # Check if kaggle CLI is available
    if ! command -v kaggle &> /dev/null; then
        warning "Kaggle CLI not found. Installing..."
        pip install kaggle
    fi
    
    # Create README with instructions
    cat > "${KAGGLE_DIR}/README.md" << EOF
# Kaggle Diabetic Retinopathy Detection Dataset

## Download Instructions

1. Install Kaggle CLI: \`pip install kaggle\`
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Create API token
   - Place kaggle.json in ~/.kaggle/
3. Run the download script

## Download Command
\`\`\`bash
# Download competition data (requires acceptance of competition rules)
kaggle competitions download -c diabetic-retinopathy-detection -p ${KAGGLE_DIR}

# Extract files
cd ${KAGGLE_DIR}
unzip train.zip -d train/
unzip test.zip -d test/
unzip sample.zip -d sample/
\`\`\`

## Dataset Information
- 35,000+ high-resolution retinal images
- Diabetic retinopathy severity scale (0-4)
- Competition format with train/test split
- Purpose: Diabetic retinopathy classification
- Size: ~88GB (compressed)

## Labels
- 0: No DR
- 1: Mild
- 2: Moderate  
- 3: Severe
- 4: Proliferative DR

## Expected Structure
\`\`\`
kaggle_dr/
├── train/
│   └── *.jpeg (training images)
├── test/
│   └── *.jpeg (test images)
├── sample/
│   └── *.jpeg (sample images)
├── trainLabels.csv
└── retinopathy_solution.csv
\`\`\`
EOF
    
    # Create download script
    cat > "${KAGGLE_DIR}/download_kaggle_dr.sh" << 'EOF'
#!/bin/bash

# Download Kaggle Diabetic Retinopathy dataset
# Make sure you have accepted the competition rules first

set -e

KAGGLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Downloading Kaggle Diabetic Retinopathy dataset..."
echo "This may take a while (88GB download)..."

# Download competition data
kaggle competitions download -c diabetic-retinopathy-detection -p "${KAGGLE_DIR}"

echo "Extracting files..."

# Extract train images
if [ -f "${KAGGLE_DIR}/train.zip" ]; then
    unzip -q "${KAGGLE_DIR}/train.zip" -d "${KAGGLE_DIR}/"
    echo "Train images extracted"
fi

# Extract test images  
if [ -f "${KAGGLE_DIR}/test.zip" ]; then
    unzip -q "${KAGGLE_DIR}/test.zip" -d "${KAGGLE_DIR}/"
    echo "Test images extracted"
fi

# Extract sample images
if [ -f "${KAGGLE_DIR}/sample.zip" ]; then
    unzip -q "${KAGGLE_DIR}/sample.zip" -d "${KAGGLE_DIR}/"
    echo "Sample images extracted"
fi

echo "Kaggle DR dataset download completed!"
echo "Training images: $(ls ${KAGGLE_DIR}/train/*.jpeg 2>/dev/null | wc -l)"
echo "Test images: $(ls ${KAGGLE_DIR}/test/*.jpeg 2>/dev/null | wc -l)"
EOF
    
    chmod +x "${KAGGLE_DIR}/download_kaggle_dr.sh"
    
    warning "Kaggle DR dataset requires manual setup - see ${KAGGLE_DIR}/README.md"
}

# Create master dataset inventory
create_inventory() {
    log "Creating dataset inventory..."
    
    cat > "${DATA_DIR}/DATASETS.md" << EOF
# Retinal Image Datasets Inventory

This document provides an overview of all datasets available in this project.

## Available Datasets

### 1. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Location**: \`data/raw/drive/\`
- **Size**: 40 images (20 train, 20 test)
- **Resolution**: 565 × 584 pixels
- **Purpose**: Vessel segmentation
- **Status**: Manual download required
- **URL**: https://drive.grand-challenge.org/

### 2. STARE (STructured Analysis of the Retina)
- **Location**: \`data/raw/stare/\`
- **Size**: 397 images
- **Purpose**: Vessel extraction and pathology detection
- **Status**: Auto-downloadable
- **URL**: http://cecas.clemson.edu/~ahoover/stare/

### 3. Messidor
- **Location**: \`data/raw/messidor/\`
- **Size**: 1,200 images
- **Purpose**: Diabetic retinopathy detection
- **Status**: Manual download required
- **URL**: http://www.adcis.net/en/third-party/messidor/

### 4. Kaggle Diabetic Retinopathy Detection
- **Location**: \`data/raw/kaggle_dr/\`
- **Size**: 35,000+ images (~88GB)
- **Purpose**: DR classification
- **Status**: Kaggle CLI required
- **URL**: https://www.kaggle.com/c/diabetic-retinopathy-detection

## Usage Instructions

1. **Download datasets**: Run \`scripts/download_datasets.sh\`
2. **Validate data**: Use \`src/data/validation.py\`
3. **Preprocess**: Use \`src/data/preprocessing.py\`
4. **Load data**: Use \`src/data/dataset.py\` and \`src/data/loaders.py\`

## Data Organization

\`\`\`
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
\`\`\`

## Quality Metrics

After downloading, run validation to ensure data quality:

\`\`\`bash
python -m src.data.validation --data-dir data/raw/stare --output-report data/validation_report.json
\`\`\`

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
EOF
    
    success "Dataset inventory created"
}

# Main execution
main() {
    log "Starting retinal image dataset download process..."
    
    # Check dependencies
    if ! command -v wget &> /dev/null; then
        error "wget is required but not installed"
        exit 1
    fi
    
    # Create directories
    create_directories
    
    # Download/setup datasets
    download_drive
    download_stare
    download_messidor
    setup_kaggle_dr
    
    # Create inventory
    create_inventory
    
    log "Dataset download process completed!"
    echo
    success "Next steps:"
    echo "1. Review README files in each dataset directory"
    echo "2. Complete manual downloads where required"
    echo "3. Run data validation: python -m src.data.validation"
    echo "4. Start preprocessing: python -m src.data.preprocessing"
}

# Run main function
main "$@"
