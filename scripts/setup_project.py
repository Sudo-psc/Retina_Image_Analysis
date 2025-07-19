#!/usr/bin/env python3
"""
Project initialization script for Retina Image Analysis.

This script sets up the development environment and downloads
initial datasets for the project.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_command(command: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version() -> bool:
    """Check if Python version is compatible."""
    python_version = sys.version_info
    
    if python_version.major != 3 or python_version.minor < 9:
        print(f"❌ Python 3.9+ required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    return True


def install_poetry() -> bool:
    """Install Poetry if not already installed."""
    try:
        subprocess.run(["poetry", "--version"], check=True, capture_output=True)
        print("✅ Poetry is already installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("🔄 Installing Poetry...")
        
        # Install Poetry
        install_cmd = [
            "curl", "-sSL", 
            "https://install.python-poetry.org", 
            "|", "python3", "-"
        ]
        
        # Use shell=True for pipe command
        try:
            subprocess.run(
                "curl -sSL https://install.python-poetry.org | python3 -",
                shell=True,
                check=True
            )
            print("✅ Poetry installed successfully!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Poetry. Please install manually:")
            print("   curl -sSL https://install.python-poetry.org | python3 -")
            return False


def setup_environment() -> bool:
    """Set up the development environment."""
    steps = [
        (["poetry", "install"], "Installing dependencies"),
        (["poetry", "run", "pre-commit", "install"], "Installing pre-commit hooks"),
    ]
    
    for command, description in steps:
        if not run_command(command, description):
            return False
    
    return True


def create_data_directories() -> None:
    """Create necessary data directories."""
    directories = [
        "data/raw/drive",
        "data/raw/stare", 
        "data/raw/messidor",
        "data/raw/kaggle_dr",
        "data/processed",
        "data/annotations",
        "models/checkpoints",
        "logs",
        "results",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")


def create_env_file() -> None:
    """Create .env file from template."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        env_file.write_text(env_example.read_text())
        print("📄 Created .env file from template")
    else:
        print("📄 .env file already exists or template not found")


def download_sample_data() -> None:
    """Download sample datasets (optional)."""
    print("\n🔄 Setting up sample data...")
    
    # Create sample data info files
    datasets_info = {
        "data/raw/drive/README.md": """# DRIVE Dataset

Download from: https://drive.grand-challenge.org/

This dataset contains:
- 40 color fundus images
- Manual vessel segmentations
- Splits for training and testing

## Instructions:
1. Register at the website
2. Download the dataset
3. Extract to this directory
4. Maintain the original structure
""",
        "data/raw/stare/README.md": """# STARE Dataset

Download from: http://cecas.clemson.edu/~ahoover/stare/

This dataset contains:
- 397 retinal images
- Manual vessel annotations
- Various pathology cases

## Instructions:
1. Download all images and annotations
2. Extract to this directory
3. Organize by image type
""",
        "data/raw/messidor/README.md": """# Messidor Dataset

Download from: http://www.adcis.net/en/third-party/messidor/

This dataset contains:
- 1,200 eye fundus images
- Diabetic retinopathy grades
- Multiple image sizes

## Instructions:
1. Request access from the website
2. Download the dataset
3. Extract to this directory
""",
        "data/raw/kaggle_dr/README.md": """# Kaggle Diabetic Retinopathy Dataset

Download from: https://www.kaggle.com/c/diabetic-retinopathy-detection

This dataset contains:
- 35,000+ retinal images
- Diabetic retinopathy severity labels
- Training and test sets

## Instructions:
1. Download from Kaggle Competition
2. Extract to this directory
3. Maintain train/test structure
""",
    }
    
    for filepath, content in datasets_info.items():
        file_path = Path(filepath)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        print(f"📄 Created: {filepath}")


def setup_git_hooks() -> None:
    """Set up additional git configuration."""
    git_commands = [
        ["git", "config", "core.autocrlf", "false"],
        ["git", "config", "core.filemode", "false"],
    ]
    
    for command in git_commands:
        try:
            subprocess.run(command, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass  # Ignore errors for optional git config


def print_next_steps() -> None:
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("🎉 Project initialization completed!")
    print("="*60)
    print("\n📋 Next Steps:")
    print("\n1. Activate the environment:")
    print("   poetry shell")
    print("\n2. Download datasets:")
    print("   - Check the README files in data/raw/ directories")
    print("   - Download and extract datasets as instructed")
    print("\n3. Run your first experiment:")
    print("   poetry run python -m pytest tests/  # Run tests")
    print("   poetry run jupyter lab  # Start Jupyter for exploration")
    print("\n4. Start development:")
    print("   git checkout -b feature/your-feature-name")
    print("   # Make your changes")
    print("   git add . && git commit -m 'feat: your feature'")
    print("\n📚 Documentation:")
    print("   - README.md: Project overview")
    print("   - CONTRIBUTING.md: Development guidelines")
    print("   - NEXT_STEPS.md: Detailed roadmap")
    print("\n🆘 Need help?")
    print("   - Check the documentation")
    print("   - Create an issue on GitHub")
    print("   - Review the project structure in README.md")
    print("\n" + "="*60)


def main() -> None:
    """Main initialization function."""
    print("🚀 Initializing Retina Image Analysis Project")
    print("="*50)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Install Poetry if needed
    if not install_poetry():
        sys.exit(1)
    
    # Set up environment
    print("\n🔧 Setting up development environment...")
    if not setup_environment():
        print("\n❌ Environment setup failed. Please check the errors above.")
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating project directories...")
    create_data_directories()
    
    # Create environment file
    print("\n⚙️ Setting up configuration...")
    create_env_file()
    
    # Set up sample data info
    print("\n📊 Setting up data information...")
    download_sample_data()
    
    # Configure git
    print("\n🔧 Configuring git...")
    setup_git_hooks()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
