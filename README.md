# GalaxyMorph ML

Automatic galaxy morphology classification using Convolutional Neural Networks (CNN).

## 🎯 Project Overview

GalaxyMorph is a machine learning pipeline that classifies galaxy morphology from astronomical images into four categories:

- **Spiral** 🌀
- **Elliptical** ⭕
- **Lenticular** 💿
- **Irregular** 🌌

### Current Phase

We're building the **training pipeline** using:

- **Dataset**: Galaxy Zoo 2
- **Architecture**: ResNet50 (pretrained backbone)
- **Framework**: PyTorch
- **Input**: 424×424×3 astronomical images

## 📁 Project Structure

```
galaxy-morph-ml/
├── configs/                 # Configuration files (YAML)
│   ├── default.yaml        # Default model & training config
│   └── dataset.yaml        # Dataset paths & specs
├── data/                   # Dataset directory (gitignored)
│   ├── images/            # Raw galaxy images
│   └── processed/         # Processed dataset
├── logs/                   # Training logs (gitignored)
├── models/                 # Trained checkpoints (gitignored)
├── notebooks/              # Jupyter notebooks for exploration
├── scripts/                # Utility scripts
├── src/                    # Main source code
│   ├── config.py          # Configuration loader
│   ├── data/              # Data loaders & preprocessing
│   ├── models/            # Model architectures
│   ├── training/          # Training loops & utilities
│   ├── evaluation/        # Metrics & evaluation
│   └── utils/             # Helper functions
├── tests/                  # Unit tests
├── Dockerfile             # Docker container definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🐳 Docker Setup

### Build the Docker Image

```bash
docker compose build
```

### Run Interactive Development Shell

```bash
docker compose run --rm app bash
```

This will:

- Build the image (if not already built)
- Mount your workspace into `/app`
- Start an interactive bash shell
- Auto-clean up when you exit

### Run a Specific Command in Docker

```bash
docker compose run --rm app python -c "import torch; print(torch.__version__)"
```

### Verify Python & Dependencies

Inside the container:

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(torch.__version__)"

# Check all dependencies
python -c "import pandas, numpy, sklearn, PIL, matplotlib, yaml; print('✓ All dependencies OK')"
```

## 🚀 Quick Start

### 1. **Build and Enter Container**

```bash
docker compose up -d --build
docker compose run --rm app bash
```

### 2. **Verify Environment**

```bash
# Inside container
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 3. **Configuration**

Edit configuration files in `configs/`:

- `configs/default.yaml` — Model, training, augmentation settings
- `configs/dataset.yaml` — Dataset paths and specs

### 4. **Next Steps**

After setup, the following components need to be implemented:

- [ ] Data loading pipeline (Galaxy Zoo 2 dataset integration)
- [ ] Data preprocessing & augmentation
- [ ] ResNet50 model wrapper
- [ ] Training loop & loss functions
- [ ] Evaluation metrics (accuracy, confusion matrix, etc.)
- [ ] Checkpoint saving & loading
- [ ] Inference script
- [ ] Jupyter notebooks for exploration

## 📦 Dependencies

**Core ML**

- `torch` — Deep learning framework
- `torchvision` — Computer vision utilities
- `pandas` — Data manipulation
- `scikit-learn` — Metrics & preprocessing
- `Pillow` — Image processing
- `numpy` — Numerical computing

**Utilities**

- `matplotlib` — Visualization
- `jupyter`, `ipython` — Interactive development
- `PyYAML` — Configuration files
- `tqdm` — Progress bars
- `python-dotenv` — Environment variables

See `requirements.txt` for exact versions.

## ⚙️ Configuration Files

### `configs/default.yaml`

Main configuration for model training:

```yaml
model:
  name: ResNet50
  num_classes: 4
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 1.0e-4
```

### `configs/dataset.yaml`

Dataset paths and specifications:

```yaml
dataset:
  data_dir: './data'
  images_dir: './data/images'
  final_dataset_file: './data/processed/galaxy_dataset.csv'
```

## 🔧 Useful Docker Commands

```bash
# Build image
docker compose build

# Run with interactive shell
docker compose run --rm app bash

# Run a Python script
docker compose run --rm app python src/data/preprocessing.py

# Run Jupyter notebook
docker compose run --rm -p 8888:8888 app jupyter notebook --ip=0.0.0.0

# View logs
docker compose logs app

# Remove dangling containers
docker compose down
```

## 📝 Development Notes

- **Working Directory**: `/app` inside container (same as project root)
- **Volumes**: All project directories are mounted, changes persist
- **Python Path**: `src/` is available as imports via Python path
- **GPU Support**: Currently CPU-only; update Dockerfile for CUDA support if needed

## 🔄 Workflow

1. **Edit code locally** (outside container)
2. **Run in Docker** (all commands)
3. **View output** (logs, models, artifacts)
4. **Iterate** on data/models

For notebooks:

```bash
docker compose run --rm -p 8888:8888 app jupyter notebook --ip=0.0.0.0 --allow-root
```

Then open: `http://localhost:8888`

## 🤝 Contributing

- Maintain Python 3.11 compatibility
- Follow PEP 8 style guidelines
- Update `requirements.txt` when adding dependencies
- Document new modules with docstrings
- Add tests for new functionality

## 📊 Galaxy Morphology Classes

The four classifications we're targeting:

1. **Spiral** — Rotating disk with spiral arms
2. **Elliptical** — Smooth, featureless ellipsoidal shape
3. **Lenticular** — Disk with prominent bulge but no spiral arms
4. **Irregular** — Chaotic, asymmetric structure

## 📚 Resources

- [Galaxy Zoo 2 Dataset](https://www.zooniverse.org/projects/galaxyzoo/galaxy-zoo-2)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs)
- [Torchvision Models](https://pytorch.org/vision/stable/models.html)

## 📄 License

[To be defined]

---

**Status**: 🚧 Environment Setup Complete - Ready for Development

Next milestone: Data preprocessing & EDA
