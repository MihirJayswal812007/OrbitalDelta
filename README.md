# 🛰️ OrbitalDelta

**Production-grade satellite change detection — from raw imagery to queryable geospatial polygons.**

OrbitalDelta is a complete, open-source geospatial software platform built around a Siamese U-Net deep learning model. It processes before/after satellite image pairs to detect and map surface changes (building construction, deforestation, flood damage, etc.) as structured, queryable geographic data.

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 🧠 **Siamese U-Net** | ResNet-18 backbone, shared encoder weights, 4-level skip connections |
| 🗺️ **Geo-aware I/O** | GeoTIFF / JP2 input with CRS & affine transform preservation |
| 🔗 **Image Registration** | ORB + SIFT keypoint matching with RANSAC homography |
| 🧩 **Tiling Engine** | Overlap-blended split/stitch for arbitrary image sizes |
| 🔍 **Polygonization** | Mask → Shapely polygons with area, confidence, and centroid attributes |
| 🗄️ **Spatial Storage** | Default: OGC GeoPackage (SQLite); optional: PostGIS |
| 🌐 **REST API** | FastAPI service with async job queue |
| 🗺️ **Map Viewer** | Leaflet + OpenStreetMap, zero API key required |

---

## 🚀 Quick Start

### 1 — Install

```bash
git clone https://github.com/yourname/orbitaldelta.git
cd orbitaldelta
pip install -e .
```

### 2 — Download Dataset & Train

```bash
# Download LEVIR-CD dataset
python -m src.data.download --dataset levir-cd --output data/raw/

# Preprocess into 256×256 patches
python -m src.data.preprocess --input data/raw/levir-cd --output data/processed/levir-cd

# Train (GPU recommended — ~4h on RTX 3080)
python scripts/train.py --config configs/train_levir.yaml
```

### 3 — Run Inference

```bash
python scripts/predict.py \
  --img-a data/processed/levir-cd/test/A/sample_0.png \
  --img-b data/processed/levir-cd/test/B/sample_0.png \
  --checkpoint checkpoints/best.pt \
  --output outputs/change_map.png
```

### 4 — Start the API + Map Viewer

```bash
python scripts/serve.py --port 8000
# Open http://localhost:8000 in your browser
```

---

## 🗂️ Project Structure

```
orbitaldelta/
├── configs/            # YAML training configurations
├── data/               # Raw and processed datasets
├── scripts/            # CLI entry points
│   ├── train.py        # Full training loop
│   ├── evaluate.py     # Evaluation on test set
│   ├── predict.py      # Single-pair inference
│   ├── pipeline.py     # End-to-end geospatial pipeline
│   └── serve.py        # FastAPI server launcher
├── src/
│   ├── data/           # Dataset, loaders, transforms
│   ├── models/         # SiameseUNet, encoders, decoders, losses
│   ├── training/       # Trainer, metrics
│   ├── registration/   # Feature matching, homography, warping
│   ├── tiling/         # TileSplitter, TileStitcher, padding
│   ├── geospatial/     # GeoReader, GeoWriter, CRS utilities
│   ├── postprocessing/ # Connected components, polygonizer, attributes
│   ├── storage/        # GeoPackageStore, PostGISStore
│   ├── api/            # FastAPI app, routes, schemas, background tasks
│   └── web/            # Leaflet map viewer (HTML, CSS, JS)
└── tests/              # pytest test suite
```

---

## 📊 Model Performance

> **Note:** Benchmarks require GPU training on LEVIR-CD. Results below are targets. Run Phase 4 on Kaggle/Colab to populate real numbers.

| Dataset | F1 Score | IoU | Precision | Recall |
|---------|----------|-----|-----------|--------|
| LEVIR-CD | ≥ 0.88 | ≥ 0.80 | — | — |
| WHU-CD | — | — | — | — |
| DSIFN-CD | — | — | — | — |

---

## 🏗️ Architecture

```
Image A → ┐
           ├─ Shared ResNet-18 Encoder → |F_A - F_B| + concat → U-Net Decoder → Change Map
Image B → ┘
```

- **Encoder**: ResNet-18 (pretrained on ImageNet), weights shared between both branches
- **Differencing**: Element-wise `|F_A − F_B|` concatenated with `(F_A, F_B)` at each scale
- **Decoder**: 4-level U-Net with ConvTranspose2d upsampling and skip connections
- **Output**: Single-channel sigmoid probability map → binary change mask at threshold 0.5
- **Loss**: BCE (α=0.5) + Soft Dice (β=0.5)

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/detect` | Submit image pair for detection |
| `GET` | `/api/v1/jobs/{id}` | Poll job status |
| `GET` | `/api/v1/detections` | List all detections |
| `GET` | `/api/v1/detections/{id}` | Get single detection |
| `POST` | `/api/v1/detections/query` | Spatial bounding-box query |

Interactive docs: `http://localhost:8000/docs`

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific phase
pytest tests/test_model.py -v
pytest tests/test_registration.py tests/test_tiling.py -v
pytest tests/test_postprocessing.py tests/test_storage.py -v
pytest tests/test_api.py -v
```

---

## 🖥️ Training on Kaggle / Google Colab

For GPU-accelerated training (Phase 4), use the provided notebook:

```bash
# Upload to Kaggle/Colab, then run:
python scripts/train.py --config configs/train_levir.yaml
```

Configuration: `configs/train_levir.yaml`
- Batch size: 8, Epochs: 200, Early stopping patience: 15
- Scheduler: CosineAnnealingWarmRestarts, LR: 1e-3
- Mixed precision (AMP): enabled

---

## 🌐 Zero-Cost Constraint

OrbitalDelta uses **only free, open-source tools**:

| Need | Solution |
|------|----------|
| Training dataset | LEVIR-CD (free academic) |
| GPU compute | Kaggle/Google Colab free tier |
| Model weights | ImageNet pretrained via torchvision |
| Spatial storage | OGC GeoPackage (SQLite-based) |
| Map tiles | OpenStreetMap / Leaflet (no API key) |
| All dependencies | MIT/Apache-2.0 licensed |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
