# 🎯 Modular AI Vision Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/matheussiqueirahub/modular-video-ai-pipeline/graphs/commit-activity)

A **production-ready, modular computer vision pipeline** for advanced video analysis. Integrates state-of-the-art AI models for object detection, segmentation, tracking, clustering, OCR, and event detection—adaptable to **sports analytics, security, retail, and surveillance**.

---

## 🌟 Features

✨ **Modular Architecture** - Clean separation of concerns with pluggable components  
🔍 **Multi-Stage Processing** - Detection → Segmentation → Tracking → Clustering → OCR → Events  
🎨 **Rich Visualization** - Annotated videos with bounding boxes, masks, IDs, and event overlays  
🚀 **Mock Mode** - Test pipeline without downloading heavy models  
🔧 **Extensible** - Easy to adapt for different domains and use cases  
📊 **Event Detection** - Temporal analysis for anomalies (dwell time, zone entry, proximity)  

---

## 🏗️ Architecture

```mermaid
graph LR
    A[Video Input] --> B[Detection RF-DETR]
    B --> C[Segmentation SAM2]
    C --> D[Tracking]
    D --> E[Clustering SigLIP]
    E --> F[OCR SmolVLM2]
    F --> G[Homography]
    G --> H[Event Detection]
    H --> I[Visualization]
    I --> J[Output Video]
```

### Pipeline Stages

| Stage | Model/Tech | Purpose |
|-------|-----------|---------|
| **Detection** | RF-DETR | Detect people and objects in frames |
| **Segmentation** | SAM2 | Generate precise masks for tracked objects |
| **Tracking** | SAM2 Video | Maintain consistent IDs across frames |
| **Clustering** | SigLIP + UMAP + K-Means | Group objects by visual similarity |
| **OCR** | SmolVLM2 | Read text from objects (numbers, signs) |
| **Homography** | OpenCV | Map coordinates to top-down view |
| **Events** | State Machine | Detect temporal patterns and anomalies |
| **Visualization** | Supervision | Rich frame annotations |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **CUDA GPU** (recommended for real models)
- **Git**

### Installation

```bash
# Clone the repository
git clone https://github.com/matheussiqueirahub/modular-video-ai-pipeline.git
cd modular-video-ai-pipeline

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### Run in Mock Mode (No Model Weights Required)

```bash
python demo.py --mock --output_path demo_output.mp4
```

#### Run with Your Own Video

```bash
python demo.py --video_path input.mp4 --output_path result.mp4
```

#### Advanced Options

```bash
python demo.py \
  --video_path sports_game.mp4 \
  --output_path analyzed_game.mp4 \
  --debug
```

---

## 📁 Project Structure

```
ai_vision_pipeline/
├── demo.py                    # Main pipeline script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── tests/                     # Unit tests
│   ├── test_detector.py
│   ├── test_segmenter.py
│   └── test_clustering.py
└── src/
    ├── detection/
    │   └── detector.py        # RF-DETR wrapper
    ├── segmentation/
    │   └── segmenter.py       # SAM2 video segmentation
    ├── clustering/
    │   └── identifier.py      # SigLIP + clustering
    ├── ocr/
    │   └── reader.py          # SmolVLM2 text recognition
    ├── homography/
    │   └── transformer.py     # Perspective transformation
    ├── events/
    │   └── analyzer.py        # Event detection engine
    └── visualization/
        └── drawer.py          # Frame annotation
```

---

## 🎓 Use Cases

### 🏀 Sports Analytics
- Track players and ball positions
- Identify teams by jersey color
- Read player numbers with OCR
- Detect key events (goals, fouls)

### 🏪 Retail Intelligence
- Count customers in zones
- Track dwell time near products
- Identify staff vs. customers
- Detect queue formation

### 🔒 Security & Surveillance
- Track individuals across cameras
- Detect loitering or suspicious behavior
- Read license plates
- Alert on zone entry violations

---

## 🧪 Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific module tests
python -m pytest tests/test_detector.py -v
```

---

## 🔧 Transitioning to Real Models

The pipeline runs in **Mock Mode** by default. To use real AI models:

1. **Install PyTorch**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Install Transformers**:
   ```bash
   pip install transformers accelerate timm
   ```

3. **Uncomment model loading** in each module (`src/*/`)

4. **Download weights** (models will auto-download on first run)

---

## 📊 Performance

| Mode | FPS (RTX 3090) | Memory Usage |
|------|----------------|--------------|
| Mock Mode | 30+ fps | ~500 MB |
| Full Pipeline | 5-10 fps | ~8 GB VRAM |

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **RF-DETR** - Real-time detection transformer
- **SAM2** - Segment Anything Model 2 for video
- **SigLIP** - Improved CLIP for visual embeddings
- **SmolVLM2** - Lightweight vision-language model
- **Supervision** - Computer vision utilities

---

## 📧 Contact

**Matheus Siqueira** - [@matheussiqueirahub](https://github.com/matheussiqueirahub)

**Project Link**: https://github.com/matheussiqueirahub/modular-video-ai-pipeline

---

⭐ **Star this repo** if you find it useful!
