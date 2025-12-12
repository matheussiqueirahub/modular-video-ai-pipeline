
# AI Vision Pipeline

 A modular, production-ready computer vision pipeline designed for video analysis. It integrates state-of-the-art models for detection, segmentation, tracking, clustering, and OCR.

## 🏗 Architecture

The system processes video streams through the following stages:

1.  **Detection (RF-DETR)**: Identifies people and key objects.
2.  **Tracking & Segmentation (SAM2)**: Maintains object identity and generates consistent masks across frames.
3.  **Visual Embeddings (SigLIP)**: Extracts high-dimensional features from object crops.
4.  **Clustering (UMAP + K-Means)**: Groups objects based on visual appearance (e.g., team jerseys, uniform types).
5.  **OCR (SmolVLM2)**: Reads text from objects (e.g., jersey numbers/names) using a lightweight Vision-Language Model.
6.  **Spatial Mapping (Homography)**: Projects video coordinates to a top-down 2D map.
7.  **Event Detection**: Analyzes state changes over time (dwell time, proximity).

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Usage

Run the demo pipeline on a video file:

```bash
python demo.py --video_path input.mp4 --debug
```

## 📁 Project Structure

- `src/detection`: Object detection wrappers.
- `src/segmentation`: SAM2 integration for precise masking.
- `src/clustering`: Unsupervised learning for object re-id and grouping.
- `src/ocr`: Optical Character Recognition using VLMs.
- `src/homography`: Coordinate transformation tools.
- `src/events`: Event logic and state machines.
- `src/visualization`: Drawing tools for overlays.
