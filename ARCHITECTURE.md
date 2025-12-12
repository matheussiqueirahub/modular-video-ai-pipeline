# Architecture Documentation

## Overview

The AI Vision Pipeline is designed with a **modular, layered architecture** that separates concerns and allows for easy extension and modification.

## Design Principles

1. **Modularity**: Each component is self-contained with clear interfaces
2. **Flexibility**: Easy to swap implementations or add new features
3. **Testability**: Mock modes allow testing without heavy dependencies
4. **Performance**: Optimized for real-time or near-real-time processing
5. **Maintainability**: Clean code with logging and error handling

## System Architecture

### Layer 1: Input Processing
- **Video Capture**: Handles video streams or file inputs
- **Frame Extraction**: Provides frames to the pipeline

### Layer 2: Detection & Segmentation
- **ObjectDetector** (`src/detection/detector.py`)
  - Wraps RF-DETR or similar detection models
  - Returns bounding boxes and class labels
  - Mock mode generates synthetic detections

- **VideoSegmenter** (`src/segmentation/segmenter.py`)
  - Uses SAM2 for temporal consistent segmentation
  - Maintains tracking state across frames
  - Outputs precise masks for each tracked object

### Layer 3: Feature Extraction & Analysis
- **VisualIdentifier** (`src/clustering/identifier.py`)
  - Extracts embeddings using SigLIP
  - Reduces dimensionality with UMAP (optional)
  - Clusters objects using K-Means
  - Use case: Team identification, uniform classification

- **SceneTextReader** (`src/ocr/reader.py`)
  - Uses Vision-Language Model (SmolVLM2)
  - More robust than traditional OCR for challenging text
  - Use case: Jersey numbers, license plates, signage

### Layer 4: Spatial Understanding
- **PerspectiveTransformer** (`src/homography/transformer.py`)
  - Computes homography matrices
  - Maps image coordinates to real-world or top-down coordinates
  - Use case: Court/field mapping, spatial analytics

### Layer 5: Temporal Analysis
- **EventAnalyzer** (`src/events/analyzer.py`)
  - State machine for event detection
  - Tracks object history and trajectories
  - Detects patterns: dwell time, zone violations, anomalies
  - Extensible event vocabulary

### Layer 6: Output & Visualization
- **PipelineVisualizer** (`src/visualization/drawer.py`)
  - Annotates frames with detections, masks, IDs
  - Displays cluster assignments and OCR results
  - Shows detected events with color-coded overlays
  - Uses Supervision library for professional aesthetics

### Layer 7: Orchestration
- **demo.py**: Main pipeline coordinator
  - Initializes all modules
  - Manages frame-by-frame processing
  - Handles video I/O
  - Coordinates data flow between modules

## Data Flow

```
Video Frame
    ↓
[Detection] → Bounding Boxes
    ↓
[Segmentation] → Masks + Track IDs
    ↓
[Clustering] → Group IDs (e.g., Team A/B)
    ↓
[OCR] → Text Labels (e.g., "23")
    ↓
[Homography] → Spatial Coordinates
    ↓
[Events] → Event List (e.g., "Player 23 entered zone")
    ↓
[Visualization] → Annotated Frame
    ↓
Output Video
```

## Key Design Decisions

### Why Mock Mode?
- Allows development and testing without 20GB+ of model weights
- Enables CI/CD pipelines to run tests quickly
- Provides a template for custom implementations

### Why Modular Classes?
- Each class can be tested independently
- Easy to swap implementations (e.g., use YOLO instead of RF-DETR)
- Clear separation allows parallel development

### Why State Machines for Events?
- Temporal events require memory of past states
- State machines are intuitive and debuggable
- Easy to add new event types without modifying core logic

## Extension Points

### Adding a New Detection Model
1. Create a new class in `src/detection/`
2. Implement `detect(frame)` method returning `[{bbox, label, score}]`
3. Update `demo.py` to use the new detector

### Adding a New Event Type
1. Add detection logic to `EventAnalyzer._check_xxx_event()`
2. Define trigger conditions and thresholds
3. Append to `self.events` list

### Adding a New Output Format
1. Extend `PipelineVisualizer` with new drawing functions
2. Or create a separate exporter (e.g., JSON, protobuf)

## Performance Considerations

### Bottlenecks
1. **Model Inference**: Detection and segmentation are GPU-bound
2. **Clustering**: UMAP can be slow with many objects
3. **Video I/O**: Codec choice affects encoding speed

### Optimizations
1. **Batch Processing**: Process multiple frames simultaneously
2. **Async I/O**: Separate video reading/writing threads
3. **Sparse OCR**: Only run OCR when needed (not every frame)
4. **Downsample Embeddings**: Use PCA before UMAP for speed

## Dependencies Graph

```
demo.py
  ├─ detector (torch, transformers)
  ├─ segmenter (torch, sam2)
  ├─ identifier (transformers, sklearn, umap)
  ├─ reader (transformers)
  ├─ transformer (opencv)
  ├─ analyzer (numpy)
  └─ visualizer (supervision, opencv)
```

## Testing Strategy

### Unit Tests
- Test each module independently with mock data
- Verify input/output contracts
- Check edge cases (empty detections, invalid frames)

### Integration Tests
- Test full pipeline with sample video
- Verify data flows correctly between modules
- Check output video quality

### Performance Tests
- Benchmark FPS on standard hardware
- Memory profiling to detect leaks
- GPU utilization monitoring

## Future Enhancements

1. **Multi-Camera Support**: Track objects across camera views
2. **Real-Time Streaming**: Process live RTSP/RTMP streams
3. **Cloud Deployment**: Containerize with Docker, deploy to Kubernetes
4. **Web UI**: Dashboard for configuration and monitoring
5. **Database Integration**: Store events and trajectories for analytics
