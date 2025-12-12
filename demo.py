
import os
import sys
import argparse
import logging
import cv2
import numpy as np
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from src.detection.detector import ObjectDetector
from src.segmentation.segmenter import VideoSegmenter
from src.clustering.identifier import VisualIdentifier
from src.ocr.reader import SceneTextReader
from src.homography.transformer import PerspectiveTransformer
from src.events.analyzer import EventAnalyzer
from src.visualization.drawer import PipelineVisualizer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    setup_logging()
    logger = logging.getLogger("DemoPipeline")

    parser = argparse.ArgumentParser(description="AI Vision Pipeline Demo")
    parser.add_argument("--video_path", type=str, default=None, help="Path to input video")
    parser.add_argument("--mock", action="store_true", default=True, help="Use mock weights (default True for portfolio demo)")
    parser.add_argument("--output_path", type=str, default="output.mp4", help="Path to save processed video")
    
    args = parser.parse_args()

    # Initialize Modules
    logger.info("Initializing Pipeline Modules...")
    
    detector = ObjectDetector()
    segmenter = VideoSegmenter()
    identifier = VisualIdentifier()
    reader = SceneTextReader()
    # Mock points for homography: src (video quadrilateral), dst (top-down rect)
    transformer = PerspectiveTransformer(
        src_points=np.array([[0,0], [1280,0], [1280,720], [0,720]], dtype=np.float32), 
        dst_points=np.array([[0,0], [100,0], [100,200], [0,200]], dtype=np.float32)
    )
    analyzer = EventAnalyzer()
    visualizer = PipelineVisualizer()

    # Video Source
    if args.video_path and os.path.exists(args.video_path):
        cap = cv2.VideoCapture(args.video_path)
    else:
        logger.warning("No video provided or file not found. Using Mock Video Stream (Black Frames).")
        cap = None

    # Video Writer
    writer = None
    
    # Processing Loop
    logger.info("Starting Processing Loop...")
    
    frame_idx = 0
    while True:
        if cap:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            # Generate dummy frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            # Add noise to make it look alive
            noise = np.random.randint(0, 50, (720, 1280, 3), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.8, noise, 0.2, 0)
            if frame_idx > 100: # Stop after 100 frames in mock
                break
        
        # Initialize Writer once we know frame size
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be careful with codecs
            writer = cv2.VideoWriter(args.output_path, fourcc, 30, (w, h))

        # 1. Detection
        detections = detector.detect(frame)
        
        # 2. Segmentation & Tracking
        tracks = segmenter.track_objects(frame_idx, frame, detections)
        
        # 3. Clustering (Extract embeddings for tracks)
        crops = []
        for t in tracks:
            bbox = t['bbox']
            # Crop logic
            if bbox[3] > bbox[1] and bbox[2] > bbox[0]:
                crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if crop.size > 0:
                    crops.append(crop)
                else:
                    crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
            else:
                 crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
        
        embeddings = identifier.extract_embeddings(crops)
        cluster_labels = identifier.cluster_embeddings(embeddings)
        
        # updates tracks with cluster info
        for i, t in enumerate(tracks):
            if i < len(cluster_labels):
                t['cluster_id'] = int(cluster_labels[i])
                
        # 4. OCR (On demand, e.g. every 30 frames or if missing)
        if frame_idx % 30 == 0:
            for t in tracks:
                # Mock crop read
                bbox = t['bbox']
                if bbox[3] > bbox[1] and bbox[2] > bbox[0]:
                    crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                    text = reader.read_text(crop)
                    if text:
                        t['ocr_text'] = text
        
        # 5. Events
        events = analyzer.update(tracks, frame_idx)
        
        # 6. Visualization
        out_frame = visualizer.draw(frame, tracks, events)
        
        # Write Frame
        if writer:
            writer.write(out_frame)

        # Show/Save logic
        # For this demo script, we just log progress
        if frame_idx % 10 == 0:
            logger.info(f"Frame {frame_idx}: Processed {len(tracks)} objects. Events: {len(events)}")

        frame_idx += 1
        
        # Break for safety
        if frame_idx > 200:
            break

    logger.info("Processing Complete.")
    if cap:
        cap.release()
    if writer:
        writer.release()

if __name__ == "__main__":
    main()
