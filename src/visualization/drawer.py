
import cv2
import numpy as np
import supervision as sv # Use supervision if available, else standard opencv

class PipelineVisualizer:
    """
    Draws analysis results onto the frame.
    Uses 'supervision' library style aesthetics or fallback to OpenCV.
    """
    def __init__(self):
        # Initialize annotators
        try:
            self.box_annotator = sv.BoxAnnotator()
            self.mask_annotator = sv.MaskAnnotator()
            self.label_annotator = sv.LabelAnnotator()
            self.use_supervision = True
        except ImportError:
            self.use_supervision = False

    def draw(self, frame: np.ndarray, tracks: list, events: list) -> np.ndarray:
        """
        Draws bounding boxes, masks, IDs, and events on the frame.
        """
        annotated_frame = frame.copy()

        if not tracks:
            return annotated_frame

        if self.use_supervision:
            # Convert pipeline tracks to supervision Detections object
            xyxy = np.array([t['bbox'] for t in tracks])
            mask = np.array([t['mask'] for t in tracks])
            confidence = np.array([0.9] * len(tracks)) # Placeholder
            class_id = np.array([t['class_id'] for t in tracks])
            tracker_id = np.array([t['id'] for t in tracks])

            detections = sv.Detections(
                xyxy=xyxy,
                mask=mask,
                confidence=confidence,
                class_id=class_id,
                tracker_id=tracker_id
            )

            # Annotate
            annotated_frame = self.mask_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = self.box_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Create labels: ID + Class + (Cluster ID if available)
            labels = []
            for t in tracks:
                label_text = f"#{t['id']} {t.get('label', '')}" 
                if 'cluster_id' in t:
                   label_text += f"-G{t['cluster_id']}"
                if 'ocr_text' in t and t['ocr_text']:
                   label_text += f" [{t['ocr_text']}]"
                labels.append(label_text)
                
            annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        else:
            # Fallback OpenCV drawing
            for t in tracks:
                x1, y1, x2, y2 = t['bbox']
                color = (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, f"ID: {t['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Events
        y_offset = 30
        for event in events:
            # Show events that happened recently (e.g., in this frame)
            if event['frame'] >= 0: # Check relevant logic
                 text = f"EVENT: {event['type']} - {event['details']}"
                 cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 y_offset += 30
                 
        return annotated_frame
