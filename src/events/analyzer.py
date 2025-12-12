
import logging
from collections import defaultdict
import time

class EventAnalyzer:
    """
    State machine for detecting temporal events based on object tracks.
    Examples: Dwell time violation, zone entry/exit, fall detection.
    """
    def __init__(self, fps: int = 30):
        self.logger = logging.getLogger(__name__)
        self.fps = fps
        self.track_history = defaultdict(list) # id -> list of (x, y, timestamp)
        self.events = []
        
        # Configuration for "Dwell" event
        self.min_dwell_frames = fps * 3 # 3 seconds

    def update(self, tracks: list, frame_idx: int):
        """
        Updates state with new tracks and detects events.
        """
        current_ids = set()
        
        for track in tracks:
            obj_id = track['id']
            bbox = track['bbox']
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            
            self.track_history[obj_id].append((center[0], center[1], frame_idx))
            current_ids.add(obj_id)
            
            self._check_dwell_event(obj_id, frame_idx)

        # Cleanup lost tracks (optional, effectively handled by forgetting old ids in logic)
        return self.events

    def _check_dwell_event(self, obj_id: int, current_frame: int):
        """
        Checks if an object has stayed relatively stationary for too long.
        This is a simplified example.
        """
        history = self.track_history[obj_id]
        if len(history) < self.min_dwell_frames:
            return

        # Check last N seconds
        recent_history = history[-self.min_dwell_frames:]
        start_frame = recent_history[0][2]
        
        if current_frame - start_frame >= self.min_dwell_frames:
            # Check displacement
            start_pos = np.array([recent_history[0][0], recent_history[0][1]])
            curr_pos = np.array([recent_history[-1][0], recent_history[-1][1]])
            distance = np.linalg.norm(curr_pos - start_pos)
            
            if distance < 50: # Threshold in pixels
                # Trigger Event
                self.events.append({
                    "frame": current_frame,
                    "type": "STATIONARY_WARNING",
                    "object_id": obj_id,
                    "details": f"Object {obj_id} stationary for > 3s"
                })
                # Debounce: Clear history or mark as reported to avoid spam
                # For simplicity, we just log it.
import numpy as np
