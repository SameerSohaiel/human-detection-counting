"""
Human detection and tracking using YOLOv8.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO
import cv2


class HumanDetector:
    """
    Handles human detection and tracking using YOLOv8.
    
    Attributes:
        model: YOLOv8 model instance
        confidence_threshold: Minimum confidence for detections
        tracker_type: Type of tracker to use ('botsort' or 'bytetrack')
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        tracker_type: str = "botsort"
    ):
        """
        Initialize the human detector.
        
        Args:
            model_name: YOLOv8 model name (e.g., 'yolov8n.pt', 'yolov8m.pt')
            confidence_threshold: Minimum confidence threshold for detections
            tracker_type: Tracker algorithm ('botsort' or 'bytetrack')
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type
        
        # COCO dataset person class ID is 0
        self.person_class_id = 0
        
        # Track unique IDs seen
        self.unique_ids = set()
        
    def detect_and_track(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect and track humans in a frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: Detection confidence score
                - track_id: Unique tracking ID
                - class_id: Class ID (always 0 for person)
        """
        # Run tracking
        results = self.model.track(
            frame,
            persist=True,
            tracker=f"{self.tracker_type}.yaml",
            classes=[self.person_class_id],
            conf=self.confidence_threshold,
            verbose=False
        )
        
        detections = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates
                bbox = boxes.xyxy[i].cpu().numpy()
                
                # Get confidence
                confidence = float(boxes.conf[i].cpu().numpy())
                
                # Get track ID (if available)
                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].cpu().numpy())
                    self.unique_ids.add(track_id)
                
                # Get class ID
                class_id = int(boxes.cls[i].cpu().numpy())
                
                detections.append({
                    'bbox': bbox.tolist(),
                    'confidence': confidence,
                    'track_id': track_id,
                    'class_id': class_id
                })
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: List of detections from detect_and_track()
            
        Returns:
            Frame with drawn detections
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det['confidence']
            track_id = det['track_id']
            
            # Draw bounding box
            color = self._get_color_for_id(track_id)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            if track_id is not None:
                label = f"ID:{track_id} {confidence:.2f}"
            else:
                label = f"Person {confidence:.2f}"
            
            # Draw label background
            (label_width, label_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return annotated_frame
    
    def get_unique_count(self) -> int:
        """
        Get the total number of unique individuals tracked.
        
        Returns:
            Number of unique track IDs seen
        """
        return len(self.unique_ids)
    
    def reset_tracking(self):
        """Reset tracking state (useful when switching videos)."""
        self.unique_ids.clear()
        # Note: YOLO tracker state persists in the model
        # For complete reset, reinitialize the model
    
    @staticmethod
    def _get_color_for_id(track_id: Optional[int]) -> tuple:
        """
        Generate a consistent color for each track ID.
        
        Args:
            track_id: Tracking ID
            
        Returns:
            BGR color tuple
        """
        if track_id is None:
            return (0, 255, 0)  # Green for untracked
        
        # Generate color from ID using simple hash
        np.random.seed(track_id)
        color = tuple(map(int, np.random.randint(0, 255, 3)))
        return color
