"""
Video processing and visualization.
"""

from typing import Optional
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime

from src.detection.detector import HumanDetector
from src.analysis.crowd_analyzer import CrowdAnalyzer


class VideoProcessor:
    """
    Processes video streams and coordinates detection, tracking, and analysis.
    
    Handles video I/O, frame-by-frame processing, visualization, and logging.
    """
    
    def __init__(
        self,
        source: str,
        detector: HumanDetector,
        analyzer: CrowdAnalyzer,
        output_dir: Path,
        display: bool = True
    ):
        """
        Initialize video processor.
        
        Args:
            source: Video source (0 for webcam, or path to video file)
            detector: Human detector instance
            analyzer: Crowd analyzer instance
            output_dir: Directory for output files
            display: Whether to show visualization window
        """
        self.source = source
        self.detector = detector
        self.analyzer = analyzer
        self.output_dir = Path(output_dir)
        self.display = display
        
        # Initialize video capture
        if source.isdigit():
            self.cap = cv2.VideoCapture(int(source))
        else:
            self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {source}")
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.frame_width}x{self.frame_height} @ {self.fps:.2f} FPS")
        if self.total_frames > 0:
            print(f"Total frames: {self.total_frames}")
        
        # CSV output path
        self.csv_path = self.output_dir / "frame_counts.csv"
        
        # Frame counter
        self.frame_index = 0
        self.start_time = None
        
    def process(self):
        """
        Main processing loop.
        
        Reads frames, performs detection/tracking, updates analytics,
        and handles visualization and logging.
        """
        self.start_time = datetime.now()
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("\nEnd of video stream")
                    break
                
                # Perform detection and tracking
                detections = self.detector.detect_and_track(frame)
                
                # Get unique count
                unique_count = self.detector.get_unique_count()
                
                # Update analytics
                current_time = datetime.now()
                stats = self.analyzer.update(
                    self.frame_index,
                    detections,
                    unique_count,
                    current_time
                )
                
                # Visualize
                if self.display:
                    vis_frame = self._create_visualization(frame, detections, stats)
                    cv2.imshow('Human Detection and Counting', vis_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nQuitting...")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        save_path = self.output_dir / f"frame_{self.frame_index:06d}.jpg"
                        cv2.imwrite(str(save_path), vis_frame)
                        print(f"\nSaved frame to {save_path}")
                
                # Print progress
                self._print_progress(stats)
                
                self.frame_index += 1
                
        finally:
            # Save CSV data
            print("\nSaving data...")
            self.analyzer.save_to_csv(self.csv_path)
            print(f"Frame data saved to: {self.csv_path}")
    
    def _create_visualization(
        self,
        frame: np.ndarray,
        detections: list,
        stats: dict
    ) -> np.ndarray:
        """
        Create visualization with detections and statistics overlay.
        
        Args:
            frame: Input frame
            detections: List of detections
            stats: Current statistics
            
        Returns:
            Annotated frame
        """
        # Draw detections
        vis_frame = self.detector.draw_detections(frame, detections)
        
        # Create info panel
        panel_height = 150
        panel = np.zeros((panel_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add statistics text
        y_offset = 30
        line_height = 30
        
        texts = [
            f"Frame: {stats['frame_index']} | Current: {stats['current_count']} people",
            f"Unique Total: {stats['unique_count_total']} people tracked",
            f"Average: {stats['average_count']:.2f} | Peak: {stats['peak_count']} (frame {stats['peak_frame']})",
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                panel,
                text,
                (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        # Add controls text
        cv2.putText(
            panel,
            "Press 'q' to quit | 's' to save frame",
            (10, panel_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1
        )
        
        # Combine frame and panel
        combined = np.vstack([vis_frame, panel])
        
        return combined
    
    def _print_progress(self, stats: dict):
        """
        Print processing progress to console.
        
        Args:
            stats: Current statistics
        """
        # Calculate processing speed
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            fps = (self.frame_index + 1) / elapsed if elapsed > 0 else 0
        else:
            fps = 0
        
        # Create progress string
        progress = f"Frame {self.frame_index:5d}"
        if self.total_frames > 0:
            percent = (self.frame_index / self.total_frames) * 100
            progress += f" ({percent:5.1f}%)"
        
        progress += f" | Count: {stats['current_count']:3d}"
        progress += f" | Unique: {stats['unique_count_total']:3d}"
        progress += f" | Avg: {stats['average_count']:5.2f}"
        progress += f" | FPS: {fps:5.1f}"
        
        # Print with carriage return to overwrite
        print(f"\r{progress}", end='', flush=True)
    
    def cleanup(self):
        """Release resources."""
        if self.cap:
            self.cap.release()
        if self.display:
            cv2.destroyAllWindows()
