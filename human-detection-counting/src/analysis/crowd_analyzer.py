"""
Crowd analysis and statistics computation.
"""

from typing import List, Dict, Any
from pathlib import Path
import csv
from datetime import datetime


class CrowdAnalyzer:
    """
    Analyzes crowd metrics and maintains statistics.
    
    Tracks per-frame counts, unique individuals, and computes
    summary statistics like average, peak, and timing information.
    """
    
    def __init__(self):
        """Initialize the crowd analyzer."""
        self.frame_data = []
        self.peak_count = 0
        self.peak_frame = 0
        self.peak_timestamp = None
        
    def update(
        self,
        frame_index: int,
        detections: List[Dict[str, Any]],
        unique_count_total: int,
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """
        Update statistics with new frame data.
        
        Args:
            frame_index: Current frame index
            detections: List of detections from detector
            unique_count_total: Total unique individuals seen so far
            timestamp: Frame timestamp (defaults to current time)
            
        Returns:
            Dictionary with current statistics:
                - frame_index: Current frame number
                - current_count: People in current frame
                - unique_count_total: Total unique people seen
                - average_count: Running average
                - peak_count: Maximum count seen
                - peak_frame: Frame where peak occurred
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        current_count = len(detections)
        
        # Record frame data
        self.frame_data.append({
            'timestamp': timestamp,
            'frame_index': frame_index,
            'current_count': current_count,
            'unique_count_total': unique_count_total
        })
        
        # Update peak
        if current_count > self.peak_count:
            self.peak_count = current_count
            self.peak_frame = frame_index
            self.peak_timestamp = timestamp
        
        # Calculate running average
        average_count = self._calculate_average()
        
        return {
            'frame_index': frame_index,
            'current_count': current_count,
            'unique_count_total': unique_count_total,
            'average_count': average_count,
            'peak_count': self.peak_count,
            'peak_frame': self.peak_frame
        }
    
    def _calculate_average(self) -> float:
        """Calculate average people per frame."""
        if not self.frame_data:
            return 0.0
        
        total = sum(frame['current_count'] for frame in self.frame_data)
        return total / len(self.frame_data)
    
    def save_to_csv(self, output_path: Path):
        """
        Save per-frame data to CSV file.
        
        Args:
            output_path: Path to save CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'frame_index', 'current_count', 'unique_count_total']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for frame in self.frame_data:
                writer.writerow({
                    'timestamp': frame['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'frame_index': frame['frame_index'],
                    'current_count': frame['current_count'],
                    'unique_count_total': frame['unique_count_total']
                })
    
    def save_summary(self, output_path: Path):
        """
        Save analytics summary to text file.
        
        Args:
            output_path: Path to save summary file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        average_count = self._calculate_average()
        total_frames = len(self.frame_data)
        
        # Calculate peak time in seconds (if we have frame data)
        peak_time_str = "N/A"
        if self.frame_data and self.peak_frame < len(self.frame_data):
            first_timestamp = self.frame_data[0]['timestamp']
            if self.peak_timestamp:
                peak_seconds = (self.peak_timestamp - first_timestamp).total_seconds()
                minutes, seconds = divmod(peak_seconds, 60)
                peak_time_str = f"{int(minutes):02d}:{seconds:05.2f}"
        
        # Get final unique count
        unique_count = self.frame_data[-1]['unique_count_total'] if self.frame_data else 0
        
        summary = f"""Human Detection and Counting - Analytics Summary
{'=' * 50}

Processing Statistics:
  Total frames processed: {total_frames}
  
Crowd Metrics:
  Average people per frame: {average_count:.2f}
  Peak simultaneous count: {self.peak_count}
  Peak occurred at frame: {self.peak_frame}
  Peak time: {peak_time_str}
  
Tracking Statistics:
  Total unique individuals: {unique_count}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_path, 'w') as f:
            f.write(summary)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        return {
            'total_frames': len(self.frame_data),
            'average_count': self._calculate_average(),
            'peak_count': self.peak_count,
            'peak_frame': self.peak_frame,
            'unique_count': self.frame_data[-1]['unique_count_total'] if self.frame_data else 0
        }
