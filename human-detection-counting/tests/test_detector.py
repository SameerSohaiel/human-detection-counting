"""
Unit tests for the HumanDetector class.
"""

import pytest
import numpy as np
from src.detection.detector import HumanDetector


class TestHumanDetector:
    """Test cases for HumanDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return HumanDetector(model_name="yolov8n.pt", confidence_threshold=0.5)
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame for testing."""
        # Create a simple 640x480 BGR image
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some variation to make it more realistic
        frame[:, :, 0] = 100  # Blue channel
        frame[:, :, 1] = 150  # Green channel
        frame[:, :, 2] = 200  # Red channel
        return frame
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.model is not None
        assert detector.confidence_threshold == 0.5
        assert detector.person_class_id == 0
        assert len(detector.unique_ids) == 0
    
    def test_detect_and_track_format(self, detector, sample_frame):
        """Test that detect_and_track returns correct format."""
        detections = detector.detect_and_track(sample_frame)
        
        # Should return a list
        assert isinstance(detections, list)
        
        # If detections exist, check format
        for det in detections:
            assert 'bbox' in det
            assert 'confidence' in det
            assert 'track_id' in det or det['track_id'] is None
            assert 'class_id' in det
            assert len(det['bbox']) == 4
            assert 0 <= det['confidence'] <= 1
    
    def test_draw_detections(self, detector, sample_frame):
        """Test drawing detections on frame."""
        # Create mock detection
        detections = [{
            'bbox': [100, 100, 200, 300],
            'confidence': 0.85,
            'track_id': 1,
            'class_id': 0
        }]
        
        annotated = detector.draw_detections(sample_frame, detections)
        
        # Should return numpy array of same shape
        assert isinstance(annotated, np.ndarray)
        assert annotated.shape == sample_frame.shape
        
        # Frame should be modified (not identical to original)
        assert not np.array_equal(annotated, sample_frame)
    
    def test_unique_count_tracking(self, detector):
        """Test unique ID tracking."""
        initial_count = detector.get_unique_count()
        assert initial_count == 0
        
        # Manually add some IDs
        detector.unique_ids.add(1)
        detector.unique_ids.add(2)
        detector.unique_ids.add(3)
        
        assert detector.get_unique_count() == 3
        
        # Adding duplicate should not increase count
        detector.unique_ids.add(2)
        assert detector.get_unique_count() == 3
    
    def test_reset_tracking(self, detector):
        """Test tracking reset."""
        detector.unique_ids.add(1)
        detector.unique_ids.add(2)
        
        assert detector.get_unique_count() == 2
        
        detector.reset_tracking()
        assert detector.get_unique_count() == 0
    
    def test_color_generation(self):
        """Test color generation for track IDs."""
        color1 = HumanDetector._get_color_for_id(1)
        color2 = HumanDetector._get_color_for_id(2)
        color_none = HumanDetector._get_color_for_id(None)
        
        # Colors should be tuples of 3 integers
        assert isinstance(color1, tuple)
        assert len(color1) == 3
        assert all(isinstance(c, (int, np.integer)) for c in color1)
        
        # Same ID should produce same color
        color1_again = HumanDetector._get_color_for_id(1)
        assert color1 == color1_again
        
        # Different IDs should produce different colors (usually)
        assert color1 != color2
        
        # None should return default green
        assert color_none == (0, 255, 0)
