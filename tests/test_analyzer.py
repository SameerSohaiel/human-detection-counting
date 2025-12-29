"""
Unit tests for the CrowdAnalyzer class.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import csv

from src.analysis.crowd_analyzer import CrowdAnalyzer


class TestCrowdAnalyzer:
    """Test cases for CrowdAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create an analyzer instance for testing."""
        return CrowdAnalyzer()
    
    @pytest.fixture
    def sample_detections(self):
        """Create sample detections for testing."""
        return [
            {'bbox': [100, 100, 200, 300], 'confidence': 0.9, 'track_id': 1},
            {'bbox': [300, 150, 400, 350], 'confidence': 0.85, 'track_id': 2},
            {'bbox': [500, 200, 600, 400], 'confidence': 0.8, 'track_id': 3},
        ]
    
    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.frame_data) == 0
        assert analyzer.peak_count == 0
        assert analyzer.peak_frame == 0
        assert analyzer.peak_timestamp is None
    
    def test_update_basic(self, analyzer, sample_detections):
        """Test basic update functionality."""
        stats = analyzer.update(
            frame_index=0,
            detections=sample_detections,
            unique_count_total=3
        )
        
        assert stats['frame_index'] == 0
        assert stats['current_count'] == 3
        assert stats['unique_count_total'] == 3
        assert stats['average_count'] == 3.0
        assert stats['peak_count'] == 3
        assert stats['peak_frame'] == 0
    
    def test_update_multiple_frames(self, analyzer, sample_detections):
        """Test updating across multiple frames."""
        # Frame 0: 3 people
        analyzer.update(0, sample_detections, 3)
        
        # Frame 1: 2 people
        analyzer.update(1, sample_detections[:2], 4)
        
        # Frame 2: 5 people (new peak)
        more_detections = sample_detections + [
            {'bbox': [50, 50, 100, 150], 'confidence': 0.75, 'track_id': 4},
            {'bbox': [700, 100, 800, 300], 'confidence': 0.7, 'track_id': 5},
        ]
        stats = analyzer.update(2, more_detections, 5)
        
        assert stats['current_count'] == 5
        assert stats['average_count'] == pytest.approx((3 + 2 + 5) / 3)
        assert stats['peak_count'] == 5
        assert stats['peak_frame'] == 2
    
    def test_peak_tracking(self, analyzer, sample_detections):
        """Test that peak is tracked correctly."""
        # Start with 2
        analyzer.update(0, sample_detections[:2], 2)
        assert analyzer.peak_count == 2
        assert analyzer.peak_frame == 0
        
        # Increase to 5
        more_detections = sample_detections + [
            {'bbox': [1, 1, 50, 50], 'confidence': 0.8, 'track_id': 4},
            {'bbox': [2, 2, 60, 60], 'confidence': 0.8, 'track_id': 5},
        ]
        analyzer.update(1, more_detections, 5)
        assert analyzer.peak_count == 5
        assert analyzer.peak_frame == 1
        
        # Decrease to 1 (peak should remain)
        analyzer.update(2, sample_detections[:1], 5)
        assert analyzer.peak_count == 5
        assert analyzer.peak_frame == 1
    
    def test_average_calculation(self, analyzer, sample_detections):
        """Test average calculation."""
        analyzer.update(0, sample_detections[:1], 1)  # 1 person
        analyzer.update(1, sample_detections[:2], 2)  # 2 people
        analyzer.update(2, sample_detections, 3)      # 3 people
        
        stats = analyzer.get_summary_stats()
        expected_avg = (1 + 2 + 3) / 3
        assert stats['average_count'] == pytest.approx(expected_avg)
    
    def test_csv_export(self, analyzer, sample_detections):
        """Test CSV export functionality."""
        # Add some data
        timestamp1 = datetime(2024, 1, 1, 10, 0, 0)
        timestamp2 = datetime(2024, 1, 1, 10, 0, 1)
        
        analyzer.update(0, sample_detections[:2], 2, timestamp1)
        analyzer.update(1, sample_detections, 3, timestamp2)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            temp_path = Path(f.name)
        
        try:
            analyzer.save_to_csv(temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            assert len(rows) == 2
            assert rows[0]['frame_index'] == '0'
            assert rows[0]['current_count'] == '2'
            assert rows[1]['frame_index'] == '1'
            assert rows[1]['current_count'] == '3'
            
        finally:
            temp_path.unlink()
    
    def test_summary_export(self, analyzer, sample_detections):
        """Test summary export functionality."""
        # Add some data
        analyzer.update(0, sample_detections[:1], 1)
        analyzer.update(1, sample_detections[:2], 2)
        analyzer.update(2, sample_detections, 3)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = Path(f.name)
        
        try:
            analyzer.save_summary(temp_path)
            
            # Read and verify
            with open(temp_path, 'r') as f:
                content = f.read()
            
            assert 'Total frames processed: 3' in content
            assert 'Peak simultaneous count: 3' in content
            assert 'Total unique individuals: 3' in content
            
        finally:
            temp_path.unlink()
    
    def test_get_summary_stats(self, analyzer, sample_detections):
        """Test summary statistics retrieval."""
        analyzer.update(0, sample_detections[:1], 1)
        analyzer.update(1, sample_detections[:2], 2)
        analyzer.update(2, sample_detections, 3)
        
        stats = analyzer.get_summary_stats()
        
        assert stats['total_frames'] == 3
        assert stats['peak_count'] == 3
        assert stats['peak_frame'] == 2
        assert stats['unique_count'] == 3
        assert stats['average_count'] == pytest.approx(2.0)
