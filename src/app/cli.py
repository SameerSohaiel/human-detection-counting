"""
Command-line interface for the human detection system.
"""

import argparse
from pathlib import Path


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Human Detection and Counting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process webcam feed
  python main.py --source 0
  
  # Process video file
  python main.py --source video.mp4
  
  # Use medium model with higher confidence
  python main.py --source video.mp4 --model yolov8m.pt --confidence 0.6
  
  # Run without display (headless)
  python main.py --source video.mp4 --no-display
        """
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Video source: 0 for webcam, or path to video file'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='YOLO model name (default: yolov8n.pt). Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Detection confidence threshold 0-1 (default: 0.5)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Output directory for logs and analytics (default: output/)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Run without visualization window (headless mode)'
    )
    
    parser.add_argument(
        '--tracker',
        type=str,
        default='botsort',
        choices=['botsort', 'bytetrack'],
        help='Tracking algorithm to use (default: botsort)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0 or args.confidence > 1:
        parser.error("Confidence threshold must be between 0 and 1")
    
    return args
