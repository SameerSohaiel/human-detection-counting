"""
Main entry point for the Human Detection and Counting System.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.app.cli import parse_arguments
from src.app.video_processor import VideoProcessor
from src.detection.detector import HumanDetector
from src.analysis.crowd_analyzer import CrowdAnalyzer
from src.utils.logging_utils import setup_output_directory


def main():
    """Main application entry point."""
    # Parse CLI arguments
    args = parse_arguments()
    
    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Output directory: {output_dir}")
    
    # Initialize components
    print(f"Loading YOLO model: {args.model}")
    detector = HumanDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        tracker_type=args.tracker
    )
    
    analyzer = CrowdAnalyzer()
    
    # Process video
    processor = VideoProcessor(
        source=args.source,
        detector=detector,
        analyzer=analyzer,
        output_dir=output_dir,
        display=not args.no_display
    )
    
    try:
        print(f"Starting video processing from source: {args.source}")
        print("Press 'q' to quit, 's' to save current frame")
        processor.process()
        
        # Save analytics summary
        summary_path = output_dir / "analytics_summary.txt"
        analyzer.save_summary(summary_path)
        print(f"\nAnalytics summary saved to: {summary_path}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        processor.cleanup()
        print("Processing complete!")


if __name__ == "__main__":
    main()
