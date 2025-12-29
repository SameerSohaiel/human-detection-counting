"""
Application module for video processing and CLI interface.
"""

from .video_processor import VideoProcessor
from .cli import parse_arguments

__all__ = ["VideoProcessor", "parse_arguments"]
