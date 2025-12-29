"""
Logging and file system utilities.
"""

from pathlib import Path
from datetime import datetime


def setup_output_directory(output_dir: str) -> Path:
    """
    Create output directory with timestamp subdirectory.
    
    Args:
        output_dir: Base output directory path
        
    Returns:
        Path to created output directory
    """
    base_dir = Path(output_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_dir / timestamp
    
    # Create directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path
