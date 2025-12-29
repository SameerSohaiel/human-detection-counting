# Human Detection and Counting System

A production-ready computer vision system for detecting, tracking, and counting humans in video streams using YOLOv8 and object tracking algorithms.

## Features

- **Real-time human detection** using YOLOv8
- **Object tracking** to count unique individuals across frames
- **Live video** (webcam) and **recorded video** support
- **Per-frame CSV logging** with timestamps and counts
- **Crowd analytics**: average count, peak count, timing
- **Configurable CLI** interface
- **Modular design** for easy robot integration

## System Requirements

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for real-time performance)
- Webcam or video files for input

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/human-detection-counting.git
cd human-detection-counting
```

### 2. Create and activate virtual environment

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download YOLO model (automatic on first run)

The YOLOv8 model will be downloaded automatically when you run the program for the first time. If you want to download it manually:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Available models:**
- `yolov8n.pt` - Nano (fastest, least accurate)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium (recommended balance)
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra large (most accurate, slowest)

## Usage

### Basic Examples

**Process webcam feed:**
```bash
python main.py --source 0
```

**Process video file:**
```bash
python main.py --source path/to/video.mp4
```

**Use a different YOLO model:**
```bash
python main.py --source 0 --model yolov8m.pt
```

**Adjust confidence threshold:**
```bash
python main.py --source video.mp4 --confidence 0.6
```

**Custom output directory:**
```bash
python main.py --source video.mp4 --output-dir results/
```

**Disable visualization (headless mode):**
```bash
python main.py --source video.mp4 --no-display
```

### All CLI Arguments

```
--source: Video source (0 for webcam, or path to video file) [required]
--model: YOLO model name (default: yolov8n.pt)
--confidence: Detection confidence threshold 0-1 (default: 0.5)
--output-dir: Output directory for logs and analytics (default: output/)
--no-display: Run without visualization window
--tracker: Tracking algorithm: botsort or bytetrack (default: botsort)
```

### Example with all options:

```bash
python main.py \
    --source videos/crowd.mp4 \
    --model yolov8m.pt \
    --confidence 0.6 \
    --output-dir results/experiment1/ \
    --tracker bytetrack
```

## Output Files

The system generates the following outputs in the specified output directory:

1. **`frame_counts.csv`** - Per-frame data:
   ```
   timestamp,frame_index,current_count,unique_count_total
   2024-01-15 10:30:45.123,0,3,3
   2024-01-15 10:30:45.156,1,4,5
   ```

2. **`analytics_summary.txt`** - Overall statistics:
   ```
   Total frames: 1500
   Average people per frame: 4.2
   Peak count: 12 (frame 450 at 00:25.5)
   Unique individuals tracked: 23
   ```

## Running Tests

```bash
pytest tests/ -v
```

For coverage report:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
human-detection-counting/
├── main.py                 # Entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── detection/         # YOLO detection and tracking
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── tracker.py
│   ├── analysis/          # Crowd analytics
│   │   ├── __init__.py
│   │   └── crowd_analyzer.py
│   ├── app/               # Application logic
│   │   ├── __init__.py
│   │   ├── video_processor.py
│   │   └── cli.py
│   └── utils/             # Shared utilities
│       ├── __init__.py
│       └── logging_utils.py
└── tests/
    ├── __init__.py
    ├── test_detector.py
    ├── test_analyzer.py
    └── fixtures/
        └── sample_detections.json
```

## Key Concepts

### YOLO (You Only Look Once)

YOLO is a state-of-the-art object detection algorithm that:
- Processes images in a single forward pass (hence "only look once")
- Divides images into a grid and predicts bounding boxes and class probabilities
- Achieves real-time performance with high accuracy
- YOLOv8 is the latest version with improved accuracy and speed

### Object Tracking

Object tracking assigns persistent IDs to detected objects across frames:
- **BoT-SORT** (default): Bot-SORT combines detection and ReID for robust tracking
- **ByteTrack**: Lighter-weight tracker using Kalman filtering and IoU matching

Tracking enables:
- Counting unique individuals (not just per-frame counts)
- Understanding movement patterns
- Reducing false positives from flickering detections

### Crowd Analysis

The system computes:
- **Current count**: People in the current frame
- **Unique count**: Total unique individuals seen across all frames
- **Average count**: Mean people per frame
- **Peak count**: Maximum simultaneous people and when it occurred

## Integration with Robot Systems

This system is designed for easy integration:

1. **Modular API**: Use `HumanDetector` and `CrowdAnalyzer` classes directly
2. **ROS/ROS2**: Wrap in a node that subscribes to image topics
3. **Custom pipelines**: Import detection module and feed frames programmatically

Example integration:

```python
from src.detection.detector import HumanDetector
from src.analysis.crowd_analyzer import CrowdAnalyzer

detector = HumanDetector(model_name="yolov8n.pt")
analyzer = CrowdAnalyzer()

# In your robot's perception loop:
for frame in camera_stream:
    detections = detector.detect_and_track(frame)
    stats = analyzer.update(frame_idx, detections)
    
    # Use stats for robot decision-making
    if stats['current_count'] > threshold:
        robot.stop()
```

## Performance Optimization

- **GPU acceleration**: Automatically uses CUDA if available
- **Model selection**: Use smaller models (yolov8n, yolov8s) for speed
- **Resolution**: Resize large frames before processing
- **Confidence threshold**: Higher values reduce false positives and speed up processing

## Troubleshooting

**Issue: Low FPS**
- Use a smaller YOLO model (yolov8n.pt)
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce input video resolution

**Issue: Too many false detections**
- Increase confidence threshold: `--confidence 0.6`
- Use a larger, more accurate model: `--model yolov8m.pt`

**Issue: Lost tracking IDs**
- Switch tracker: `--tracker botsort`
- Reduce occlusion in the scene
- Ensure good lighting conditions

## License

MIT License - see LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this system in research, please cite:

```bibtex
@software{human_detection_counting,
  title={Human Detection and Counting System},
  author={Sameer Sohaiel M},
  year={2024},
  url={https://github.com/SameerSohaiel/human-detection-counting}
}
```

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
