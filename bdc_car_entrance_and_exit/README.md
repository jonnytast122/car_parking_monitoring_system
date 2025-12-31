# BDC Car Entrance and Exit Counter - User Manual

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [How to Run](#how-to-run)
6. [Input Files](#input-files)
7. [Output Files](#output-files)
8. [Configuration](#configuration)
9. [How the Code Works](#how-the-code-works)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This system automatically counts vehicles entering and exiting the BDC facility using computer vision and deep learning. It consists of two main components:

- **Entrance Counter** (`car_entrance_counter.py`): Counts vehicles entering through the entrance gate
- **Exit Counter** (`car_exit_counter.py`): Counts vehicles exiting through the exit gate

Both systems use:
- **YOLO** (You Only Look Once) for vehicle detection
- **DeepSORT** for tracking vehicles across frames
- **Spatial-temporal filtering** to prevent duplicate counts

---

## System Requirements

### Hardware
- CPU: Intel Core i5 or better (GPU recommended for faster processing)
- RAM: 8GB minimum, 16GB recommended
- Storage: 2GB free space for model and output videos

### Software
- Python 3.8 or higher
- OpenCV (cv2)
- Ultralytics YOLO
- DeepSORT
- NumPy

---

## Installation

1. **Install Python dependencies**:
```bash
pip install opencv-python numpy ultralytics deep-sort-realtime
```

2. **Download YOLO model**:

   **Option A: Automatic Download (Recommended)**
   - The model will automatically download from Ultralytics GitHub when you first run the script
   - No manual download needed
   - Requires internet connection on first run
   - Model will be cached for future use

   **Option B: Manual Download**
   - Download `yolov12x.pt` manually from [Ultralytics GitHub](https://github.com/ultralytics/assets/releases)
   - Or download YOLO11: `yolo11x.pt` or YOLO8: `yolo8x.pt`
   - Place the model file in the same directory as the Python files
   - Update `MODEL_PATH` in the code if using a different model name:
     ```python
     MODEL_PATH = "yolo11x.pt"  # or your model filename
     ```

3. **Prepare video files**:
   - Place your entrance video as `BDC-Entrance-Cuted.mp4`
   - Place your exit video as `BDC-exit-Cuted.mp4`

---

## Quick Start

### Count Entrance Vehicles
```bash
cd bdc_car_entrance_and_exit
python car_entrance_counter.py
```

### Count Exit Vehicles
```bash
cd bdc_car_entrance_and_exit
python car_exit_counter.py
```

---

## How to Run

### Running Entrance Counter

1. **Navigate to the directory**:
   ```bash
   cd bdc_car_entrance_and_exit
   ```

2. **Run the script**:
   ```bash
   python car_entrance_counter.py
   ```

3. **During execution**:
   - A window will open showing the video with detections
   - Press `q` to quit early
   - Press `p` to pause/resume
   - Console will show real-time counting updates

4. **Output**:
   - Processed video saved as `output_entrance_count.mp4`
   - Console displays final statistics

### Running Exit Counter

1. **Navigate to the directory**:
   ```bash
   cd bdc_car_entrance_and_exit
   ```

2. **Run the script**:
   ```bash
   python car_exit_counter.py
   ```

3. **Same controls and output format as entrance counter**

---

## Input Files

### Required Files

| File | Description | Location |
|------|-------------|----------|
| `BDC-Entrance-Cuted.mp4` | Video of entrance gate | Same directory |
| `BDC-exit-Cuted.mp4` | Video of exit gate | Same directory |
| `yolov12x.pt` | YOLO model weights | Same directory or specified path |
| `car_entrance_counter.py` | Entrance counting script | Same directory |
| `car_exit_counter.py` | Exit counting script | Same directory |

### Video Requirements
- Format: MP4 (recommended)
- Resolution: Any (will be auto-detected)
- FPS: Any (will be auto-detected)
- Quality: Higher quality = better detection accuracy

---

## Output Files

### Video Output

**Entrance Output**: `output_entrance_count.mp4`
- Shows green detection zone
- Displays bounding boxes around detected vehicles
- Shows tracking IDs and "[COUNTED]" labels
- Displays real-time count in top-left corner

**Exit Output**: `output_exit_count.mp4`
- Shows blue detection zone
- Same visual elements as entrance output
- Different zone color to distinguish from entrance

### Console Output

During processing, you'll see:
```
Processing video: BDC-Entrance-Cuted.mp4
Frame size: 1920x1080
FPS: 30
Detection region: 4 points defined
--------------------------------------------------
[COUNT] Frame 150: Car ID 5 passed through detection zone! Total count: 1
        Already counted IDs: [5]
Processed 30 frames | FPS: 28.5 | Count: 1
...
==================================================
PROCESSING COMPLETE
==================================================
Total frames processed: 5400
Total time: 180.5 seconds
Average FPS: 29.9
Total vehicles entered: 42
Output saved to: output_entrance_count.mp4
==================================================
```

---

## Configuration

### Basic Configuration

Edit the `main()` function in each script:

```python
def main():
    # File paths
    VIDEO_PATH = "BDC-Entrance-Cuted.mp4"  # Input video
    MODEL_PATH = "yolov12x.pt"              # YOLO model
    OUTPUT_PATH = "output_entrance_count.mp4"  # Output video

    # Detection zone (polygon coordinates)
    DETECTION_REGION = [
        (41, 569),   # Bottom-left
        (46, 208),   # Top-left
        (452, 217),  # Top-right
        (443, 572)   # Bottom-right
    ]
```

### Advanced Parameters

#### Detection Sensitivity
```python
# In process_video() method:
results = self.model(frame,
    conf=0.3,  # Confidence threshold (0.0-1.0)
               # Lower = more detections (may include false positives)
               # Higher = fewer detections (may miss some vehicles)
    iou=0.7,   # IoU threshold for NMS
    imgsz=640  # Input image size
)
```

#### Duplicate Prevention
```python
# Time window for duplicate detection
TEMPORAL_THRESHOLD_MINUTES = 0.33  # 20 seconds

# Distance threshold for duplicate detection
SPATIAL_THRESHOLD_PIXELS = 150  # pixels
```

#### Tracking Parameters
```python
self.tracker = DeepSort(
    max_age=60,              # Frames to keep track alive
    n_init=2,                # Frames to confirm track
    max_iou_distance=0.95,   # Size change tolerance
    max_cosine_distance=0.9, # Appearance change tolerance
    nn_budget=300            # Feature storage
)
```

### Customizing Detection Zone

To change the detection zone:

1. **Extract a frame** from your video:
   ```bash
   python save_frame.py
   ```

2. **Open the frame** in an image editor

3. **Get pixel coordinates** of your desired zone corners

4. **Update DETECTION_REGION** in the code:
   ```python
   DETECTION_REGION = [
       (x1, y1),  # Bottom-left corner
       (x2, y2),  # Top-left corner
       (x3, y3),  # Top-right corner
       (x4, y4)   # Bottom-right corner
   ]
   ```

---

## How the Code Works

### System Architecture

```
┌─────────────────┐
│  Input Video    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  YOLO Detection │ ← Detects vehicles in each frame
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Overlaps │ ← Removes duplicate detections
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ DeepSORT Track  │ ← Tracks vehicles across frames
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Zone Detection  │ ← Checks if vehicle is in zone
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Count Vehicle   │ ← Counts when entering zone
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Output Video   │ ← Annotated video with counts
└─────────────────┘
```

### Processing Flow

1. **Frame-by-Frame Processing**:
   - Reads each frame from the input video
   - Processes at original video FPS

2. **Vehicle Detection (YOLO)**:
   - Detects cars in each frame
   - Filters by confidence threshold
   - Only detects class 2 (cars) from COCO dataset

3. **Duplicate Filtering**:
   - Removes overlapping detections of the same vehicle
   - Keeps detection with highest confidence

4. **Vehicle Tracking (DeepSORT)**:
   - Assigns unique IDs to each vehicle
   - Tracks vehicles across frames
   - Maintains ID even with brief occlusions

5. **Zone Detection**:
   - Checks if vehicle center is inside detection polygon
   - Uses OpenCV's `pointPolygonTest`

6. **Counting Logic**:
   - Counts when vehicle **enters** the zone (transition from outside → inside)
   - Each track ID counted only once
   - Prevents duplicate counts with spatial-temporal filtering

7. **Visualization**:
   - Draws detection zone (green for entrance, blue for exit)
   - Shows bounding boxes around vehicles
   - Displays track IDs and count status
   - Updates count display in real-time

8. **Output Generation**:
   - Saves annotated video
   - Prints statistics to console

### Key Components

#### CarEntranceCounter / CarExitCounter Class

**Main Methods**:
- `__init__()`: Initialize detector with video and model
- `process_video()`: Main processing loop
- `count_vehicle()`: Counting logic
- `is_point_in_region()`: Check if point is in zone
- `draw_annotations()`: Draw visual elements
- `filter_overlapping_detections()`: Remove duplicates

**Key Data Structures**:
- `counted_ids`: Set of already counted track IDs
- `inside_region`: Dict tracking which vehicles are in zone
- `track_positions`: Dict storing vehicle trajectories
- `counted_positions`: List of recent count positions (for duplicate prevention)

### Duplicate Prevention Strategy

The system uses **multiple layers** to prevent counting the same vehicle twice:

1. **Track ID-based**: Each track ID counted only once
2. **Spatial-temporal filtering**: Prevents counting if:
   - Another vehicle was counted within X pixels in last Y seconds
3. **DeepSORT persistence**: Maintains IDs even during brief occlusions
4. **Overlap filtering**: Removes duplicate YOLO detections

---

## Troubleshooting

### Problem: No detections / Low count

**Solutions**:
1. Lower confidence threshold:
   ```python
   conf=0.2  # Lower from 0.3
   ```
2. Increase image size:
   ```python
   imgsz=1280  # Increase from 640
   ```
3. Check detection zone coordinates
4. Verify video quality

### Problem: Too many false positives

**Solutions**:
1. Increase confidence threshold:
   ```python
   conf=0.5  # Increase from 0.3
   MIN_CONFIDENCE = 0.6  # Increase from 0.5
   ```
2. Verify detection zone doesn't include unwanted areas

### Problem: Same vehicle counted multiple times

**Solutions**:
1. Increase duplicate thresholds:
   ```python
   TEMPORAL_THRESHOLD_MINUTES = 0.5  # Increase time window
   SPATIAL_THRESHOLD_PIXELS = 200    # Increase distance
   ```
2. Adjust DeepSORT parameters for better tracking

### Problem: Video processing is slow

**Solutions**:
1. Use GPU acceleration (install `torch` with CUDA)
2. Reduce image size:
   ```python
   imgsz=416  # Smaller size
   ```
3. Disable display during processing:
   ```python
   counter.process_video(display=False)
   ```
4. Process fewer frames (modify code to skip frames)

### Problem: Display window freezes

**Solutions**:
1. Close other applications
2. Reduce display update frequency (already optimized in code)
3. Disable display:
   ```python
   counter.process_video(display=False)
   ```

---

## Summary

This vehicle counting system provides:
- **Automatic counting** of vehicles entering and exiting
- **Visual tracking** with unique IDs
- **Duplicate prevention** to ensure accurate counts
- **Configurable parameters** for different scenarios
- **Detailed output** videos and statistics

For questions or issues, refer to the code comments or consult the development team.
