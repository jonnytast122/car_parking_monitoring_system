# Mall Car Entrance and Exit Counter - User Manual

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
10. [Advanced Features](#advanced-features)
11. [Troubleshooting](#troubleshooting)

---

## Overview

This system automatically tracks and counts vehicles entering and exiting a mall parking area using a **single camera with two detection zones**. Unlike the BDC system which uses separate cameras for entrance and exit, this system monitors both directions simultaneously.

### Key Features
- **Dual-zone detection**: Separate IN (green) and OUT (blue) detection zones
- **Bidirectional counting**: Tracks vehicles entering AND exiting in one video
- **Smart duplicate prevention**: Prevents counting the same vehicle twice
- **Stationary vehicle handling**: Maintains tracking even when cars stop (e.g., waiting at barriers)
- **Spatial-temporal filtering**: Detects when DeepSORT assigns new IDs to the same car
- **Real-time visualization**: Shows counts, tracking boxes, and zone overlays

### Use Cases
- Shopping mall parking entrance/exit monitoring
- Parking garage occupancy tracking
- Traffic flow analysis
- Capacity management

---

## System Requirements

### Hardware
- **CPU**: Intel Core i5 or better (GPU recommended for faster processing)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 3GB free space (for model, input video, and output)
- **GPU** (Optional): NVIDIA GPU with CUDA for 3-5x faster processing

### Software
- **Python**: 3.8 or higher
- **OpenCV**: Computer vision library
- **Ultralytics YOLO**: Object detection model
- **DeepSORT**: Multi-object tracking
- **NumPy**: Numerical computing

---

## Installation

### Step 1: Install Python Dependencies

```bash
pip install opencv-python numpy ultralytics deep-sort-realtime
```

### Step 2: Download YOLO Model

**Option A: Automatic Download (Recommended)**
- The model automatically downloads from Ultralytics GitHub on first run
- No manual steps needed
- Requires internet connection initially
- Model cached in `~/.ultralytics/` for future use

**Option B: Manual Download**
- Download `yolov12x.pt` from [Ultralytics GitHub](https://github.com/ultralytics/assets/releases)
- Alternative models: `yolo11x.pt` or `yolo8x.pt`
- Place in `mall_car_entrance_and_exit/` directory
- Update `MODEL_PATH` if using different model:
  ```python
  MODEL_PATH = "yolo11x.pt"
  ```

### Step 3: Prepare Video File

Place your surveillance video in the directory:
- Default name: `20251223172447350.MP4`
- Or update `VIDEO_PATH` in code to match your filename

---

## Quick Start

### Run the Counter

```bash
cd mall_car_entrance_and_exit
python car_in_out_counter.py
```

### Expected Output

The system will:
1. Load video and YOLO model
2. Display processing window with:
   - Green IN zone
   - Blue OUT zone
   - Tracking boxes around vehicles
   - Real-time counts
3. Save processed video as `output_in_out_count.mp4`
4. Print statistics to console

---

## How to Run

### Basic Usage

1. **Navigate to directory**:
   ```bash
   cd mall_car_entrance_and_exit
   ```

2. **Run the script**:
   ```bash
   python car_in_out_counter.py
   ```

3. **Monitor progress**:
   - Watch the display window (updates in real-time)
   - Check console for detailed logs
   - Press `q` to quit early
   - Press `p` to pause/resume

4. **Review results**:
   - Check console for final statistics
   - Watch `output_in_out_count.mp4` for visual verification

### Advanced Usage

**Disable display** (faster processing):
```python
counter.process_video(output_path=OUTPUT_PATH, display=False)
```

**Custom output path**:
```python
counter.process_video(output_path="my_output.mp4", display=True)
```

---

## Input Files

### Required Files

| File | Description | Default Path | Required |
|------|-------------|--------------|----------|
| `car_in_out_counter.py` | Main processing script | Same directory | ✅ Yes |
| `20251223172447350.MP4` | Input video | `video/20251223172447350.MP4` | ✅ Yes |
| `yolov12x.pt` | YOLO model | Auto-downloaded or manual | ✅ Yes |

### Video Requirements

- **Format**: MP4, AVI, MOV (any OpenCV-supported format)
- **Resolution**: Any (auto-detected)
- **FPS**: Any (auto-detected)
- **Duration**: Any length
- **Quality**: Higher quality = better accuracy
- **View**: Should show both entrance and exit zones clearly

### Recommended Video Setup

For best results:
- Fixed camera position (no movement)
- Clear view of both zones
- Good lighting conditions
- Minimal obstructions
- 720p or higher resolution

---

## Output Files

### Video Output

**File**: `output_in_out_count.mp4`

**Visual Elements**:
- **Green zone**: IN detection area (vehicles entering)
- **Blue zone**: OUT detection area (vehicles exiting)
- **Bounding boxes**: Colored boxes around each detected vehicle
- **Track IDs**: Unique identifier for each vehicle
- **Status labels**: `[IN]` or `[OUT]` markers on counted vehicles
- **Count display**: Real-time IN and OUT counts (top-left corner)

**Video Properties**:
- Same resolution as input video
- Same FPS as input video
- MP4 format with H.264 codec

### Console Output

**During Processing**:
```
Processing video: video/20251223172447350.MP4
Frame size: 1920x1080
FPS: 30
IN region: 4 points defined
OUT region: 4 points defined
--------------------------------------------------
[IN] Frame 245: Car ID 3 entered! Total IN: 1 | All IN IDs: [3]
[OUT] Frame 892: Car ID 7 exited! Total OUT: 1 | All OUT IDs: [7]
[DUPLICATE DETECTED] Frame 1205: Track ID 12 is the same car as already counted ID 3 - preventing re-count
Processed 30 frames | FPS: 28.3 | IN: 5 | OUT: 3
Processed 60 frames | FPS: 27.9 | IN: 8 | OUT: 5
...
```

**Final Statistics**:
```
==================================================
PROCESSING COMPLETE
==================================================
Total frames processed: 12450
Total time: 421.3 seconds
Average FPS: 29.5

VEHICLE COUNTING (DUPLICATE-FREE):
  Total vehicles IN: 145
  Unique IN track IDs: [1, 3, 5, 7, 9, ...]
  Total vehicles OUT: 138
  Unique OUT track IDs: [2, 4, 6, 8, 10, ...]
  Net difference (IN - OUT): 7

Output saved to: output_in_out_count.mp4
==================================================
```

---

## Configuration

### Basic Configuration

Edit `main()` function in `car_in_out_counter.py`:

```python
def main():
    # File paths
    VIDEO_PATH = "video/20251223172447350.MP4"  # Input video
    MODEL_PATH = "yolov12x.pt"                   # YOLO model
    OUTPUT_PATH = "output_in_out_count.mp4"      # Output video

    # IN detection region (green zone) - vehicles entering
    IN_REGION = [
        (416, 481),   # Bottom-left corner
        (1536, 377),  # Top-left corner
        (1593, 868),  # Top-right corner
        (449, 977)    # Bottom-right corner
    ]

    # OUT detection region (blue zone) - vehicles exiting
    OUT_REGION = [
        (354, 1058),  # Bottom-left corner
        (1555, 1001), # Top-left corner
        (1598, 1535), # Top-right corner
        (316, 1597)   # Bottom-right corner
    ]
```

### Detection Parameters

**Confidence Threshold** (line 356):
```python
conf=0.8  # Detection confidence (0.0-1.0)
          # Lower = more detections (may include false positives)
          # Higher = fewer detections (may miss vehicles)
          # Default: 0.8 (80% confidence)
```

**Minimum Confidence Filter** (line 360):
```python
MIN_CONFIDENCE = 0.6  # Additional filter
                       # Default: 0.6 (60%)
```

**IoU Threshold** (line 356):
```python
iou=0.3  # Non-maximum suppression threshold
         # Lower = more aggressive duplicate removal
         # Higher = keep more overlapping detections
         # Default: 0.3 (aggressive)
```

**Image Size** (line 356):
```python
imgsz=640  # YOLO input size
           # Options: 416, 640, 1280, 1920
           # Larger = better small object detection (slower)
           # Default: 640 (balanced)
```

### Tracking Parameters

**DeepSORT Configuration** (lines 27-33):
```python
self.tracker = DeepSort(
    max_age=50,              # Frames to keep track alive (default: 50)
    n_init=3,                # Frames to confirm track (default: 3)
    max_iou_distance=0.8,    # Size change tolerance (default: 0.8)
    max_cosine_distance=0.5, # Appearance tolerance (default: 0.5)
    nn_budget=100            # Feature storage (default: 100)
)
```

**Increasing `max_age`**: Better for slow-moving or stationary vehicles
**Decreasing `n_init`**: Faster ID assignment (but more false IDs)
**Increasing distance thresholds**: More lenient tracking (handles appearance changes better)

### Duplicate Detection

**Spatial Threshold** (line 152):
```python
distance_threshold=100  # Pixels
                        # If new track within 100px of lost counted track
                        # = likely same vehicle with new ID
```

**Temporal Window** (line 169):
```python
if current_frame - lost_frame > 150:  # Frames
                                       # Tracks lost >150 frames ago
                                       # are no longer checked
```

---

## How the Code Works

### System Architecture

```
┌──────────────────┐
│  Input Video     │ Single camera view with 2 zones
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ YOLO Detection   │ Detect cars & trucks (classes 2, 7)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Filter Overlaps  │ Remove duplicate detections (same car detected twice)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Zone Filtering   │ Only track vehicles inside IN or OUT zones
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ DeepSORT Track   │ Assign unique IDs, track across frames
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Position Track   │ Monitor active/lost tracks for duplicate detection
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Spatial Duplicate│ Check if new ID = same car as lost counted ID
│    Detection     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Zone Entry Check │ Did vehicle enter IN zone? OUT zone?
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Count Vehicle    │ Count only once per zone, never in both zones
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Annotate & Save  │ Draw zones, boxes, IDs, counts
└──────────────────┘
```

### Processing Flow

#### 1. Frame Reading
- Reads video frame by frame
- Processes at original video FPS
- Handles all standard video formats

#### 2. Vehicle Detection (YOLO)
- Detects cars (class 2) and trucks (class 7) from COCO dataset
- Applies confidence threshold (0.8)
- Uses aggressive NMS to prevent duplicate detections
- Filters out motorcycles, buses, and other vehicles

#### 3. Overlap Filtering
- Removes duplicate YOLO detections of same vehicle
- Keeps detection with highest confidence
- Uses IoU (Intersection over Union) calculation
- Threshold: 0.3 (aggressive filtering)

#### 4. Zone-Based Filtering
- Checks vehicle center point location
- Only tracks vehicles inside IN or OUT zones
- Uses OpenCV `pointPolygonTest`
- Ignores vehicles outside defined zones

#### 5. DeepSORT Tracking
- Assigns unique track IDs to each vehicle
- Maintains IDs across frames
- Handles occlusions and appearance changes
- Parameters optimized for stationary vehicles

#### 6. Position Tracking
- **Active tracks**: Stores current position of all tracked vehicles
- **Lost tracks**: Records last position of disappeared vehicles
- Used for spatial duplicate detection
- Cleaned up after 150 frames

#### 7. Spatial Duplicate Detection
**Purpose**: Prevents re-counting when DeepSORT assigns new ID to same car

**How it works**:
1. When new track appears, check against recently lost tracks
2. If new track within 100px of lost counted track (within last 150 frames)
3. → Likely same vehicle, mark as duplicate
4. Inherit counted status from original ID
5. Don't count again

**Example**:
```
Frame 100: Car ID 5 enters IN zone → Counted (IN count = 1)
Frame 150: Car stops, ID lost due to poor detection
Frame 200: Car detected again, assigned new ID 12
           Position check: ID 12 is 50px from last position of ID 5
           → Detected as duplicate
           → ID 12 marked as already counted
           → Not counted again
```

#### 8. Zone Entry Detection
- Tracks current and previous zone status for each vehicle
- **IN counting**: Triggers when vehicle enters IN zone (outside → inside transition)
- **OUT counting**: Triggers when vehicle enters OUT zone (outside → inside transition)
- Each vehicle counted only once per zone

#### 9. Mutual Exclusion
**Critical feature**: Once counted in ANY zone, never counted in other zone

```python
counted_any_ids = set()  # Global tracking

if track_id in counted_any_ids:
    # Already counted in IN or OUT - skip
    return None
```

**Why this matters**:
- Prevents vehicle counted as both IN and OUT
- Handles cases where vehicle passes through both zones
- Ensures accurate net difference (IN - OUT)

#### 10. Visualization & Output
- Draws filled polygons for zones (20% transparency)
- Shows bounding boxes with unique colors per track ID
- Displays track IDs and status (`[IN]` or `[OUT]`)
- Updates count display in real-time
- Saves annotated video

---

## Advanced Features

### Multi-Layer Duplicate Prevention

This system uses **4 layers** of duplicate prevention:

#### Layer 1: YOLO NMS
- Removes overlapping detections in same frame
- IoU threshold: 0.3 (aggressive)

#### Layer 2: Custom Overlap Filtering
- Additional filtering after YOLO
- Catches cases YOLO misses
- Keeps highest confidence detection

#### Layer 3: Track ID-Based
- Each track ID counted only once
- Set-based tracking: `counted_any_ids`

#### Layer 4: Spatial-Temporal Filtering
- Detects ID reassignments
- Checks position + time window
- Prevents re-counting stopped vehicles

### Stationary Vehicle Handling

**Challenge**: When vehicles stop (e.g., at barrier), YOLO may lose detection, causing DeepSORT to drop the track and assign new ID when vehicle moves again.

**Solution**:
1. **Increased `max_age`** (50 frames): Keeps tracks alive longer during brief occlusions
2. **Lost track recording**: Saves last known position
3. **Spatial duplicate detection**: Matches new tracks to lost positions
4. **Automatic inheritance**: New ID inherits counted status

**Example Scenario**:
```
1. Car approaches entrance barrier
2. ID 5 assigned, enters IN zone → Counted
3. Car stops at barrier for ticket
4. YOLO detection fails (occlusion/angle)
5. DeepSORT drops ID 5 after 50 frames
6. Car gets ticket, moves forward
7. YOLO detects again, assigns new ID 12
8. System checks: ID 12 is 30px from last position of ID 5
9. → Duplicate detected, ID 12 marked as counted
10. → Not counted again ✅
```

### Vehicle Type Filtering

```python
def is_car(self, class_id):
    return class_id in [2, 7]  # Cars and trucks only
```

- **Class 2**: Passenger cars, sedans, SUVs
- **Class 7**: Pickup trucks, vans
- **Excluded**: Motorcycles (3), buses (5), pedestrians, bicycles

---

## Troubleshooting

### Problem: Low count / Missing vehicles

**Possible Causes**:
1. Confidence threshold too high
2. Detection zones don't cover entry/exit paths
3. Poor video quality
4. Small vehicles in frame

**Solutions**:
1. **Lower confidence**:
   ```python
   conf=0.5  # Reduce from 0.8
   MIN_CONFIDENCE = 0.4  # Reduce from 0.6
   ```

2. **Increase image size**:
   ```python
   imgsz=1280  # Increase from 640
   ```

3. **Check zone coordinates**:
   - Use frame extraction tool to verify zones
   - Ensure zones cover vehicle paths

4. **Verify video quality**:
   - Check if vehicles are clearly visible
   - Improve lighting if possible

---

### Problem: Same vehicle counted multiple times

**Possible Causes**:
1. DeepSORT losing track IDs
2. Spatial duplicate detection not catching re-IDs
3. Vehicles moving between zones

**Solutions**:
1. **Increase tracking persistence**:
   ```python
   max_age=100  # Increase from 50
   ```

2. **Increase spatial threshold**:
   ```python
   distance_threshold=150  # Increase from 100
   ```

3. **Check zone overlap**:
   - Ensure IN and OUT zones don't overlap
   - Vehicles shouldn't pass through both zones

---

### Problem: Too many false detections

**Possible Causes**:
1. Low confidence threshold
2. Detecting non-vehicles (shadows, reflections)
3. Zone includes unwanted areas

**Solutions**:
1. **Increase confidence**:
   ```python
   conf=0.9  # Increase from 0.8
   MIN_CONFIDENCE = 0.8  # Increase from 0.6
   ```

2. **Refine detection zones**:
   - Make zones more specific
   - Exclude problematic areas

---

### Problem: Slow processing

**Possible Causes**:
1. High-resolution video
2. Large YOLO model
3. CPU-only processing
4. Display window updates

**Solutions**:
1. **Use GPU**:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Reduce image size**:
   ```python
   imgsz=416  # Reduce from 640
   ```

3. **Disable display**:
   ```python
   counter.process_video(display=False)
   ```

4. **Use smaller model**:
   ```python
   MODEL_PATH = "yolo11n.pt"  # Nano model (faster but less accurate)
   ```

---

### Problem: Display window freezes

**Possible Causes**:
1. High CPU usage
2. Large frame size
3. Too many tracked objects

**Solutions**:
1. **Disable display**:
   ```python
   counter.process_video(display=False)
   ```

2. **Process video without viewing**:
   - Check output video after processing completes

3. **Close other applications**:
   - Free up system resources

---

### Problem: Vehicles counted in wrong zone

**Possible Causes**:
1. Zone coordinates incorrect
2. Zones positioned incorrectly
3. Vehicle path crosses both zones

**Solutions**:
1. **Verify zone coordinates**:
   - Extract a frame and check polygon positions
   - Update coordinates to match actual entry/exit paths

2. **Adjust zone size/position**:
   - Make zones more specific to entry/exit areas
   - Ensure clear separation between IN and OUT

3. **Check video angle**:
   - Camera should have clear view of distinct entry/exit paths

---

## Customizing Detection Zones

### Step-by-Step Guide

1. **Extract a frame** from your video:
   ```bash
   cd ..
   python save_frame.py  # Update with your video path
   ```

2. **Open frame** in image editor (e.g., Paint, GIMP, Photoshop)

3. **Enable pixel coordinates** display (hover to see x, y values)

4. **Mark zone corners**:
   - Click on each corner of your desired zone
   - Record (x, y) coordinates

5. **Update code** with new coordinates:
   ```python
   IN_REGION = [
       (x1, y1),  # Bottom-left
       (x2, y2),  # Top-left
       (x3, y3),  # Top-right
       (x4, y4)   # Bottom-right
   ]
   ```

6. **Test and adjust**:
   - Run with display enabled
   - Verify zones cover correct areas
   - Refine as needed

### Zone Design Tips

- **IN zone**: Should cover area where vehicles first appear when entering
- **OUT zone**: Should cover area where vehicles appear when exiting
- **No overlap**: Keep zones separate to avoid confusion
- **Full coverage**: Ensure entire vehicle passes through zone
- **Perspective**: Account for camera angle and vehicle size changes

---

## Summary

The Mall Car Entrance and Exit Counter provides:

✅ **Simultaneous bidirectional counting** with single camera
✅ **Smart duplicate prevention** with 4-layer system
✅ **Stationary vehicle handling** for barrier/ticket scenarios
✅ **Spatial-temporal filtering** to catch DeepSORT ID reassignments
✅ **Mutual exclusion** - vehicles never counted in both zones
✅ **Real-time visualization** with color-coded zones
✅ **Detailed statistics** including net difference (IN - OUT)
✅ **Configurable parameters** for different scenarios

### Quick Reference

**Default Detection Zones**:
- IN zone (green): Entrance area
- OUT zone (blue): Exit area

**Key Files**:
- Input: `video/20251223172447350.MP4`
- Output: `output_in_out_count.mp4`
- Script: `car_in_out_counter.py`

**Keyboard Controls**:
- `q`: Quit processing
- `p`: Pause/resume

For questions, issues, or customization help, refer to the code comments or contact the development team.
