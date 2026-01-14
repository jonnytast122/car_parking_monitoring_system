import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
import torch


class CarExitCounter:
    def __init__(self, video_path, model_path, detection_region=None, counting_line_position=0.7, enable_counting=True, show_tracking=True, temporal_threshold_minutes=0.3, spatial_threshold_pixels=150, process_interval=5):
        """
        Initialize the car exit counter

        Args:
            video_path: Path to the video file
            model_path: Path to the YOLO model
            detection_region: List of points defining the detection polygon [(x,y), ...]
            counting_line_position: Position of counting line (0-1, relative to frame height)
            enable_counting: Enable vehicle counting (default: True)
            show_tracking: Show tracking boxes and IDs (default: True)
            temporal_threshold_minutes: Time window in minutes for duplicate detection (default: 0.3 = 18 seconds)
            spatial_threshold_pixels: Distance threshold in pixels for duplicate detection (default: 150)
            process_interval: Process detection/counting every N frames (default: 5, set to 1 for every frame)
        """
        self.video_path = video_path
        self.model = YOLO(model_path)

        # Check GPU availability and set device
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("=" * 60)
            print(f"GPU DETECTED: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Running on GPU")
            print("=" * 60)
        else:
            self.device = 'cpu'
            print("=" * 60)
            print("WARNING: GPU NOT DETECTED - Running on CPU")
            print("Possible reasons:")

            # Check PyTorch CUDA support
            if not torch.cuda.is_available():
                if torch.version.cuda is None:
                    print("  1. PyTorch is not compiled with CUDA support")
                    print("     Solution: Install PyTorch with CUDA support:")
                    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
                else:
                    print("  1. CUDA is available in PyTorch but no GPU detected")
                    print("     Possible causes:")
                    print("     - No NVIDIA GPU hardware installed")
                    print("     - GPU drivers not installed or outdated")
                    print("     - CUDA toolkit version mismatch")
                    print(f"     - PyTorch CUDA version: {torch.version.cuda}")
                    print("     Solution: Install/update NVIDIA GPU drivers from:")
                    print("     https://www.nvidia.com/Download/index.aspx")

            print("=" * 60)

        # Move model to the selected device
        self.model.to(self.device)

        # Optimized DeepSORT parameters for better tracking consistency
        # VERY AGGRESSIVE settings to handle cars moving closer to camera (extreme scale/appearance changes)
        # max_age: Number of frames to keep track alive without detection (increased for better persistence)
        # n_init: Consecutive detections needed before track is confirmed (prevents false positives)
        # max_iou_distance: Maximum distance for matching detections to tracks (HIGHER = more lenient with size changes)
        # max_cosine_distance: Cosine distance threshold for appearance features (HIGHER = more lenient with appearance changes)
        self.tracker = DeepSort(
            max_age=20,              # MAXIMUM persistence - keep tracks alive very long
            n_init=2,                 # Quick confirmation - reduces delay in ID assignment
            max_iou_distance=0.7,    # Very high leniency for size changes (small→large)
            max_cosine_distance=0.9,  # VERY high appearance tolerance - handles dramatic changes
            nn_budget=500             # Large feature budget for better matching
        )

        self.enable_counting = enable_counting
        self.show_tracking = show_tracking
        self.process_interval = process_interval  # Process every N frames

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Detection region setup
        self.detection_region = detection_region
        if self.detection_region is not None:
            self.detection_polygon = np.array(self.detection_region, dtype=np.int32)
            # Calculate center of detection region
            self.region_center_x = int(np.mean([p[0] for p in self.detection_region]))
            self.region_center_y = int(np.mean([p[1] for p in self.detection_region]))
        else:
            self.detection_polygon = None
            self.region_center_x = None
            self.region_center_y = None

        # Counting line setup
        self.counting_line_y = int(self.height * counting_line_position)

        # Tracking data
        self.counted_ids = set()
        self.exit_count = 0
        self.track_positions = {}  # {track_id: [previous_y_positions]}
        self.inside_region = {}  # {track_id: is_inside}
        self.track_last_counted_time = {}  # {track_id: timestamp} - prevent re-counting within cooldown period

        # Spatial-temporal filtering to prevent duplicate counts
        # ADJUSTABLE thresholds to handle cars moving closer to camera (extreme screen movement)
        self.counted_positions = []  # [(x, y, timestamp), ...]
        self.spatial_threshold = spatial_threshold_pixels  # pixels - don't count if within this distance
        self.temporal_threshold = temporal_threshold_minutes * 60.0  # convert minutes to seconds

        # Colors
        self.colors = {}

    def get_color(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.colors:
            # Convert track_id to integer for seeding
            seed = hash(str(track_id)) % (2**32)
            np.random.seed(seed)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def is_car(self, class_id):
        """Check if detected object is a car or truck (COCO dataset)"""
        # COCO classes: 2=car, 7=truck
        return class_id in [2, 7]

    def calculate_iou(self, box1, box2):
        """
        Calculate IoU (Intersection over Union) between two bounding boxes

        Args:
            box1, box2: [x1, y1, x2, y2]

        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def filter_overlapping_detections(self, detections, iou_threshold=0.5):
        """
        Remove overlapping detections (same car detected multiple times)
        Keeps the detection with highest confidence

        Args:
            detections: List of (bbox, confidence, class_id)
            iou_threshold: IoU threshold for considering detections as duplicates

        Returns:
            Filtered list of detections
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)

        filtered = []

        for det in detections:
            bbox, conf, cls = det
            # Convert [left, top, width, height] to [x1, y1, x2, y2]
            x1, y1, w, h = bbox
            box1 = [x1, y1, x1 + w, y1 + h]

            # Check if this detection overlaps significantly with any already accepted detection
            is_duplicate = False
            for filtered_det in filtered:
                filtered_bbox, _, _ = filtered_det
                x1_f, y1_f, w_f, h_f = filtered_bbox
                box2 = [x1_f, y1_f, x1_f + w_f, y1_f + h_f]

                iou = self.calculate_iou(box1, box2)
                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(det)

        return filtered

    def is_point_in_region(self, x, y):
        """
        Check if a point is inside the detection region

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            True if point is inside the region, False otherwise
        """
        if self.detection_polygon is None:
            return True  # No region defined, accept all points

        # Use OpenCV's pointPolygonTest
        result = cv2.pointPolygonTest(self.detection_polygon, (float(x), float(y)), False)
        return result >= 0

    def is_duplicate_count(self, x, y, current_time):
        """
        Check if this position is too close to a recently counted position
        This prevents counting the same stopped vehicle twice with different IDs

        Args:
            x: X coordinate
            y: Y coordinate
            current_time: Current timestamp

        Returns:
            True if this is likely a duplicate count, False otherwise
        """
        # Remove old positions outside the temporal window
        self.counted_positions = [
            (px, py, t) for px, py, t in self.counted_positions
            if current_time - t <= self.temporal_threshold
        ]

        # Check if current position is too close to any recent count
        for px, py, t in self.counted_positions:
            distance = np.sqrt((x - px)**2 + (y - py)**2)
            if distance <= self.spatial_threshold:
                return True  # Too close to a recent count - likely duplicate

        return False

    def count_vehicle(self, track_id, center_x, center_y, current_time):
        """
        Count vehicle when it passes through the detection region with direction validation
        Only adds to count, never subtracts (no reverse detection)

        Args:
            track_id: Unique ID of the tracked vehicle
            center_x: Current X position of vehicle center
            center_y: Current Y position of vehicle center
            current_time: Current timestamp for spatial-temporal filtering
        """
        # Initialize position history for this track if needed
        if track_id not in self.track_positions:
            self.track_positions[track_id] = []

        # Store current Y position (keep last 10 positions for direction calculation)
        self.track_positions[track_id].append(center_y)
        if len(self.track_positions[track_id]) > 10:
            self.track_positions[track_id].pop(0)

        # Check if vehicle is inside the detection region
        is_inside = self.is_point_in_region(center_x, center_y)

        # Track previous inside/outside state
        was_inside = self.inside_region.get(track_id, False)
        self.inside_region[track_id] = is_inside

        # Count when vehicle enters the region (wasn't inside before, but is now)
        if self.enable_counting and is_inside and not was_inside:
            # Check temporal cooldown: has enough time passed since last count?
            COOLDOWN_SECONDS = 25.0  # Don't re-count same ID within 20 seconds
            last_counted_time = self.track_last_counted_time.get(track_id, -float('inf'))
            time_since_last_count = current_time - last_counted_time

            if time_since_last_count < COOLDOWN_SECONDS:
                # Too soon - likely a reverse/re-exit, skip counting
                return False

            # Verify direction: car should be moving UPWARD (exiting) not DOWNWARD (entering)
            # Need at least 3 position samples to determine direction
            if len(self.track_positions[track_id]) >= 3:
                # Calculate direction from position history
                # Negative movement = moving up (exiting), Positive = moving down (entering)
                recent_positions = self.track_positions[track_id][-5:]  # Last 5 positions

                # Calculate average movement direction
                movements = []
                for i in range(1, len(recent_positions)):
                    movements.append(recent_positions[i] - recent_positions[i-1])

                avg_movement = sum(movements) / len(movements) if movements else 0

                # EXIT DIRECTION: Y should be DECREASING (moving up in frame = exiting)
                # If average movement is positive or too small, car is entering/stationary
                MIN_MOVEMENT_THRESHOLD = -2.0  # pixels per frame (negative = upward)

                if avg_movement > MIN_MOVEMENT_THRESHOLD:
                    # Car is entering or not moving properly - don't count as exit
                    return False

            # All checks passed - count the vehicle exit
            self.counted_ids.add(track_id)
            self.track_last_counted_time[track_id] = current_time
            self.exit_count += 1
            return True

        return False

    def draw_annotations(self, frame, tracks):
        """Draw bounding boxes, IDs, and detection region"""
        # Draw detection region polygon
        if self.detection_polygon is not None:
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.detection_polygon], (255, 0, 0))  # Blue for exit
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Draw polygon border
            cv2.polylines(frame, [self.detection_polygon], True, (255, 0, 0), 3)

            # Add label
            cv2.putText(frame, "EXIT ZONE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Draw tracked vehicles (only if show_tracking is enabled)
        if self.show_tracking:
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Get color for this track
                color = self.get_color(track_id)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw center point
                cv2.circle(frame, (center_x, center_y), 5, color, -1)

                # Draw track ID
                label = f"ID: {track_id}"
                if self.enable_counting and track_id in self.counted_ids:
                    label += " [COUNTED]"

                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Draw trajectory
                if track_id in self.track_positions and len(self.track_positions[track_id]) > 1:
                    for i in range(1, len(self.track_positions[track_id])):
                        pt1 = (center_x, self.track_positions[track_id][i-1])
                        pt2 = (center_x, self.track_positions[track_id][i])
                        cv2.line(frame, pt1, pt2, color, 2)

        return frame

    def draw_statistics(self, frame, fps=0):
        """Draw count statistics on frame"""
        # Background for statistics
        if self.enable_counting:
            cv2.rectangle(frame, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 110), (255, 255, 255), 2)

            # Display exit count
            cv2.putText(frame, f"EXITS COUNTED: {self.exit_count}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display FPS
            cv2.putText(frame, f"FPS: {fps:.2f}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        return frame

    def process_video(self, output_path="BDC-exit-Cuted.mp4", display=True):
        """
        Process video and count vehicles

        Args:
            output_path: Path to save output video
            display: Whether to display video while processing
        """
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        frame_count = 0
        start_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_actual = 0
        last_tracks = []  # Keep last known tracks for smooth display

        print(f"Processing video: {self.video_path}")
        print(f"Frame size: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        if self.detection_region:
            print(f"Detection region: {len(self.detection_region)} points defined")
        print(f"Process interval: Every {self.process_interval} frame(s)")
        print("-" * 50)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            fps_frame_count += 1

            # Calculate FPS every second
            fps_elapsed = time.time() - fps_start_time
            if fps_elapsed >= 1.0:
                fps_actual = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Process detection/counting only every N frames (skip frames for performance)
            if frame_count % self.process_interval != 0:
                # Skip detection but still draw last known tracks and statistics for smooth display
                frame = self.draw_annotations(frame, last_tracks)
                frame = self.draw_statistics(frame, fps_actual)

                # Write frame with annotations
                out.write(frame)

                # Display
                if display:
                    cv2.imshow('Car Exit Counter', frame)

                # Check for key press
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Stopped by user")
                        break
                    elif key == ord('p'):
                        print("Paused. Press any key to continue...")
                        cv2.waitKey(0)

                continue

            # Run YOLO detection with lower confidence to detect more cars
            # conf=0.25: Lower confidence to catch more cars that might be missed
            # iou=0.5: Moderate IoU for NMS - balanced suppression
            # imgsz=1280: Larger image size for better detection of small/distant objects
            # agnostic_nms=True: Apply NMS across all classes
            results = self.model(frame, conf=0.35, iou=0.5, imgsz=640, agnostic_nms=True, verbose=False, device=self.device)[0]

            # Prepare detections for DeepSORT with confidence filtering
            detections = []
            MIN_CONFIDENCE = 0.35  # Lower threshold to detect more cars that might be missed

            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only track cars with sufficient confidence
                if self.is_car(class_id) and confidence >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Calculate center point of bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Only add detection if center is inside the detection region
                    # This ensures vehicles outside are not tracked, displayed, or counted
                    if self.is_point_in_region(center_x, center_y):
                        # DeepSORT expects [left, top, width, height]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append((bbox, confidence, class_id))

            # Apply custom overlap filtering to remove duplicate detections
            # This catches cases where YOLO detects the same car twice (e.g., at different scales)
            # Lower threshold = more aggressive filtering (0.3 = very aggressive, 0.5 = moderate)
            detections = self.filter_overlapping_detections(detections, iou_threshold=0.5)

            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Save tracks for drawing on skipped frames (prevents flashing)
            last_tracks = tracks

            # Count vehicles
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Calculate center point (consistent with detection filtering)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check and count (with direction validation and temporal filtering)
                current_time = time.time() - start_time
                if self.count_vehicle(track_id, center_x, center_y, current_time):
                    print(f"[+COUNT] Frame {frame_count}: Car ID {track_id} EXITED (moving upward)! Total: {self.exit_count}")
                    print(f"         Counted IDs: {sorted(self.counted_ids)}")

            # Draw annotations
            frame = self.draw_annotations(frame, tracks)
            frame = self.draw_statistics(frame, fps_actual)

            # Write frame
            out.write(frame)

            # Display
            if display:
                # Resize for display if too large
                display_frame = frame.copy()
                if self.width > 1280:
                    scale = 1280 / self.width
                    new_width = 1280
                    new_height = int(self.height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))

                cv2.imshow('Car Exit Counter', display_frame)

                # Press 'q' to quit, 'p' to pause
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopped by user")
                    break
                elif key == ord('p'):
                    cv2.waitKey(0)

            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count} frames | FPS: {fps_actual:.2f} | Count: {self.exit_count}")

        # Cleanup
        self.cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.2f}")
        print(f"Total vehicles exited: {self.exit_count}")
        print(f"Output saved to: {output_path}")
        print("=" * 50)


def main():
    # Configuration
    VIDEO_PATH = "BDC-exit-Cuted.mp4"
    MODEL_PATH = "yolo12m.pt"
    OUTPUT_PATH = "output_exit_count.mp4"
    COUNTING_LINE_POSITION = 0.6  # 50% of frame height

    # Detection region polygon for exit (detection area - NOT counting line)
    DETECTION_REGION = [
        (0, 575),
        (0, 282),
        (765, 282),
        (765, 575)
    ]

    # Duplicate prevention settings (ADJUSTABLE)
    TEMPORAL_THRESHOLD_MINUTES = 2.0   # Time window: 1.0 min = 60 seconds (catch ID switches)
    SPATIAL_THRESHOLD_PIXELS = 300     # Distance threshold: 300 pixels (large area to catch same car)

    # Performance settings
    PROCESS_INTERVAL = 5  # Process detection/counting every frame for better detection (1 = every frame, 5 = every 5th frame)

    # Create counter with tracking visualization
    counter = CarExitCounter(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        detection_region=DETECTION_REGION,
        counting_line_position=COUNTING_LINE_POSITION,
        enable_counting=True,                              # Enable counting
        show_tracking=True,                                # Show tracking boxes and IDs
        temporal_threshold_minutes=TEMPORAL_THRESHOLD_MINUTES,  # Adjustable time window
        spatial_threshold_pixels=SPATIAL_THRESHOLD_PIXELS,      # Adjustable spatial distance
        process_interval=PROCESS_INTERVAL                       # Process every N frames for performance
    )

    # Process video
    counter.process_video(output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()
