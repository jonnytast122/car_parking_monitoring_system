

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


class CarExitCounterStream:
    def __init__(self, stream_url, model_path, detection_region=None, counting_line_position=0.5, enable_counting=True, show_tracking=True, temporal_threshold_minutes=0.3, spatial_threshold_pixels=150):
        """
        Initialize the car exit counter for RTSP stream

        Args:
            stream_url: RTSP stream URL (e.g., rtsp://user:pass@host:port/stream)
            model_path: Path to the YOLO model
            detection_region: List of points defining the detection polygon [(x,y), ...]
            counting_line_position: Position of counting line (0-1, relative to frame height)
            enable_counting: Enable vehicle counting (default: True)
            show_tracking: Show tracking boxes and IDs (default: True)
            temporal_threshold_minutes: Time window in minutes for duplicate detection (default: 0.3 = 18 seconds)
            spatial_threshold_pixels: Distance threshold in pixels for duplicate detection (default: 150)
        """
        self.stream_url = stream_url
        self.model = YOLO(model_path)

        # Optimized DeepSORT parameters for better tracking consistency
        # VERY AGGRESSIVE settings to handle cars moving closer to camera (extreme scale/appearance changes)
        # max_age: Number of frames to keep track alive without detection (increased for better persistence)
        # n_init: Consecutive detections needed before track is confirmed (prevents false positives)
        # max_iou_distance: Maximum distance for matching detections to tracks (HIGHER = more lenient with size changes)
        # max_cosine_distance: Cosine distance threshold for appearance features (HIGHER = more lenient with appearance changes)
        self.tracker = DeepSort(
            max_age=30,              # MAXIMUM persistence - keep tracks alive very long
            n_init=2,                 # Quick confirmation - reduces delay in ID assignment
            max_iou_distance=0.7,    # Very high leniency for size changes (small→large)
            max_cosine_distance=0.9,  # VERY high appearance tolerance - handles dramatic changes
            nn_budget=500             # Large feature budget for better matching
        )

        self.enable_counting = enable_counting
        self.show_tracking = show_tracking

        # Initialize video capture with RTSP options
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 25  # Default FPS for streams

        self.connect_stream()

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

        # Spatial-temporal filtering to prevent duplicate counts
        # ADJUSTABLE thresholds to handle cars moving closer to camera (extreme screen movement)
        self.counted_positions = []  # [(x, y, timestamp), ...]
        self.spatial_threshold = spatial_threshold_pixels  # pixels - don't count if within this distance
        self.temporal_threshold = temporal_threshold_minutes * 60.0  # convert minutes to seconds

        # Colors
        self.colors = {}

    def connect_stream(self):
        """Connect to RTSP stream with retry logic"""
        print(f"Connecting to stream: {self.stream_url}")

        # Release existing capture if any
        if self.cap is not None:
            self.cap.release()

        # Set FFmpeg environment variables to use TCP and reduce errors
        import os
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer"

        # Create video capture with RTSP options
        self.cap = cv2.VideoCapture(self.stream_url, cv2.CAP_FFMPEG)

        # Set buffer size to reduce latency and corrupted frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Set additional properties to handle stream issues
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))

        # Try to read stream properties
        if self.cap.isOpened():
            # Try to get a test frame to verify connection
            ret, frame = self.cap.read()
            if ret:
                self.width = frame.shape[1]
                self.height = frame.shape[0]

                # Try to get FPS, use default if not available
                fps_from_stream = self.cap.get(cv2.CAP_PROP_FPS)
                if fps_from_stream > 0:
                    self.fps = int(fps_from_stream)

                print(f"Stream connected successfully!")
                print(f"Frame size: {self.width}x{self.height}")
                print(f"FPS: {self.fps}")
                return True
            else:
                print("Failed to read frame from stream")
                return False
        else:
            print("Failed to open stream")
            return False

    def reconnect_stream(self):
        """Attempt to reconnect to stream"""
        print("Attempting to reconnect to stream...")
        time.sleep(2)  # Wait before reconnecting
        return self.connect_stream()

    def get_color(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.colors:
            # Convert track_id to integer for seeding
            seed = hash(str(track_id)) % (2**32)
            np.random.seed(seed)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def is_car(self, class_id):
        """Check if detected object is a car (class 2 in COCO dataset)"""
        # COCO classes: 2=car (only cars, no motorcycles, buses, or trucks)
        return class_id == 2

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
        Count vehicle when it passes through the detection region

        Args:
            track_id: Unique ID of the tracked vehicle
            center_x: Current X position of vehicle center
            center_y: Current Y position of vehicle center
            current_time: Current timestamp for spatial-temporal filtering
        """
        # Check if vehicle is inside the detection region
        is_inside = self.is_point_in_region(center_x, center_y)

        # Track previous inside/outside state
        was_inside = self.inside_region.get(track_id, False)
        self.inside_region[track_id] = is_inside

        # Count when vehicle enters the region (wasn't inside before, but is now)
        # This counts any vehicle that passes through the polygon
        if self.enable_counting and track_id not in self.counted_ids and is_inside and not was_inside:
            # Count based on track_id only - rely on DeepSORT to prevent duplicates
            self.counted_ids.add(track_id)
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

    def process_stream(self, save_output=False, output_path="output_exit_stream.mp4", display=True):
        """
        Process RTSP stream and count vehicles

        Args:
            save_output: Whether to save output video (default: False for streaming)
            output_path: Path to save output video (if save_output=True)
            display: Whether to display video while processing
        """
        # Video writer (optional for streaming)
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            print(f"Recording output to: {output_path}")

        frame_count = 0
        start_time = time.time()
        fps_start_time = time.time()
        fps_frame_count = 0
        fps_actual = 0
        consecutive_failures = 0
        max_failures = 30  # Reconnect after 30 consecutive failed reads

        print(f"Processing stream: {self.stream_url}")
        print(f"Frame size: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        if self.detection_region:
            print(f"Detection region: {len(self.detection_region)} points defined")
        print("-" * 50)
        print("Press 'q' to quit, 'p' to pause")
        print("-" * 50)

        # Initialize window with WINDOW_NORMAL for better responsiveness
        if display:
            cv2.namedWindow('Car Exit Counter - LIVE', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Car Exit Counter - LIVE', 1280, 720)

        while True:
            ret, frame = self.cap.read()

            if not ret or frame is None:
                consecutive_failures += 1

                # Don't print every failure to reduce console spam from H.264 errors
                if consecutive_failures % 10 == 0:
                    print(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")

                if consecutive_failures >= max_failures:
                    print("Too many consecutive failures. Attempting to reconnect...")
                    if self.reconnect_stream():
                        consecutive_failures = 0
                        continue
                    else:
                        print("Reconnection failed. Exiting...")
                        break

                time.sleep(0.05)  # Shorter sleep for faster recovery
                continue

            # Check if frame is valid (not corrupted)
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                consecutive_failures += 1
                continue

            # Reset failure counter on successful read
            consecutive_failures = 0
            frame_count += 1
            fps_frame_count += 1

            # Calculate FPS every second
            fps_elapsed = time.time() - fps_start_time
            if fps_elapsed >= 1.0:
                fps_actual = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_start_time = time.time()

            # Wrap processing in try-except to handle corrupted frames
            try:
                # Run YOLO detection with strict confidence to prevent false detections
                # conf=0.8: High confidence to only detect real cars, prevent hallucinations
                # iou=0.5: Moderate IoU for NMS - balanced suppression
                # imgsz=640: Image size for detection (larger = better for small objects, but slower)
                # agnostic_nms=True: Apply NMS across all classes
                results = self.model(frame, conf=0.8, iou=0.5, imgsz=640, agnostic_nms=True, verbose=False)[0]
            except Exception as e:
                # Skip corrupted frames that cause detection errors
                print(f"Error processing frame {frame_count}: {e}")
                continue

            # Prepare detections for DeepSORT with confidence filtering
            detections = []
            MIN_CONFIDENCE = 0.8  # Match detection confidence - prevents hallucinations and false detections

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

                # Check and count (with current time for spatial-temporal filtering)
                current_time = time.time() - start_time
                if self.count_vehicle(track_id, center_x, center_y, current_time):
                    print(f"[COUNT] Frame {frame_count}: Car ID {track_id} passed through exit zone! Total count: {self.exit_count}")
                    print(f"        Already counted IDs: {sorted(self.counted_ids)}")

            # Draw annotations
            frame = self.draw_annotations(frame, tracks)
            frame = self.draw_statistics(frame, fps_actual)

            # Write frame (if recording)
            if save_output and out is not None:
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

                cv2.imshow('Car Exit Counter - LIVE', display_frame)

            # Check for key press
            if display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopped by user")
                    break
                elif key == ord('p'):
                    print("Paused. Press any key to continue...")
                    cv2.waitKey(0)

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames | FPS: {fps_actual:.2f} | Count: {self.exit_count} | Elapsed: {elapsed:.1f}s")

        # Cleanup
        self.cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print("STREAM STOPPED")
        print("=" * 50)
        print(f"Total frames processed: {frame_count}")
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Average FPS: {frame_count/elapsed:.2f}")
        print(f"Total vehicles exited: {self.exit_count}")
        if save_output:
            print(f"Output saved to: {output_path}")
        print("=" * 50)


def main():
    # Configuration
    STREAM_URL = "rtsp://apt:apt123@desktop-bdc:8554/exit"
    MODEL_PATH = "yolo12m.pt"
    OUTPUT_PATH = "output_exit_stream.mp4"
    COUNTING_LINE_POSITION = 0.5  # 50% of frame height

    # Detection region polygon for exit (detection area - NOT counting line)
    DETECTION_REGION = [
        (39, 574),
        (38, 282),
        (723, 281),
        (725, 575)
    ]

    # Duplicate prevention settings (ADJUSTABLE)
    TEMPORAL_THRESHOLD_MINUTES = 1.0   # Time window: 1.0 min = 60 seconds (catch ID switches)
    SPATIAL_THRESHOLD_PIXELS = 300     # Distance threshold: 300 pixels (large area to catch same car)

    # Create counter with tracking visualization
    counter = CarExitCounterStream(
        stream_url=STREAM_URL,
        model_path=MODEL_PATH,
        detection_region=DETECTION_REGION,
        counting_line_position=COUNTING_LINE_POSITION,
        enable_counting=True,                              # Enable counting
        show_tracking=True,                                # Show tracking boxes and IDs
        temporal_threshold_minutes=TEMPORAL_THRESHOLD_MINUTES,  # Adjustable time window
        spatial_threshold_pixels=SPATIAL_THRESHOLD_PIXELS       # Adjustable spatial distance
    )

    # Process stream (set save_output=True to record the stream)
    counter.process_stream(save_output=True, output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()
