import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


class CarEntranceCounter:
    def __init__(self, video_path, model_path, detection_region=None, counting_line_position=0.5, enable_counting=True, show_tracking=True, temporal_threshold_minutes=0.33, spatial_threshold_pixels=150):
        """
        Initialize the car entrance counter

        Args:
            video_path: Path to the video file
            model_path: Path to the YOLO model
            detection_region: List of points defining the detection polygon [(x,y), ...]
            counting_line_position: Position of counting line (0-1, relative to frame height)
            enable_counting: Enable vehicle counting (default: True)
            show_tracking: Show tracking boxes and IDs (default: True)
            temporal_threshold_minutes: Time window in minutes for duplicate detection (default: 0.33 = 20 seconds)
            spatial_threshold_pixels: Distance threshold in pixels for duplicate detection (default: 150)
        """
        self.video_path = video_path
        self.model = YOLO(model_path)

        # Optimized DeepSORT parameters for better tracking consistency
        # VERY AGGRESSIVE settings to handle cars moving closer to camera (extreme scale/appearance changes)
        # max_age: Number of frames to keep track alive without detection (increased for better persistence)
        # n_init: Consecutive detections needed before track is confirmed (prevents false positives)
        # max_iou_distance: Maximum distance for matching detections to tracks (HIGHER = more lenient with size changes)
        # max_cosine_distance: Cosine distance threshold for appearance features (HIGHER = more lenient with appearance changes)
        self.tracker = DeepSort(
            max_age=60,               # Keep tracks alive even longer - prevents losing track during brief occlusions
            n_init=2,                 # Confirm tracks faster (2-3 frames) - prevents delayed ID assignment
            max_iou_distance=0.95,    # Almost ignore IOU - very lenient with size changes (small→large)
            max_cosine_distance=0.9,  # VERY lenient appearance matching - critical for dramatic perspective changes
            nn_budget=300             # Store more appearance features for better matching
        )

        self.enable_counting = enable_counting
        self.show_tracking = show_tracking

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
        self.entrance_count = 0
        self.track_positions = {}  # {track_id: [previous_y_positions]}
        self.inside_region = {}  # {track_id: is_inside}

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
            # Count every car that passes through the region (no duplicate filtering)
            self.counted_ids.add(track_id)
            self.entrance_count += 1
            return True

        return False

    def draw_annotations(self, frame, tracks):
        """Draw bounding boxes, IDs, and detection region"""
        # Draw detection region polygon
        if self.detection_polygon is not None:
            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.detection_polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)

            # Draw polygon border
            cv2.polylines(frame, [self.detection_polygon], True, (0, 255, 0), 3)

            # Add label
            cv2.putText(frame, "DETECTION ZONE", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

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

    def draw_statistics(self, frame):
        """Draw count statistics on frame"""
        # Background for statistics
        if self.enable_counting:
            cv2.rectangle(frame, (10, 10), (350, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 70), (255, 255, 255), 2)

            # Display entrance count
            cv2.putText(frame, f"VEHICLES PASSED: {self.entrance_count}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame

    def process_video(self, output_path="output_entrance_count.mp4", display=True):
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

        print(f"Processing video: {self.video_path}")
        print(f"Frame size: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        if self.detection_region:
            print(f"Detection region: {len(self.detection_region)} points defined")
        print("-" * 50)

        # Initialize window with WINDOW_NORMAL for better responsiveness
        if display:
            cv2.namedWindow('Car Entrance Counter', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Car Entrance Counter', 1280, 720)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLO detection with AGGRESSIVE NMS to prevent duplicate detections
            # conf=0.3: Lower confidence to catch small/distant objects
            # iou=0.7: HIGHER IoU for NMS - more aggressive suppression of overlapping boxes
            # imgsz=640: Image size for detection (larger = better for small objects, but slower)
            # agnostic_nms=True: Apply NMS across all classes
            results = self.model(frame, conf=0.3, iou=0.7, imgsz=640, agnostic_nms=True, verbose=False)[0]

            # Prepare detections for DeepSORT with confidence filtering
            detections = []
            MIN_CONFIDENCE = 0.5  # Higher minimum confidence to reduce false detections (was 0.4)

            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only track cars with sufficient confidence
                if self.is_car(class_id) and confidence >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Calculate center to check if in region
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Only add detection if it's inside the detection region
                    # This ensures vehicles outside are not tracked, displayed, or counted
                    if self.is_point_in_region(center_x, center_y):
                        # DeepSORT expects [left, top, width, height]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append((bbox, confidence, class_id))

            # Apply custom overlap filtering to remove duplicate detections
            # This catches cases where YOLO detects the same car twice (e.g., at different scales)
            detections = self.filter_overlapping_detections(detections, iou_threshold=0.6)

            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Count vehicles
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)

                # Calculate center
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Check and count (with current time for spatial-temporal filtering)
                current_time = time.time() - start_time
                if self.count_vehicle(track_id, center_x, center_y, current_time):
                    print(f"[COUNT] Frame {frame_count}: Car ID {track_id} passed through detection zone! Total count: {self.entrance_count}")
                    print(f"        Already counted IDs: {sorted(self.counted_ids)}")

            # Draw annotations
            frame = self.draw_annotations(frame, tracks)
            frame = self.draw_statistics(frame)

            # Write frame
            out.write(frame)

            # Display - only update every 3rd frame to prevent freezing
            if display and frame_count % 3 == 0:
                # Resize for display if too large
                display_frame = frame.copy()
                if self.width > 1280:
                    scale = 1280 / self.width
                    new_width = 1280
                    new_height = int(self.height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))

                cv2.imshow('Car Entrance Counter', display_frame)

            # Check for key press on every frame for responsiveness
            if display:
                # Check for key press to allow quitting/pausing
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopped by user")
                    break
                elif key == ord('p'):
                    print("Paused. Press any key to continue...")
                    cv2.waitKey(0)

            # Print progress every 30 frames
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"Processed {frame_count} frames | FPS: {fps_actual:.2f} | Count: {self.entrance_count}")

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
        print(f"Total vehicles entered: {self.entrance_count}")
        print(f"Output saved to: {output_path}")
        print("=" * 50)


def main():
    # Configuration
    VIDEO_PATH = "BDC-Entrance-Cuted.mp4"
    MODEL_PATH = "yolo12x.pt"
    OUTPUT_PATH = "output_entrance_count.mp4"
    COUNTING_LINE_POSITION = 0.5  # 50% of frame height

    # Detection region polygon (detection area - NOT counting line)
    DETECTION_REGION = [
        (41, 569),
        (46, 208),
        (452, 217),
        (443, 572)
    ]

    # Duplicate prevention settings (ADJUSTABLE)
    TEMPORAL_THRESHOLD_MINUTES = 0.33  # Time window: 0.33 min = 20 seconds (adjust as needed)
    SPATIAL_THRESHOLD_PIXELS = 150     # Distance threshold in pixels (adjust as needed)

    # Create counter with tracking visualization
    counter = CarEntranceCounter(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        detection_region=DETECTION_REGION,
        counting_line_position=COUNTING_LINE_POSITION,
        enable_counting=True,                              # Enable counting
        show_tracking=True,                                # Show tracking boxes and IDs
        temporal_threshold_minutes=TEMPORAL_THRESHOLD_MINUTES,  # Adjustable time window
        spatial_threshold_pixels=SPATIAL_THRESHOLD_PIXELS       # Adjustable spatial distance
    )

    # Process video (set display=False to disable window and prevent freezing)
    counter.process_video(output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()
