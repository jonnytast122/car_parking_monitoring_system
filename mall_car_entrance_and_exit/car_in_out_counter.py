import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time


class CarInOutCounter:
    def __init__(self, video_path, model_path, in_region=None, out_region=None,
                 enable_counting=True, show_tracking=True):
        """
        Initialize the car in/out counter with two separate detection zones

        Args:
            video_path: Path to the video file
            model_path: Path to the YOLO model
            in_region: List of points defining the IN detection polygon [(x,y), ...]
            out_region: List of points defining the OUT detection polygon [(x,y), ...]
            enable_counting: Enable vehicle counting (default: True)
            show_tracking: Show tracking boxes and IDs (default: True)
        """
        self.video_path = video_path
        self.model = YOLO(model_path)

        # Optimized DeepSORT parameters to prevent ID switches and duplicate counting
        # Especially important for stationary vehicles that may stop and start again
        self.tracker = DeepSort(
            max_age=50,             # Keep tracks alive much longer (150 frames ~5 seconds at 30fps) for stopped cars
            n_init=3,                # Require 3 consecutive detections to confirm track
            max_iou_distance=0.8,    # Higher leniency for bounding box changes (cars stopping/starting)
            max_cosine_distance=0.5, # More tolerant appearance matching (lighting/angle changes when stopped)
            nn_budget=100            # Standard feature budget
        )

        self.enable_counting = enable_counting
        self.show_tracking = show_tracking

        # Initialize video capture
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # IN region setup
        self.in_region = in_region
        if self.in_region is not None:
            self.in_polygon = np.array(self.in_region, dtype=np.int32)
        else:
            self.in_polygon = None

        # OUT region setup
        self.out_region = out_region
        if self.out_region is not None:
            self.out_polygon = np.array(self.out_region, dtype=np.int32)
        else:
            self.out_polygon = None

        # Tracking data
        self.counted_in_ids = set()
        self.counted_out_ids = set()
        self.counted_any_ids = set()  # Track ALL counted vehicles to prevent double counting
        self.in_count = 0
        self.out_count = 0
        self.inside_in_region = {}   # {track_id: is_inside_in_region}
        self.inside_out_region = {}  # {track_id: is_inside_out_region}

        # Spatial duplicate detection - track vehicle positions to detect re-IDs
        self.track_positions = {}     # {track_id: (center_x, center_y, frame_number)}
        self.lost_tracks = {}         # {track_id: (center_x, center_y, frame_number)} - recently lost tracks

        # Colors
        self.colors = {}

    def get_color(self, track_id):
        """Get consistent color for each track ID"""
        if track_id not in self.colors:
            seed = hash(str(track_id)) % (2**32)
            np.random.seed(seed)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def is_car(self, class_id):
        """Check if detected object is a car or truck (classes 2, 7 in COCO dataset)"""
        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck (includes pickup trucks)
        return class_id in [2, 7]

    def calculate_iou(self, box1, box2):
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
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
        """
        if len(detections) <= 1:
            return detections

        # Sort by confidence (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        filtered = []

        for det in detections:
            bbox, conf, cls = det
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

    def is_point_in_polygon(self, x, y, polygon):
        """Check if a point is inside a polygon"""
        if polygon is None:
            return False

        result = cv2.pointPolygonTest(polygon, (float(x), float(y)), False)
        return result >= 0

    def check_spatial_duplicate(self, track_id, center_x, center_y, current_frame, distance_threshold=100):
        """
        Check if a new track is spatially close to a recently lost counted track
        This prevents re-counting when DeepSORT assigns a new ID to a stationary car

        Args:
            track_id: Current track ID
            center_x, center_y: Current position
            current_frame: Current frame number
            distance_threshold: Max distance (pixels) to consider as same car

        Returns:
            original_track_id if this is likely a duplicate, None otherwise
        """
        # Check against recently lost tracks (within last 150 frames)
        for lost_id, (lost_x, lost_y, lost_frame) in list(self.lost_tracks.items()):
            # Only check tracks lost recently
            if current_frame - lost_frame > 150:
                # Clean up old lost tracks
                del self.lost_tracks[lost_id]
                continue

            # Calculate distance
            distance = np.sqrt((center_x - lost_x)**2 + (center_y - lost_y)**2)

            # If new track appears very close to a recently lost counted track
            if distance < distance_threshold and lost_id in self.counted_any_ids:
                return lost_id

        return None

    def count_vehicle(self, track_id, center_x, center_y, current_frame):
        """
        Count vehicle when it enters IN or OUT regions
        Once counted in any region, the vehicle will not be counted again in the other region
        Uses spatial duplicate detection to prevent re-counting when DeepSORT creates new IDs

        Args:
            track_id: Unique ID of the tracked vehicle
            center_x: Current X position of vehicle center
            center_y: Current Y position of vehicle center
            current_frame: Current frame number for spatial duplicate detection
        """
        # Check if this is a spatial duplicate (new ID for already counted car)
        original_id = self.check_spatial_duplicate(track_id, center_x, center_y, current_frame)
        if original_id is not None:
            # This is a re-ID of an already counted vehicle - mark it as counted
            if track_id not in self.counted_any_ids:
                print(f"[DUPLICATE DETECTED] Frame {current_frame}: Track ID {track_id} is the same car as already counted ID {original_id} - preventing re-count")
                self.counted_any_ids.add(track_id)
                # Inherit the counted status from original ID
                if original_id in self.counted_in_ids:
                    self.counted_in_ids.add(track_id)
                if original_id in self.counted_out_ids:
                    self.counted_out_ids.add(track_id)
            return None

        counted_new = False

        # Check IN region
        is_in_in_region = self.is_point_in_polygon(center_x, center_y, self.in_polygon)
        was_in_in_region = self.inside_in_region.get(track_id, False)
        self.inside_in_region[track_id] = is_in_in_region

        # Count when vehicle enters IN region (DUPLICATE PREVENTION)
        # Only counts if: 1) ID never counted in ANY region, 2) Just entered (wasn't in before)
        if (self.enable_counting and track_id not in self.counted_any_ids
            and is_in_in_region and not was_in_in_region):
            self.counted_in_ids.add(track_id)  # Mark this ID as counted in IN region
            self.counted_any_ids.add(track_id)  # Mark as counted globally
            self.in_count += 1
            counted_new = True
            return 'IN'

        # Check OUT region
        is_in_out_region = self.is_point_in_polygon(center_x, center_y, self.out_polygon)
        was_in_out_region = self.inside_out_region.get(track_id, False)
        self.inside_out_region[track_id] = is_in_out_region

        # Count when vehicle enters OUT region (DUPLICATE PREVENTION)
        # Only counts if: 1) ID never counted in ANY region, 2) Just entered (wasn't in before)
        if (self.enable_counting and track_id not in self.counted_any_ids
            and is_in_out_region and not was_in_out_region):
            self.counted_out_ids.add(track_id)  # Mark this ID as counted in OUT region
            self.counted_any_ids.add(track_id)  # Mark as counted globally
            self.out_count += 1
            return 'OUT'

        return None

    def draw_annotations(self, frame, tracks):
        """Draw bounding boxes, IDs, and detection regions"""
        # Draw IN region polygon (green)
        if self.in_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.in_polygon], (0, 255, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [self.in_polygon], True, (0, 255, 0), 3)

            # Add label at top of IN region
            in_center_x = int(np.mean([p[0] for p in self.in_region]))
            in_center_y = int(np.min([p[1] for p in self.in_region])) - 10
            cv2.putText(frame, "IN ZONE", (in_center_x - 50, in_center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        # Draw OUT region polygon (blue)
        if self.out_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.out_polygon], (255, 0, 0))
            cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
            cv2.polylines(frame, [self.out_polygon], True, (255, 0, 0), 3)

            # Add label at top of OUT region
            out_center_x = int(np.mean([p[0] for p in self.out_region]))
            out_center_y = int(np.min([p[1] for p in self.out_region])) - 10
            cv2.putText(frame, "OUT ZONE", (out_center_x - 60, out_center_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

        # Draw tracked vehicles
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

                # Draw track ID with status
                label = f"ID: {track_id}"
                if track_id in self.counted_in_ids:
                    label += " [IN]"
                elif track_id in self.counted_out_ids:
                    label += " [OUT]"

                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def draw_statistics(self, frame):
        """Draw count statistics on frame"""
        if self.enable_counting:
            # Background for statistics
            cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (400, 120), (255, 255, 255), 2)

            # Display IN count (green)
            cv2.putText(frame, f"CARS IN:  {self.in_count}", (20, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Display OUT count (blue)
            cv2.putText(frame, f"CARS OUT: {self.out_count}", (20, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame

    def process_video(self, output_path="output_in_out_count.mp4", display=True):
        """
        Process video and count vehicles going in and out

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
        if self.in_region:
            print(f"IN region: {len(self.in_region)} points defined")
        if self.out_region:
            print(f"OUT region: {len(self.out_region)} points defined")
        print("-" * 50)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLO detection with aggressive NMS to prevent duplicate detections
            # conf=0.6: High confidence for reliable detections
            # iou=0.3: Aggressive NMS - removes overlapping boxes more strictly (IMPORTANT for stationary cars)
            results = self.model(frame, conf=0.8, iou=0.3, imgsz=640, agnostic_nms=True, verbose=False)[0]

            # Prepare detections for DeepSORT
            detections = []
            MIN_CONFIDENCE = 0.6  # Match detection confidence

            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only track cars/trucks with sufficient confidence
                if self.is_car(class_id) and confidence >= MIN_CONFIDENCE:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Only add detection if it's inside either detection region
                    if (self.is_point_in_polygon(center_x, center_y, self.in_polygon) or
                        self.is_point_in_polygon(center_x, center_y, self.out_polygon)):
                        # DeepSORT expects [left, top, width, height]
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        detections.append((bbox, confidence, class_id))

            # Filter overlapping detections (CRITICAL for stationary vehicles)
            # Lower threshold = more aggressive filtering of duplicates
            raw_detection_count = len(detections)
            detections = self.filter_overlapping_detections(detections, iou_threshold=0.3)
            filtered_count = len(detections)

            # Log when duplicates are removed (helps verify stationary cars aren't tracked multiple times)
            if raw_detection_count > filtered_count:
                duplicates_removed = raw_detection_count - filtered_count
                print(f"Frame {frame_count}: Filtered {duplicates_removed} duplicate detection(s) | {raw_detection_count} -> {filtered_count}")

            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)

            # Track currently active IDs to detect lost tracks
            current_active_ids = set()

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

                # Track this ID as active
                current_active_ids.add(track_id)

                # Update position tracking
                self.track_positions[track_id] = (center_x, center_y, frame_count)

                # Check and count
                direction = self.count_vehicle(track_id, center_x, center_y, frame_count)
                if direction:
                    if direction == 'IN':
                        print(f"[IN] Frame {frame_count}: Car ID {track_id} entered! Total IN: {self.in_count} | All IN IDs: {sorted(self.counted_in_ids)}")
                    elif direction == 'OUT':
                        print(f"[OUT] Frame {frame_count}: Car ID {track_id} exited! Total OUT: {self.out_count} | All OUT IDs: {sorted(self.counted_out_ids)}")

            # Detect lost tracks (IDs that were active but are no longer present)
            for prev_id, (prev_x, prev_y, prev_frame) in list(self.track_positions.items()):
                if prev_id not in current_active_ids:
                    # Track was lost - save its last position
                    if prev_id not in self.lost_tracks:
                        self.lost_tracks[prev_id] = (prev_x, prev_y, frame_count)
                    # Remove from active tracking
                    del self.track_positions[prev_id]

            # Draw annotations
            frame = self.draw_annotations(frame, tracks)
            frame = self.draw_statistics(frame)

            # Write frame
            out.write(frame)

            # Display
            if display:
                display_frame = frame.copy()
                if self.width > 1280:
                    scale = 1280 / self.width
                    new_width = 1280
                    new_height = int(self.height * scale)
                    display_frame = cv2.resize(display_frame, (new_width, new_height))

                cv2.imshow('Car In/Out Counter', display_frame)

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
                print(f"Processed {frame_count} frames | FPS: {fps_actual:.2f} | IN: {self.in_count} | OUT: {self.out_count}")

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
        print(f"\nVEHICLE COUNTING (DUPLICATE-FREE):")
        print(f"  Total vehicles IN: {self.in_count}")
        print(f"  Unique IN track IDs: {sorted(self.counted_in_ids)}")
        print(f"  Total vehicles OUT: {self.out_count}")
        print(f"  Unique OUT track IDs: {sorted(self.counted_out_ids)}")
        print(f"  Net difference (IN - OUT): {self.in_count - self.out_count}")
        print(f"\nOutput saved to: {output_path}")
        print("=" * 50)


def main():
    # Configuration
    VIDEO_PATH = "video/20251223172447350.MP4"  # Change to your video path
    MODEL_PATH = "yolo12x.pt"
    OUTPUT_PATH = "output_in_out_count.mp4"

    # IN detection region (green zone)
    IN_REGION = [
        (416, 481),
        (1536, 377),
        (1593, 868),
        (449, 977)
    ]

    # OUT detection region (blue zone)
    OUT_REGION = [
        (354, 1058),
        (1555, 1001),
        (1598, 1535),
        (316, 1597)
    ]

    # Create counter
    counter = CarInOutCounter(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        in_region=IN_REGION,
        out_region=OUT_REGION,
        enable_counting=True,
        show_tracking=True
    )

    # Process video
    counter.process_video(output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()
