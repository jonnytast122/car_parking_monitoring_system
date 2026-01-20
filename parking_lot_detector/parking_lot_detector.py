import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import torch


class ParkingLotDetector:
    def __init__(self, image_paths, model_path, parking_spots_jsons,
                 conf_threshold=0.3, iou_threshold=0.4, car_conf_filter=0.3,
                 imgsz=1280,
                 enable_enhancement=True, clahe_clip_limit=3.0, gamma=1.2):
        """
        Initialize the parking lot occupancy detector for multiple images

        Args:
            image_paths: List of paths to image files (e.g., [image1.jpg, image2.jpg])
            model_path: Path to the YOLO model
            parking_spots_jsons: List of paths to JSON files with parking spot coordinates
            conf_threshold: YOLO confidence threshold (0.0-1.0). Lower values detect more but may have false positives. Default: 0.3
            iou_threshold: YOLO IoU threshold for NMS. Lower values keep more detections. Default: 0.4
            car_conf_filter: Additional confidence filter for cars. Default: 0.3
            imgsz: YOLO input image size. Higher = better small object detection. Default: 1280
            enable_enhancement: Enable CLAHE image enhancement for dark areas. Default: True
            clahe_clip_limit: CLAHE clip limit for contrast enhancement. Default: 3.0
            gamma: Gamma correction value for brightening dark areas. Default: 1.2
        """
        # Convert single inputs to lists for consistency
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if isinstance(parking_spots_jsons, str):
            parking_spots_jsons = [parking_spots_jsons]

        self.image_paths = image_paths
        self.parking_spots_jsons = parking_spots_jsons
        self.num_images = len(image_paths)
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

        # YOLO detection parameters
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.car_conf_filter = car_conf_filter
        self.imgsz = imgsz

        # Image enhancement parameters
        self.enable_enhancement = enable_enhancement
        self.gamma = gamma
        if self.enable_enhancement:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            print(f"Image enhancement enabled (CLAHE with clip limit: {clahe_clip_limit}, Gamma: {gamma})")
        else:
            print("Image enhancement disabled")

        # Load parking spot coordinates from JSON for each image
        self.parking_spots_list = []
        for json_path in self.parking_spots_jsons:
            spots = self.load_parking_spots(json_path)
            self.parking_spots_list.append(spots)

        # Load images and get dimensions
        self.images = []
        self.dimensions = []
        for i, image_path in enumerate(self.image_paths):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            self.images.append(image)
            h, w = image.shape[:2]
            self.dimensions.append((w, h))
            print(f"Loaded image {i+1}: {image_path} ({w}x{h})")

        # Use first image dimensions as reference
        self.width, self.height = self.dimensions[0]

        # Statistics for each image
        self.total_spots_list = [len(spots) for spots in self.parking_spots_list]
        self.occupied_spots_list = [0] * self.num_images
        self.available_spots_list = [0] * self.num_images

        # Combined statistics
        self.total_spots = sum(self.total_spots_list)
        self.occupied_spots = 0
        self.available_spots = 0

        # Print configuration
        print(f"\nYOLO Configuration:")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        print(f"  - IoU threshold: {self.iou_threshold}")
        print(f"  - Car confidence filter: {self.car_conf_filter}")
        print(f"  - Input image size: {self.imgsz}")
        print(f"  - Occupancy detection: Center point inside polygon")
        print(f"  - Number of images: {self.num_images}")
        print(f"  - Total parking spots across all images: {self.total_spots}")

    def load_parking_spots(self, json_path):
        """Load parking spot coordinates from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        parking_spots = []
        for shape in data['shapes']:
            if shape['label'] == 'parking' and shape['shape_type'] == 'polygon':
                # Convert points to numpy array
                points = np.array(shape['points'], dtype=np.int32)
                parking_spots.append(points)

        print(f"Loaded {len(parking_spots)} parking spots from JSON")
        return parking_spots

    def is_car(self, class_id):
        """Check if detected object is a car (class 2 in COCO dataset)"""
        # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        # Only detect cars for parking spots
        return class_id == 2

    def apply_gamma_correction(self, frame, gamma):
        """
        Apply gamma correction to brighten dark areas

        Args:
            frame: Input BGR frame
            gamma: Gamma value (>1.0 brightens, <1.0 darkens)

        Returns:
            Gamma corrected frame
        """
        # Build lookup table for gamma correction
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                         for i in np.arange(0, 256)]).astype("uint8")

        # Apply gamma correction using lookup table
        return cv2.LUT(frame, table)

    def preprocess_frame(self, frame):
        """
        Enhance frame for better detection in dark areas using gamma correction and CLAHE

        Applies both gamma correction to brighten dark areas and CLAHE for adaptive
        contrast enhancement. This combination is particularly effective for detecting
        objects in shadowed areas like cars under trees.

        Args:
            frame: Input BGR frame

        Returns:
            Enhanced frame (or original if enhancement is disabled)
        """
        if not self.enable_enhancement:
            return frame

        # First apply gamma correction to brighten dark/shadowed areas
        gamma_corrected = self.apply_gamma_correction(frame, self.gamma)

        # Convert to LAB color space
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel (lightness)
        l_enhanced = self.clahe.apply(l)

        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        return enhanced_bgr

    def get_bbox_center(self, bbox):
        """
        Calculate the center point of a bounding box

        Args:
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            (center_x, center_y) tuple
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        return (center_x, center_y)

    def is_point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using OpenCV pointPolygonTest

        Args:
            point: (x, y) tuple representing the point
            polygon: numpy array of polygon points

        Returns:
            True if point is inside polygon, False otherwise
        """
        # pointPolygonTest returns positive if inside, negative if outside, 0 if on edge
        result = cv2.pointPolygonTest(polygon, point, False)
        return result >= 0  # Inside or on edge

    def is_spot_occupied(self, spot_polygon, detections):
        """
        Check if a parking spot is occupied by any detected car
        Uses center point of car bounding box to determine occupancy

        Args:
            spot_polygon: Polygon coordinates of parking spot
            detections: List of detected car bounding boxes

        Returns:
            True if occupied, False if available
        """
        for bbox in detections:
            # Get the center point of the car's bounding box
            center = self.get_bbox_center(bbox)
            # Check if center point is inside the parking polygon
            if self.is_point_in_polygon(center, spot_polygon):
                return True
        return False

    def draw_parking_spots(self, frame, spot_statuses, img_idx):
        """
        Draw parking spots on frame with color indicating occupancy

        Args:
            frame: Video frame
            spot_statuses: List of boolean values (True=occupied, False=available)
            img_idx: Index of the image being processed
        """
        parking_spots = self.parking_spots_list[img_idx]
        width, height = self.dimensions[img_idx]

        # Calculate scale factor based on image width (reference: 1920px)
        scale = width / 1920.0
        scale = max(0.25, min(scale, 1.5))

        font_scale_number = 0.35 * scale
        font_scale_status = 0.25 * scale
        thickness = max(1, int(2 * scale))
        border_thickness = max(1, int(2 * scale))
        offset_x = int(15 * scale)
        offset_y = int(8 * scale)

        for i, (polygon, is_occupied) in enumerate(zip(parking_spots, spot_statuses)):
            # Choose color: Red for occupied, Green for available
            color = (0, 0, 255) if is_occupied else (0, 255, 0)

            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw polygon border
            cv2.polylines(frame, [polygon], True, color, border_thickness)

            # Add spot number and status
            center = np.mean(polygon, axis=0).astype(int)
            status_text = "OCCUPIED" if is_occupied else "AVAILABLE"
            cv2.putText(frame, f"#{i+1}", (center[0] - offset_x, center[1] - offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_number, (255, 255, 255), thickness)
            cv2.putText(frame, status_text, (center[0] - int(offset_x * 2), center[1] + offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale_status, (255, 255, 255), max(1, thickness // 2))

        return frame

    def draw_statistics(self, frame, img_idx):
        """Draw occupancy statistics on frame for specific image"""
        width, height = self.dimensions[img_idx]

        # Calculate scale factor based on image width (reference: 1920px)
        scale = width / 1920.0
        scale = max(0.3, min(scale, 1.5))

        box_width = int(250 * scale)
        box_height = int(110 * scale)
        padding = int(10 * scale)
        font_scale = 0.45 * scale
        thickness = max(1, int(2 * scale))
        line_height = int(22 * scale)

        # Background for statistics
        cv2.rectangle(frame, (padding, padding), (box_width, box_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (padding, padding), (box_width, box_height), (255, 255, 255), max(1, thickness // 2))

        # Display statistics for this image
        text_x = int(15 * scale)
        cv2.putText(frame, f"Camera {img_idx + 1}", (text_x, padding + line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), thickness)
        cv2.putText(frame, f"Total Spots: {self.total_spots_list[img_idx]}", (text_x, padding + line_height * 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        cv2.putText(frame, f"Occupied: {self.occupied_spots_list[img_idx]}", (text_x, padding + line_height * 3),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)
        cv2.putText(frame, f"Available: {self.available_spots_list[img_idx]}", (text_x, padding + line_height * 4),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

        return frame

    def process_single_image(self, img_idx):
        """
        Process a single image and return the annotated frame

        Args:
            img_idx: Index of the image to process

        Returns:
            Annotated frame with parking spot detection
        """
        image = self.images[img_idx]
        parking_spots = self.parking_spots_list[img_idx]
        width, height = self.dimensions[img_idx]

        print(f"\nProcessing image {img_idx + 1}: {self.image_paths[img_idx]}")
        print(f"Image size: {width}x{height}")
        print(f"Parking spots: {len(parking_spots)}")

        # Make a copy of the original image to draw on
        frame = image.copy()

        # Enhance frame for better detection in dark areas
        processed_frame = self.preprocess_frame(frame)

        # Run YOLO detection
        results = self.model(
            processed_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            agnostic_nms=True,
            verbose=False,
            device=self.device
        )[0]

        # Calculate scale factor for car labels
        label_scale = width / 1920.0
        label_scale = max(0.25, min(label_scale, 1.5))
        car_font_scale = 0.35 * label_scale
        car_thickness = max(1, int(2 * label_scale))
        car_text_offset = int(8 * label_scale)

        # Collect car detections
        car_detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if self.is_car(class_id) and confidence >= self.car_conf_filter:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_detections.append([x1, y1, x2, y2])

                # Draw car bounding box
                if confidence > 0.48:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), car_thickness)
                    cv2.putText(frame, f"Car {confidence:.2f}", (x1, y1 - car_text_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, car_font_scale, (255, 0, 255), car_thickness)

        print(f"Detected {len(car_detections)} cars")

        # Check occupancy for each parking spot
        spot_statuses = []
        self.occupied_spots_list[img_idx] = 0
        self.available_spots_list[img_idx] = 0

        for spot_polygon in parking_spots:
            is_occupied = self.is_spot_occupied(spot_polygon, car_detections)
            spot_statuses.append(is_occupied)

            if is_occupied:
                self.occupied_spots_list[img_idx] += 1
            else:
                self.available_spots_list[img_idx] += 1

        # Draw parking spots with status
        frame = self.draw_parking_spots(frame, spot_statuses, img_idx)

        # Draw statistics
        frame = self.draw_statistics(frame, img_idx)

        return frame

    def process_images(self, output_path="output_parking_lot.jpg", display=True):
        """
        Process all images and display them stacked vertically (top and bottom)

        Args:
            output_path: Path to save combined output image
            display: Whether to display image after processing
        """
        start_time = time.time()

        print("=" * 60)
        print(f"PROCESSING {self.num_images} IMAGES")
        print("=" * 60)

        # Process each image
        processed_frames = []
        for img_idx in range(self.num_images):
            frame = self.process_single_image(img_idx)
            processed_frames.append(frame)

        # Calculate combined statistics
        self.occupied_spots = sum(self.occupied_spots_list)
        self.available_spots = sum(self.available_spots_list)

        # Resize all frames to the same width for stacking
        target_width = max(self.dimensions[i][0] for i in range(self.num_images))
        resized_frames = []
        for i, frame in enumerate(processed_frames):
            w, h = self.dimensions[i]
            if w != target_width:
                scale = target_width / w
                new_height = int(h * scale)
                frame = cv2.resize(frame, (target_width, new_height))
            resized_frames.append(frame)

        # Stack frames vertically (top to bottom)
        combined_frame = np.vstack(resized_frames)

        # Add combined statistics bar at the bottom
        combined_height, combined_width = combined_frame.shape[:2]
        stats_bar_height = 50
        stats_bar = np.zeros((stats_bar_height, combined_width, 3), dtype=np.uint8)

        # Draw combined statistics on the bar
        cv2.putText(stats_bar, f"COMBINED: Total: {self.total_spots} | Occupied: {self.occupied_spots} | Available: {self.available_spots}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Append stats bar to combined frame
        combined_frame = np.vstack([combined_frame, stats_bar])

        # Save output image
        cv2.imwrite(output_path, combined_frame)

        # Display
        if display:
            # Resize for display if too large
            display_frame = combined_frame.copy()
            display_height, display_width = display_frame.shape[:2]

            if display_width > 1280 or display_height > 900:
                scale = min(1280 / display_width, 900 / display_height)
                new_width = int(display_width * scale)
                new_height = int(display_height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))

            cv2.imshow('Parking Lot Detection - Multi Camera', display_frame)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Images processed: {self.num_images}")
        for i in range(self.num_images):
            print(f"  Camera {i+1}: {self.occupied_spots_list[i]} occupied, {self.available_spots_list[i]} available")
        print(f"\nCOMBINED TOTAL:")
        print(f"  Total parking spots: {self.total_spots}")
        print(f"  Occupied: {self.occupied_spots}")
        print(f"  Available: {self.available_spots}")
        print(f"\nOutput saved to: {output_path}")
        print("=" * 60)


def main():
    # Configuration for multiple images
    IMAGE_PATHS = [
        "IMG_1115.JPG",  # Camera 1 (top)
        "IMG_1116.JPG",  # Camera 2 (bottom)
    ]
    MODEL_PATH = "yolov12x.pt"
    PARKING_SPOTS_JSONS = [
        "IMG_1115.json",  # JSON for Camera 1
        "IMG_1116.json",  # JSON for Camera 2
    ]
    OUTPUT_PATH = "combined_parking_output.jpg"

    # Create detector with optimized settings for multiple images
    detector = ParkingLotDetector(
        image_paths=IMAGE_PATHS,
        model_path=MODEL_PATH,
        parking_spots_jsons=PARKING_SPOTS_JSONS,
        # YOLO detection parameters
        conf_threshold=0.2,
        iou_threshold=0.4,
        car_conf_filter=0.2,
        imgsz=1920,
        # Image enhancement
        enable_enhancement=True,
        clahe_clip_limit=4.0,
        gamma=1.3
    )

    # Process all images and display stacked vertically
    detector.process_images(output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()