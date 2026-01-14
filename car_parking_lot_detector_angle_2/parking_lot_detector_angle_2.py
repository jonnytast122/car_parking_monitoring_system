import cv2
import numpy as np
import json
from ultralytics import YOLO
import time
import torch


class ParkingLotDetector:
    def __init__(self, image_path, model_path, parking_spots_json,
                 conf_threshold=0.3, iou_threshold=0.4, car_conf_filter=0.3,
                 occupancy_threshold=0.5, imgsz=1280,
                 enable_enhancement=True, clahe_clip_limit=3.0, gamma=1.2):
        """
        Initialize the parking lot occupancy detector

        Args:
            image_path: Path to the image file
            model_path: Path to the YOLO model
            parking_spots_json: Path to JSON file with parking spot coordinates
            conf_threshold: YOLO confidence threshold (0.0-1.0). Lower values detect more but may have false positives. Default: 0.3
            iou_threshold: YOLO IoU threshold for NMS. Lower values keep more detections. Default: 0.4
            car_conf_filter: Additional confidence filter for cars. Default: 0.3
            occupancy_threshold: Minimum overlap (0.0-1.0) to mark spot occupied. Default: 0.5 (50%)
            imgsz: YOLO input image size. Higher = better small object detection. Default: 1280
            enable_enhancement: Enable CLAHE image enhancement for dark areas. Default: True
            clahe_clip_limit: CLAHE clip limit for contrast enhancement. Default: 3.0
            gamma: Gamma correction value for brightening dark areas. Default: 1.2
        """
        self.image_path = image_path
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
        self.occupancy_threshold = occupancy_threshold
        self.imgsz = imgsz

        # Image enhancement parameters
        self.enable_enhancement = enable_enhancement
        self.gamma = gamma
        if self.enable_enhancement:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(8, 8))
            print(f"Image enhancement enabled (CLAHE with clip limit: {clahe_clip_limit}, Gamma: {gamma})")
        else:
            print("Image enhancement disabled")

        # Load parking spot coordinates from JSON
        self.parking_spots = self.load_parking_spots(parking_spots_json)

        # Load image and get dimensions
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image from {image_path}")
        self.height, self.width = self.image.shape[:2]

        # Statistics
        self.total_spots = len(self.parking_spots)
        self.occupied_spots = 0
        self.available_spots = 0

        # Print configuration
        print(f"YOLO Configuration:")
        print(f"  - Confidence threshold: {self.conf_threshold}")
        print(f"  - IoU threshold: {self.iou_threshold}")
        print(f"  - Car confidence filter: {self.car_conf_filter}")
        print(f"  - Input image size: {self.imgsz}")
        print(f"  - Occupancy threshold: {self.occupancy_threshold * 100}%")

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

    def calculate_iou(self, bbox, polygon):
        """
        Calculate intersection area between bounding box and parking spot polygon

        Args:
            bbox: [x1, y1, x2, y2] bounding box
            polygon: numpy array of polygon points

        Returns:
            Intersection area as percentage of parking spot area
        """
        x1, y1, x2, y2 = bbox

        # Create mask for bounding box
        bbox_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (x1, y1), (x2, y2), 255, -1)

        # Create mask for parking polygon
        polygon_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [polygon], 255)

        # Calculate intersection
        intersection = cv2.bitwise_and(bbox_mask, polygon_mask)
        intersection_area = np.sum(intersection > 0)
        polygon_area = np.sum(polygon_mask > 0)

        if polygon_area == 0:
            return 0.0

        # Return intersection as percentage of parking spot area
        return intersection_area / polygon_area

    def is_spot_occupied(self, spot_polygon, detections):
        """
        Check if a parking spot is occupied by any detected car

        Args:
            spot_polygon: Polygon coordinates of parking spot
            detections: List of detected car bounding boxes

        Returns:
            True if occupied, False if available
        """
        for bbox in detections:
            overlap = self.calculate_iou(bbox, spot_polygon)
            if overlap >= self.occupancy_threshold:
                return True
        return False

    def draw_parking_spots(self, frame, spot_statuses):
        """
        Draw parking spots on frame with color indicating occupancy

        Args:
            frame: Video frame
            spot_statuses: List of boolean values (True=occupied, False=available)
        """
        for i, (polygon, is_occupied) in enumerate(zip(self.parking_spots, spot_statuses)):
            # Choose color: Red for occupied, Green for available
            color = (0, 0, 255) if is_occupied else (0, 255, 0)

            # Draw filled polygon with transparency
            overlay = frame.copy()
            cv2.fillPoly(overlay, [polygon], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

            # Draw polygon border
            cv2.polylines(frame, [polygon], True, color, 2)

            # Add spot number and status
            center = np.mean(polygon, axis=0).astype(int)
            status_text = "OCCUPIED" if is_occupied else "AVAILABLE"
            cv2.putText(frame, f"#{i+1}", (center[0]-20, center[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (center[0]-40, center[1]+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        return frame

    def draw_statistics(self, frame):
        """Draw occupancy statistics on frame"""
        # Background for statistics
        cv2.rectangle(frame, (10, 10), (400, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 130), (255, 255, 255), 2)

        # Display statistics
        cv2.putText(frame, f"Total Spots: {self.total_spots}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Occupied: {self.occupied_spots}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Available: {self.available_spots}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return frame

    def process_image(self, output_path="output_parking_lot.jpg", display=True):
        """
        Process image and detect parking lot occupancy

        Args:
            output_path: Path to save output image
            display: Whether to display image after processing
        """
        start_time = time.time()

        print(f"Processing image: {self.image_path}")
        print(f"Image size: {self.width}x{self.height}")
        print(f"Total parking spots: {self.total_spots}")
        print("-" * 50)

        # Make a copy of the original image to draw on
        frame = self.image.copy()

        # Enhance frame for better detection in dark areas
        processed_frame = self.preprocess_frame(frame)

        # Run YOLO detection with optimized parameters for small objects in dark areas
        results = self.model(
            processed_frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            agnostic_nms=True,
            verbose=False,
            device=self.device
        )[0]

        # Collect car detections
        car_detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Only process cars with additional confidence filter
            if self.is_car(class_id) and confidence >= self.car_conf_filter:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                car_detections.append([x1, y1, x2, y2])

                # Draw car bounding box on original frame (not enhanced)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(frame, f"Car {confidence:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        print(f"Detected {len(car_detections)} cars")

        # Check occupancy for each parking spot
        spot_statuses = []
        self.occupied_spots = 0
        self.available_spots = 0

        for spot_polygon in self.parking_spots:
            is_occupied = self.is_spot_occupied(spot_polygon, car_detections)
            spot_statuses.append(is_occupied)

            if is_occupied:
                self.occupied_spots += 1
            else:
                self.available_spots += 1

        # Draw parking spots with status
        frame = self.draw_parking_spots(frame, spot_statuses)

        # Draw statistics
        frame = self.draw_statistics(frame)

        # Save output image
        cv2.imwrite(output_path, frame)

        # Display
        if display:
            # Resize for display if too large
            display_frame = frame.copy()
            if self.width > 1280:
                scale = 1280 / self.width
                new_width = 1280
                new_height = int(self.height * scale)
                display_frame = cv2.resize(display_frame, (new_width, new_height))

            cv2.imshow('Parking Lot Detection', display_frame)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 50)
        print("PROCESSING COMPLETE")
        print("=" * 50)
        print(f"Processing time: {elapsed:.2f} seconds")
        print(f"Total parking spots: {self.total_spots}")
        print(f"Occupied: {self.occupied_spots}, Available: {self.available_spots}")
        print(f"Output saved to: {output_path}")
        print("=" * 50)


def main():
    # Configuration
    IMAGE_PATH = "sample_region.jpg"
    MODEL_PATH = "yolov12x.pt"
    PARKING_SPOTS_JSON = "sample_region.json"
    OUTPUT_PATH = "parking_output.jpg"

    # Create detector with optimized settings for small objects in dark/shadowed areas
    detector = ParkingLotDetector(
        image_path=IMAGE_PATH,
        model_path=MODEL_PATH,
        parking_spots_json=PARKING_SPOTS_JSON,
        # YOLO detection parameters (lowered for better detection in shadows)
        conf_threshold=0.2,        # Lower confidence for dark/shadowed areas
        iou_threshold=0.4,         # Lower IoU for small objects
        car_conf_filter=0.2,       # Lower car confidence filter for shadowed vehicles
        occupancy_threshold=0.5,   # 50% overlap to mark spot as occupied
        imgsz=1920,                # Higher resolution for better detection
        # Image enhancement for dark/shadowed areas
        enable_enhancement=True,   # Enable image enhancement
        clahe_clip_limit=4.0,      # Higher CLAHE for stronger contrast in shadows
        gamma=1.3                  # Gamma correction to brighten shadowed areas
    )

    # Process image
    detector.process_image(output_path=OUTPUT_PATH, display=True)


if __name__ == "__main__":
    main()
