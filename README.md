# Vehicle Detection and Counting System

Automated vehicle counting and parking detection system using YOLO and DeepSORT for entrance/exit monitoring and parking occupancy tracking.

## Overview

This repository contains three main vehicle detection and counting systems:

1. **BDC Car Entrance and Exit Counter** - Separate entrance and exit monitoring with individual cameras
2. **Mall Car Entrance and Exit Counter** - Bidirectional counting with single camera and dual zones
3. **Parking Lot Detector** - Parking space occupancy detection and monitoring



## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd car-detection-in-and-out

# Install dependencies
pip install opencv-python numpy ultralytics deep-sort-realtime
```

### Run BDC Car Entrance and Exit Counter System

```bash
cd bdc_car_entrance_and_exit
python car_entrance_counter.py  # For entrance
python car_exit_counter.py      # For exit
```

### Run Mall Car Entrance and Exit Counter System

```bash
cd mall_car_entrance_and_exit
python car_in_out_counter.py
```

### Run Parking Detector Angle 1

```bash
cd car_parking_lot_detector_angle_1
python parking_lot_detector_angle_1.py
```

### Run Parking Detector Angle 2

```bash
cd car_parking_lot_detector_angle_2
python parking_lot_detector_angle_2.py
```

## Repository Structure

```
car-detection-in-and-out/
├── bdc_car_entrance_and_exit/          # BDC facility monitoring
│   ├── BDC-Entrance-Cuted.mp4          # Input video (entrance) *
│   ├── BDC-exit-Cuted.mp4              # Input video (exit) *
│   ├── car_entrance_counter.py         # Entrance counting script
│   ├── car_exit_counter.py             # Exit counting script
│   ├── output_entrance_count.mp4       # Output video (entrance) *
│   ├── output_exit_count.mp4           # Output video (exit) *
│   └── README.md                       # Detailed documentation
│
├── mall_car_entrance_and_exit/         # Mall parking monitoring
│   ├── 20251223172447350.MP4           # Input video *
│   ├── car_in_out_counter.py           # Bidirectional counting script
│   ├── output_in_out_count.mp4         # Output video *
│   └── README.md                       # Detailed documentation
│
├── car_parking_lot_detector_angle_1/   # Parking detector (angle 1)
│   ├── parking_lot_detector_angle_1.py # Parking detection script
│   ├── *.jpg                           # Reference frame *
│   └── *.json                          # Parking spot annotations *
│
├── car_parking_lot_detector_angle_2/   # Parking detector (angle 2)
│   ├── parking_detector.py             # Parking detection script
│   ├── *.jpg                           # Reference frame *
│   └── *.json                          # Parking spot annotations *
│
├── .gitignore                          # Git ignore rules
├── README.md                             # Project overview                         
└── requirements.txt                    # Python dependencies
```                         

