# Autonomous Path Navigation System

A real-time autonomous path navigation and analysis system using a monocular camera (or video file). The system combines state-of-the-art computer vision models and pathfinding algorithms to provide adaptive cruise control, steering instructions, and collision avoidance.

## Features

*   **Object Detection:** Uses `YOLOv8` to detect vehicles, pedestrians, and obstacles in the frame.
*   **Object Tracking:** Implements `DeepSORT` to track objects across multiple frames, assigning consistent IDs to moving objects.
*   **Depth Estimation:** Utilizes the `MiDaS` model from PyTorch Hub to generate a relative depth map and determine the proximity of objects.
*   **Relative Velocity Calculation:** Analyzes depth changes over time to determine if a tracked object is moving closer (braking) or further away (accelerating).
*   **A\* Pathfinding:** Generates a real-time top-down obstacle grid from depth and tracking data, mapping a safe path to follow.
*   **Adaptive Driving Commands:** Intelligently issues driving commands (`ACCELERATE`, `MAINTAIN_SPEED`, `SLOW_DOWN`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`) based on a decoupled steering and speed-control logic, allowing for realistic Adaptive Cruise Control.

## Setup & Installation

1.  Navigate to your project directory.
2.  Install dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Dependencies include: `opencv-python`, `ultralytics`, `timm`, `deep-sort-realtime`, `torch`, `numpy`.*

## Usage

You can run the system in two modes:

### 1. Live Webcam Feed
To run the system using your default connected webcam:
```powershell
python live_video.py
```

### 2. Pre-recorded Dashcam Video
To run the system on a video file (e.g., a dashcam recording):
```powershell
python video_file.py --video path/to/your/video.mp4
```

## HUD Overview

The Heads-Up Display (HUD) provides the following insights:
*   **RISK SCORE**: A value representing the proximity of the closest object directly in front of the vehicle.
    *   *Green (< 160)*: Safe to accelerate or maintain speed.
    *   *Orange (> 160)*: Moderately close.
    *   *Red (> 220)*: Imminent collision.
*   **CMD**: The current autonomous driving command.
*   **Minimap (Top Right)**: A top-down grid visualization showing detected obstacles (in black) and the green safe path calculated by the A* algorithm.
*   **Bounding Boxes**: Highlights detected objects with their `ID`, proximity (`Prox`), and relative velocity (`RelV`).

## How it Works

The system operates on a tight continuous loop:
1. It reads a frame from the camera/video.
2. Passes it through **YOLOv8** for object localization and **DeepSORT** for tracking moving objects over time.
3. Passes the same frame to **MiDaS** for depth analysis.
4. Translates depth thresholds and tracked obstacle coordinates into a 2D grid.
5. Computes speed control decisions (Adaptive Cruise Control) independently of steering decisions so that the vehicle can safely follow the car in front or maneuver around static obstacles.
6. Combines all the visualizations and presents the final HUD output.
