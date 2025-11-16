# Hand Gesture Analyzer

Real-time hand detection and finger counting using MediaPipe and OpenCV

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red)

## Overview

Hand Gesture Analyzer is an interactive AI application that uses your laptop webcam to:

- Detect hands in real-time (left and right)
- Count fingers extended on each hand (0-5)
- Display live visual overlays with hand landmarks and finger counts
- Log detections for analysis and tracking

### Core Features

Real-time Hand Detection

- Powered by MediaPipe Hands for accurate hand landmark detection
- Supports simultaneous detection of both left and right hands
- High-performance tracking with minimal latency

Finger Counting

- Accurately counts extended fingers (0-5) on each hand
- Distinguishes between left and right hands
- Individual finger status tracking (thumb, index, middle, ring, pinky)
  - Also a fun feature -> Detects when you throw the finger and alerts

Visual Overlays

- Hand skeleton visualization with landmarks and connections
- Color-coded hands (Green for left, Blue for right)
- Live finger count display
- FPS monitoring
- Detection statistics

Optional Features

- Text-to-speech announcements
- Detection logging to CSV
- Screenshot capture
- Session statistics

## Project Structure

```
Hand-Gesture-Analyzer/
├── hand_gesture_analyzer.py    # Main application
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── src/
│   ├── __init__.py
│   ├── models.py               # Hand detection and finger counting
│   └── visualization.py        # Drawing utilities
├── data/                       # Screenshots and data
├── logs/                       # Detection logs
│   └── hand_detections.csv
└── README.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/macOS/Linux

### Setup

1. Clone or download this repository

```bash
cd Hand-Gesture-Analyzer
```

2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\\Scripts\\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run the main application:

```bash
python hand_gesture_analyzer.py
```

### Controls

- 'q' - Quit the application
- 's' - Save screenshot of current frame
- 'r' - Reset detection statistics

### What You'll See

The application displays:

- Live camera feed with your hands detected
- Hand landmarks (21 points per hand) connected with lines
- Finger count for each detected hand at the top
- Hand label (Left/Right) color-coded
- FPS counter at top-left
- Detection statistics at bottom

### How It Works

1. Hand Detection: MediaPipe Hands detects up to 2 hands in the frame
2. Landmark Tracking: 21 landmarks per hand are tracked in 3D space
3. Finger Counting: Algorithm determines which fingers are extended based on landmark positions
4. Handedness: Left vs Right hand is automatically determined
5. Visualization: Results are overlaid on the live video feed

## Configuration

Edit `config.py` to customize:

```python
# Detection settings
MAX_NUM_HANDS = 2  # Maximum hands to detect (1 or 2)
HAND_DETECTION_CONFIDENCE = 0.5  # Detection threshold (0-1)
HAND_TRACKING_CONFIDENCE = 0.5  # Tracking threshold (0-1)

# Visualization
SHOW_HAND_LANDMARKS = True
SHOW_HAND_CONNECTIONS = True
SHOW_FINGER_COUNT = True

# Optional features
ENABLE_TTS = False  # Text-to-speech announcements
ENABLE_LOGGING = True  # Log detections to CSV
ENABLE_SCREENSHOTS = True  # Allow screenshots
```

## How Finger Counting Works

The algorithm uses hand landmark positions to determine finger states:

- Thumb: Checks if thumb tip is to the left/right of thumb IP joint (handedness-dependent)
- Other Fingers: Checks if fingertip is above the PIP joint (second joint from tip)

Each finger is independently classified as UP or DOWN, and the total count (0-5) is calculated.

## Detection Logging

When `ENABLE_LOGGING = True`, detections are saved to `logs/hand_detections.csv`:

```csv
timestamp,hand_label,finger_count,thumb,index,middle,ring,pinky
2025-11-15 10:30:45.123,Left,3,True,True,True,False,False
2025-11-15 10:30:45.156,Right,5,True,True,True,True,True
```

## Use Cases

- Sign language recognition foundation
- Gesture-based control interfaces
- Hand therapy and rehabilitation tracking
- Educational demonstrations
- Input systems that can be used for gaming
- Accessibility tools

## Technical Details

### Hand Landmarks

MediaPipe Hands provides 21 landmarks per hand:

- 0: Wrist
- 1-4: Thumb (CMC, MCP, IP, TIP)
- 5-8: Index finger (MCP, PIP, DIP, TIP)
- 9-12: Middle finger (MCP, PIP, DIP, TIP)
- 13-16: Ring finger (MCP, PIP, DIP, TIP)
- 17-20: Pinky (MCP, PIP, DIP, TIP)

### Performance

- FPS: 25-30 on average laptop webcam
- Latency: <50ms processing time per frame
- Accuracy: >95% finger counting accuracy in good lighting

## Troubleshooting

Camera not detected

- Check `config.py` and set correct `CAMERA_INDEX` (usually 0)
- Ensure no other app is using the webcam

Low FPS

- Reduce `FRAME_WIDTH` and `FRAME_HEIGHT` in `config.py`
- Set `MAX_NUM_HANDS = 1` if only detecting one hand

Inaccurate finger counting

- Ensure good lighting conditions
- Keep hand flat and fingers clearly separated
- Adjust `HAND_DETECTION_CONFIDENCE` threshold

Import errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Activate your virtual environment if using one

## Requirements

- Python 3.8+
- opencv-python
- mediapipe
- numpy
- pyttsx3 (optional, for text-to-speech)

See `requirements.txt` for complete list.

## Acknowledgments

- MediaPipe by Google for hand detection and landmark tracking
- OpenCV for computer vision utilities

## Future Enhancements

- Gesture recognition (peace sign, thumbs up, etc.)
- Hand pose classification
- Multi-hand gesture combinations
