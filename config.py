"""
Configuration file for Hand Gesture Analyzer
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# Hand detection settings
HAND_DETECTION_CONFIDENCE = 0.5
HAND_TRACKING_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2  # Can detect up to 2 hands

# Finger counting settings
FINGER_TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Visualization settings
SHOW_HAND_LANDMARKS = True
SHOW_HAND_CONNECTIONS = True
SHOW_FINGER_COUNT = True
SHOW_HAND_LABEL = True  # Show "Left" or "Right"
FONT_SCALE = 1.0
FONT_THICKNESS = 2

# Optional features
ENABLE_TTS = False  # Text-to-speech
ENABLE_LOGGING = True
ENABLE_SCREENSHOTS = True

# Logging
LOG_FILE = os.path.join(LOG_DIR, 'hand_detections.csv')

# Colors (BGR format for OpenCV)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_CYAN = (255, 255, 0)
COLOR_MAGENTA = (255, 0, 255)

# Hand-specific colors
LEFT_HAND_COLOR = (0, 255, 0)   # Green
RIGHT_HAND_COLOR = (255, 0, 0)  # Blue
