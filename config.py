"""
Configuration file for Monster Analyzer
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Model paths
POSE_MODEL_PATH = os.path.join(MODEL_DIR, 'pose_landmark_lite.tflite')
OBJECT_DETECTION_MODEL_PATH = os.path.join(MODEL_DIR, 'detect.tflite')
FLAVOR_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'monster_flavor_classifier.tflite')

# Webcam settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS_TARGET = 30

# Detection settings
CONFIDENCE_THRESHOLD = 0.5
POSE_CONFIDENCE_THRESHOLD = 0.5
HAND_CAN_PROXIMITY_THRESHOLD = 100  # pixels

# Monster Energy flavors (update this list with actual flavors you want to detect)
MONSTER_FLAVORS = [
    "Monster Energy Original",
    "Monster Ultra White",
    "Monster Ultra Blue",
    "Monster Ultra Red",
    "Monster Ultra Paradise",
    "Monster Ultra Rosa",
    "Monster Ultra Gold",
    "Monster Ultra Violet",
    "Monster Ultra Sunrise",
    "Monster Ultra Watermelon",
    "Monster Mango Loco",
    "Monster Pacific Punch",
    "Monster Pipeline Punch",
    "Monster Juice Monster Pacific Punch",
    "Unknown Monster Flavor"
]

# Flavor emoji mapping (optional for fun display)
FLAVOR_EMOJIS = {
    "Monster Energy Original": "⚡",
    "Monster Ultra White": "❄️",
    "Monster Ultra Blue": "💙",
    "Monster Ultra Red": "❤️",
    "Monster Ultra Paradise": "🍃",
    "Monster Ultra Rosa": "🌸",
    "Monster Ultra Gold": "🏆",
    "Monster Ultra Violet": "💜",
    "Monster Ultra Sunrise": "🌅",
    "Monster Ultra Watermelon": "🍉",
    "Monster Mango Loco": "🥭",
    "Monster Pacific Punch": "🌊",
    "Monster Pipeline Punch": "🌺",
    "Unknown Monster Flavor": "❓"
}

# Visualization settings
SHOW_POSE_SKELETON = True
SHOW_BOUNDING_BOXES = True
SHOW_CONFIDENCE_BAR = True
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Optional features
ENABLE_TTS = False  # Text-to-speech
ENABLE_LOGGING = True  # Log detections
LOG_FILE = os.path.join(LOG_DIR, 'detections.csv')

# Colors (BGR format for OpenCV)
COLOR_SKELETON = (0, 255, 0)  # Green
COLOR_BBOX_CAN = (0, 255, 255)  # Yellow
COLOR_BBOX_HAND = (255, 0, 255)  # Magenta
COLOR_TEXT = (255, 255, 255)  # White
COLOR_CONFIDENCE_BAR_BG = (50, 50, 50)  # Dark gray
COLOR_CONFIDENCE_BAR_FG = (0, 255, 0)  # Green

# Hand keypoint indices for MediaPipe Pose
# Wrist indices in MediaPipe Pose Landmarker
LEFT_WRIST_IDX = 15
RIGHT_WRIST_IDX = 16

# Object detection labels
OBJECT_LABELS = ['can', 'bottle', 'cup']  # Adjust based on your model
