"""
Quick Start Guide for Hand Gesture Analyzer
Simple script to get you started quickly
"""

print("""
╔═══════════════════════════════════════════════════════════════╗
║         Hand Gesture Analyzer - Quick Start Guide            ║
╚═══════════════════════════════════════════════════════════════╝

SETUP:
------
1. Make sure you have Python 3.8+ installed
2. Install dependencies:
   
   pip install -r requirements.txt

3. Run the analyzer:
   
   python hand_gesture_analyzer.py


CONTROLS:
---------
While running:
  • Press 'q' to quit
  • Press 's' to save a screenshot
  • Press 'r' to reset detection statistics


WHAT IT DOES:
-------------
✓ Detects your hands (left and/or right) in real-time
✓ Counts how many fingers are extended (0-5)
✓ Shows visual overlays with hand landmarks
✓ Displays which hand (Left/Right) is detected
✓ Logs detections to CSV file (optional)


TIPS:
-----
• Ensure good lighting for best results
• Keep your hand flat with fingers clearly separated
• The camera feed is mirrored for intuitive interaction
• Green = Left hand, Blue = Right hand


CONFIGURATION:
--------------
Edit config.py to customize:
  • MAX_NUM_HANDS: Detect 1 or 2 hands
  • ENABLE_TTS: Turn on voice announcements
  • ENABLE_LOGGING: Save detections to CSV
  • Hand detection confidence thresholds


TROUBLESHOOTING:
----------------
Camera not working?
  → Check CAMERA_INDEX in config.py (usually 0)

Low performance?
  → Reduce FRAME_WIDTH/FRAME_HEIGHT in config.py
  → Set MAX_NUM_HANDS = 1

Fingers not counting correctly?
  → Ensure good lighting
  → Keep fingers clearly separated
  → Face your palm toward the camera


Ready to go! Run: python hand_gesture_analyzer.py

═══════════════════════════════════════════════════════════════
""")

if __name__ == "__main__":
    import os
    import sys
    
    # Check if dependencies are installed
    try:
        import cv2
        import mediapipe as mp
        import numpy as np
        print("✓ All required dependencies are installed!")
        print("\nYou're ready to run:")
        print("  python hand_gesture_analyzer.py")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
