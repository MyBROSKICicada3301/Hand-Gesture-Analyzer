"""
Hand Gesture Analyzer - Real-time Hand and Finger Detection
Main application for detecting hands and counting fingers using webcam
"""
import cv2
import numpy as np
import time
import csv
import os
from datetime import datetime
from typing import Optional

# Import project modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import config
from src.models import HandDetector
from src.visualization import (
    draw_hand_landmarks, draw_finger_count, draw_bounding_box,
    draw_fps, draw_status_message, draw_hand_summary
)

# Optional: Text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class HandGestureAnalyzer:
    """Main application class for hand gesture recognition"""
    
    def __init__(self):
        """Initialize the Hand Gesture Analyzer"""
        print("=" * 60)
        print("Hand Gesture Analyzer - Real-time Finger Counting")
        print("=" * 60)
        
        # Initialize hand detector
        self.hand_detector = HandDetector(
            max_num_hands=config.MAX_NUM_HANDS,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE
        )
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam!")
        
        print(f"Webcam initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        
        # Initialize TTS if enabled
        self.tts_engine = None
        if config.ENABLE_TTS and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                print("Text-to-speech enabled")
            except:
                print("Text-to-speech initialization failed")
        
        # State tracking
        self.last_announced_count = {}  # Track last announced count for each hand
        self.detection_count = {'left': 0, 'right': 0}
        self.screenshot_count = 0
        
        # Initialize logging
        if config.ENABLE_LOGGING:
            self._initialize_logging()
        
        print("\nHand Gesture Analyzer ready!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'r' to reset detection stats")
        print("=" * 60 + "\n")
    
    def _initialize_logging(self):
        """Initialize CSV logging"""
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Create CSV file with headers if it doesn't exist
        if not os.path.exists(config.LOG_FILE) or os.path.getsize(config.LOG_FILE) == 0:
            with open(config.LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'hand_label', 'finger_count', 
                    'thumb', 'index', 'middle', 'ring', 'pinky'
                ])
    
    def _log_detection(self, hand_info: dict):
        """Log hand detection to CSV"""
        if not config.ENABLE_LOGGING:
            return
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        fingers = hand_info['fingers_up']
        
        with open(config.LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                hand_info['label'],
                hand_info['finger_count'],
                fingers[0],  # thumb
                fingers[1],  # index
                fingers[2],  # middle
                fingers[3],  # ring
                fingers[4]   # pinky
            ])
    
    def _announce(self, text: str):
        """Announce text via TTS"""
        if self.tts_engine:
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                pass
    
    def _save_screenshot(self, frame: np.ndarray):
        """Save current frame as screenshot"""
        if not config.ENABLE_SCREENSHOTS:
            return
        
        os.makedirs(config.DATA_DIR, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(config.DATA_DIR, f'screenshot_{timestamp}.jpg')
        cv2.imwrite(filename, frame)
        self.screenshot_count += 1
        print(f"Screenshot saved: {filename}")
    
    def _is_middle_finger_only(self, fingers_up: list) -> bool:
        """
        Check if only the middle finger is up
        
        Args:
            fingers_up: List of boolean values [thumb, index, middle, ring, pinky]
            
        Returns:
            True if only middle finger is extended
        """
        # Middle finger is at index 2
        # Check if middle is up and all others are down
        return (not fingers_up[0] and  # thumb down
                not fingers_up[1] and  # index down
                fingers_up[2] and      # middle up
                not fingers_up[3] and  # ring down
                not fingers_up[4])     # pinky down
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Processed frame with visualizations
        """
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hand_detector.detect_hands(frame_rgb)
        
        # Get hand information
        hands_info = self.hand_detector.get_hand_info(
            results, 
            config.FRAME_WIDTH, 
            config.FRAME_HEIGHT
        )
        
        # Draw visualizations
        if config.SHOW_HAND_LANDMARKS and results.multi_hand_landmarks:
            for hand_info in hands_info:
                draw_hand_landmarks(
                    frame, 
                    hand_info['landmarks'], 
                    hand_info['label'],
                    config.LEFT_HAND_COLOR if hand_info['label'] == 'Left' else config.RIGHT_HAND_COLOR
                )
        
        # Draw hand summary at top
        draw_hand_summary(frame, hands_info)
        
        # Check for middle finger gesture and display warning
        middle_finger_detected = False
        for hand_info in hands_info:
            if self._is_middle_finger_only(hand_info['fingers_up']):
                middle_finger_detected = True
                break
        
        if middle_finger_detected:
            # Draw warning message in large red text
            warning_text = "No No Nooo... Bad move soldier"
            
            # Get text size for centering
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            thickness = 3
            (text_width, text_height), baseline = cv2.getTextSize(
                warning_text, font, font_scale, thickness
            )
            
            # Calculate center position
            x = (config.FRAME_WIDTH - text_width) // 2
            y = config.FRAME_HEIGHT // 2
            
            # Draw background rectangle
            padding = 20
            cv2.rectangle(
                frame,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + padding),
                (0, 0, 0),
                -1
            )
            
            # Draw red border
            cv2.rectangle(
                frame,
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + padding),
                (0, 0, 255),
                4
            )
            
            # Draw shadow
            cv2.putText(
                frame,
                warning_text,
                (x + 3, y + 3),
                font,
                font_scale,
                (0, 0, 0),
                thickness + 2
            )
            
            # Draw main text in red
            cv2.putText(
                frame,
                warning_text,
                (x, y),
                font,
                font_scale,
                (0, 0, 255),
                thickness
            )
        
        # Process each detected hand
        for hand_info in hands_info:
            hand_label = hand_info['label'].lower()
            finger_count = hand_info['finger_count']
            
            # Update detection count
            self.detection_count[hand_label] += 1
            
            # Log detection
            self._log_detection(hand_info)
            
            # Announce if changed (and TTS enabled)
            if config.ENABLE_TTS:
                last_count = self.last_announced_count.get(hand_label)
                if last_count != finger_count:
                    self._announce(f"{hand_info['label']} hand: {finger_count} fingers")
                    self.last_announced_count[hand_label] = finger_count
        
        return frame
    
    def run(self):
        """Main application loop"""
        fps_time = time.time()
        fps = 0
        
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Mirror frame for more intuitive interaction
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame = self.process_frame(frame)
                
                # Calculate FPS
                current_time = time.time()
                fps = 1.0 / (current_time - fps_time)
                fps_time = current_time
                
                # Draw FPS
                draw_fps(frame, fps)
                
                # Draw status message
                status = f"Detections - Left: {self.detection_count['left']} | Right: {self.detection_count['right']}"
                draw_status_message(frame, status)
                
                # Display frame
                cv2.imshow('Hand Gesture Analyzer', frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self._save_screenshot(frame)
                elif key == ord('r'):
                    self.detection_count = {'left': 0, 'right': 0}
                    print("\nDetection stats reset")
        
        finally:
            # Cleanup
            self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        print("\nCleaning up...")
        self.cap.release()
        cv2.destroyAllWindows()
        self.hand_detector.close()
        
        # Print final stats
        print("\n" + "=" * 60)
        print("Session Summary:")
        print(f"  Left hand detections: {self.detection_count['left']}")
        print(f"  Right hand detections: {self.detection_count['right']}")
        print(f"  Screenshots saved: {self.screenshot_count}")
        if config.ENABLE_LOGGING:
            print(f"  Log file: {config.LOG_FILE}")
        print("=" * 60)


def main():
    """Entry point"""
    try:
        analyzer = HandGestureAnalyzer()
        analyzer.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
