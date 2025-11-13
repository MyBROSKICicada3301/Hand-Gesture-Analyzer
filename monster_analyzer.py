"""
Monster Analyzer - Real-time Monster Energy Flavor Detection
Main application for detecting and classifying Monster Energy cans using webcam
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
from src.models import PoseEstimator, ObjectDetector, FlavorClassifier, check_proximity
from src.visualization import (
    draw_pose_skeleton, draw_bounding_box, draw_flavor_prediction,
    draw_confidence_bar, draw_wrist_indicator, draw_fps, draw_status_message
)

# Optional: Text-to-speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False


class MonsterAnalyzer:
    """Main application class for Monster Energy can detection"""
    
    def __init__(self):
        """Initialize the Monster Analyzer"""
        print("=" * 60)
        print("🔥 Monster Analyzer - Real-time Flavor Detection")
        print("=" * 60)
        
        # Initialize models
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=config.POSE_CONFIDENCE_THRESHOLD,
            min_tracking_confidence=config.POSE_CONFIDENCE_THRESHOLD
        )
        
        # Object detector (may not load if model not present)
        self.object_detector = None
        if os.path.exists(config.OBJECT_DETECTION_MODEL_PATH):
            self.object_detector = ObjectDetector(config.OBJECT_DETECTION_MODEL_PATH)
        else:
            print("⚠ Object detection model not found. Using fallback detection.")
            print(f"  Expected path: {config.OBJECT_DETECTION_MODEL_PATH}")
        
        # Flavor classifier
        self.flavor_classifier = None
        if os.path.exists(config.FLAVOR_CLASSIFIER_PATH):
            self.flavor_classifier = FlavorClassifier(
                config.FLAVOR_CLASSIFIER_PATH,
                config.MONSTER_FLAVORS
            )
        else:
            print("⚠ Flavor classifier not found. Detection will be limited.")
            print(f"  Expected path: {config.FLAVOR_CLASSIFIER_PATH}")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(config.CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open webcam!")
        
        print(f"✓ Webcam initialized: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        
        # Initialize TTS if enabled
        self.tts_engine = None
        if config.ENABLE_TTS and TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                print("✓ Text-to-speech enabled")
            except:
                print("⚠ Text-to-speech initialization failed")
        
        # State tracking
        self.last_announced_flavor = None
        self.detection_count = {}
        
        # Initialize logging
        if config.ENABLE_LOGGING:
            self._initialize_logging()
        
        print("\n✓ Monster Analyzer ready!")
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'r' to reset detection stats")
        print("=" * 60 + "\n")
    
    def _initialize_logging(self):
        """Initialize CSV logging"""
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        # Create log file with headers if it doesn't exist
        if not os.path.exists(config.LOG_FILE):
            with open(config.LOG_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'flavor', 'confidence', 'hand_used'])
    
    def _log_detection(self, flavor: str, confidence: float, hand_used: str):
        """Log detection to CSV"""
        if not config.ENABLE_LOGGING:
            return
        
        with open(config.LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                flavor,
                f"{confidence:.4f}",
                hand_used
            ])
        
        # Update detection count
        if flavor in self.detection_count:
            self.detection_count[flavor] += 1
        else:
            self.detection_count[flavor] = 1
    
    def _announce_flavor(self, flavor: str):
        """Announce flavor using TTS"""
        if self.tts_engine and flavor != self.last_announced_flavor:
            try:
                self.tts_engine.say(f"You're drinking {flavor}")
                self.tts_engine.runAndWait()
                self.last_announced_flavor = flavor
            except:
                pass
    
    def _detect_cans_fallback(self, frame: np.ndarray):
        """
        Fallback can detection using simple color-based detection
        (for when object detection model is not available)
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for typical Monster can colors
        # This is a simplified approach - adjust ranges as needed
        color_ranges = [
            # Green (Original Monster)
            ((40, 40, 40), (80, 255, 255)),
            # Blue
            ((90, 50, 50), (130, 255, 255)),
            # White/Light colors
            ((0, 0, 180), (180, 30, 255)),
        ]
        
        detections = []
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, lower, upper)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Can-like aspect ratio (tall and narrow)
                    if 1.5 < aspect_ratio < 4.0:
                        detections.append({
                            'bbox': (x, y, x + w, y + h),
                            'class_id': 0,
                            'score': 0.8  # Dummy confidence
                        })
        
        return detections
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame and return annotated result
        
        Args:
            frame: Input frame from webcam (BGR)
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        display_frame = frame.copy()
        
        # Convert BGR to RGB for pose estimation
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run pose estimation
        pose_results = self.pose_estimator.detect_pose(rgb_frame)
        
        # Draw pose skeleton
        if config.SHOW_POSE_SKELETON and pose_results.pose_landmarks:
            draw_pose_skeleton(display_frame, pose_results, config.COLOR_SKELETON)
        
        # Get wrist positions
        wrist_positions = self.pose_estimator.get_wrist_positions(pose_results)
        
        # Draw wrist indicators
        for wrist_name, wrist_pos in wrist_positions.items():
            if wrist_pos:
                # Update positions based on actual frame size
                actual_wrist_pos = (
                    int(wrist_pos[0] * w / 640),
                    int(wrist_pos[1] * h / 480)
                )
                draw_wrist_indicator(display_frame, actual_wrist_pos, config.COLOR_BBOX_HAND)
        
        # Detect cans
        if self.object_detector:
            can_detections = self.object_detector.detect(frame, config.CONFIDENCE_THRESHOLD)
        else:
            can_detections = self._detect_cans_fallback(frame)
        
        # Process each detected can
        flavor_detected = None
        max_confidence = 0.0
        hand_used = None
        
        for detection in can_detections:
            bbox = detection['bbox']
            score = detection['score']
            
            # Draw bounding box
            if config.SHOW_BOUNDING_BOXES:
                draw_bounding_box(
                    display_frame, bbox, "Can", score,
                    config.COLOR_BBOX_CAN, 2
                )
            
            # Check if hand is near can
            for wrist_name, wrist_pos in wrist_positions.items():
                if wrist_pos:
                    # Scale wrist position to frame size
                    actual_wrist_pos = (
                        int(wrist_pos[0] * w / 640),
                        int(wrist_pos[1] * h / 480)
                    )
                    
                    if check_proximity(actual_wrist_pos, bbox, config.HAND_CAN_PROXIMITY_THRESHOLD):
                        # Hand is holding can - classify flavor
                        xmin, ymin, xmax, ymax = bbox
                        
                        # Ensure bounds are within frame
                        xmin = max(0, xmin)
                        ymin = max(0, ymin)
                        xmax = min(w, xmax)
                        ymax = min(h, ymax)
                        
                        # Crop can region
                        can_roi = frame[ymin:ymax, xmin:xmax]
                        
                        if can_roi.size > 0 and self.flavor_classifier:
                            # Classify flavor
                            flavor, confidence = self.flavor_classifier.classify(can_roi)
                            
                            if confidence > max_confidence:
                                flavor_detected = flavor
                                max_confidence = confidence
                                hand_used = wrist_name
                        
                        # Draw connection line
                        cv2.line(
                            display_frame,
                            actual_wrist_pos,
                            ((xmin + xmax) // 2, (ymin + ymax) // 2),
                            (0, 255, 0),
                            2
                        )
        
        # Display flavor prediction
        if flavor_detected:
            emoji = config.FLAVOR_EMOJIS.get(flavor_detected, "")
            draw_flavor_prediction(
                display_frame, flavor_detected, max_confidence,
                emoji, (20, 50), 1.0, config.COLOR_TEXT
            )
            
            if config.SHOW_CONFIDENCE_BAR:
                draw_confidence_bar(
                    display_frame, max_confidence, (20, 90), 300, 20,
                    config.COLOR_CONFIDENCE_BAR_BG, config.COLOR_CONFIDENCE_BAR_FG
                )
            
            # Log detection
            self._log_detection(flavor_detected, max_confidence, hand_used)
            
            # Announce if TTS enabled
            if config.ENABLE_TTS:
                self._announce_flavor(flavor_detected)
        else:
            # Show status message
            if len(can_detections) > 0:
                draw_status_message(display_frame, "Can detected - move hand closer", (20, 50))
            else:
                draw_status_message(display_frame, "No can detected", (20, 50), (128, 128, 128))
        
        return display_frame
    
    def run(self):
        """Main application loop"""
        print("Starting Monster Analyzer... Press 'q' to quit\n")
        
        # FPS calculation
        fps_start_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0
        
        while True:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame
            display_frame = self.process_frame(frame)
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                fps_end_time = time.time()
                current_fps = fps_frame_count / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                fps_frame_count = 0
            
            # Draw FPS
            draw_fps(display_frame, current_fps)
            
            # Display frame
            cv2.imshow('Monster Analyzer', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(config.DATA_DIR, f"screenshot_{timestamp}.jpg")
                cv2.imwrite(filename, display_frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('r'):
                # Reset stats
                self.detection_count = {}
                self.last_announced_flavor = None
                print("Detection stats reset")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\nCleaning up...")
        
        # Print detection statistics
        if self.detection_count:
            print("\n" + "=" * 60)
            print("Detection Statistics:")
            print("=" * 60)
            sorted_detections = sorted(self.detection_count.items(), 
                                      key=lambda x: x[1], reverse=True)
            for flavor, count in sorted_detections:
                print(f"  {flavor}: {count} detections")
            print("=" * 60)
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        self.pose_estimator.close()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        print("✓ Cleanup complete")


def main():
    """Entry point"""
    try:
        app = MonsterAnalyzer()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
