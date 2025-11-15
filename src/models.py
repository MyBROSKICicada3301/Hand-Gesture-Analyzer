"""
Model utilities for Hand Gesture Analyzer
Handles hand detection and finger counting using MediaPipe
"""
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional
import cv2


class HandDetector:
    """Handles hand detection and landmark tracking using MediaPipe Hands"""
    
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize hand detector
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_hands(self, image: np.ndarray) -> Optional[object]:
        """
        Detect hands in image
        
        Args:
            image: RGB image array
            
        Returns:
            Hand landmarks results or None
        """
        results = self.hands.process(image)
        return results
    
    def count_fingers(self, hand_landmarks, handedness) -> Tuple[int, List[bool]]:
        """
        Count number of fingers extended
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: Hand classification (Left/Right)
            
        Returns:
            Tuple of (finger_count, fingers_up_list)
            fingers_up_list: [thumb, index, middle, ring, pinky]
        """
        fingers_up = []
        
        # Landmark indices
        tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        
        # Get landmarks as a list
        landmarks = hand_landmarks.landmark
        
        # Thumb (different logic - check if tip is to the left/right of IP joint)
        # Need to check handedness for thumb
        is_right_hand = handedness.classification[0].label == "Right"
        
        if is_right_hand:
            # For right hand, thumb is up if tip x < IP x
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        else:
            # For left hand, thumb is up if tip x > IP x
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        
        # Other four fingers (check if tip is above PIP joint)
        for tip_id in tip_ids[1:]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        
        finger_count = sum(fingers_up)
        
        return finger_count, fingers_up
    
    def get_hand_info(self, results, frame_width: int, frame_height: int) -> List[Dict]:
        """
        Extract detailed hand information from detection results
        
        Args:
            results: MediaPipe hands results
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            List of hand info dictionaries containing landmarks, label, finger count, etc.
        """
        hands_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Get hand label (Left or Right)
                hand_label = handedness.classification[0].label
                
                # Count fingers
                finger_count, fingers_up = self.count_fingers(hand_landmarks, handedness)
                
                # Get bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * frame_width)
                x_max = int(max(x_coords) * frame_width)
                y_min = int(min(y_coords) * frame_height)
                y_max = int(max(y_coords) * frame_height)
                
                bbox = (x_min, y_min, x_max, y_max)
                
                # Get wrist position
                wrist = hand_landmarks.landmark[0]
                wrist_pos = (int(wrist.x * frame_width), int(wrist.y * frame_height))
                
                hand_info = {
                    'landmarks': hand_landmarks,
                    'label': hand_label,
                    'finger_count': finger_count,
                    'fingers_up': fingers_up,
                    'bbox': bbox,
                    'wrist_pos': wrist_pos
                }
                
                hands_info.append(hand_info)
        
        return hands_info
    
    def close(self):
        """Release resources"""
        self.hands.close()
