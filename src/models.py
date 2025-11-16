import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional
import cv2


class HandDetector:
    
    def __init__(self, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_hands(self, image: np.ndarray) -> Optional[object]:
        results = self.hands.process(image)
        return results
    
    def count_fingers(self, hand_landmarks, handedness) -> Tuple[int, List[bool]]:
        fingers_up = []
        
        tip_ids = [4, 8, 12, 16, 20]
        
        landmarks = hand_landmarks.landmark
        
        is_right_hand = handedness.classification[0].label == "Right"
        
        if is_right_hand:
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        else:
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        
        for tip_id in tip_ids[1:]:
            if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                fingers_up.append(True)
            else:
                fingers_up.append(False)
        
        finger_count = sum(fingers_up)
        
        return finger_count, fingers_up
    
    def get_hand_info(self, results, frame_width: int, frame_height: int) -> List[Dict]:
        hands_info = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                hand_label = handedness.classification[0].label
                
                finger_count, fingers_up = self.count_fingers(hand_landmarks, handedness)
                
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * frame_width)
                x_max = int(max(x_coords) * frame_width)
                y_min = int(min(y_coords) * frame_height)
                y_max = int(max(y_coords) * frame_height)
                
                bbox = (x_min, y_min, x_max, y_max)
                
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
        self.hands.close()
