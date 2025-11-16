import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import mediapipe as mp


def draw_hand_landmarks(image: np.ndarray, hand_landmarks, hand_label: str, 
                       color: Tuple[int, int, int] = (0, 255, 0)):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    
    mp_drawing.draw_landmarks(
        image,
        hand_landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style()
    )


def draw_finger_count(image: np.ndarray, finger_count: int, hand_label: str,
                     position: Tuple[int, int], 
                     color: Tuple[int, int, int] = (255, 255, 255),
                     font_scale: float = 1.5):
    text = f"{hand_label} Hand: {finger_count} finger{'s' if finger_count != 1 else ''}"
    
    cv2.putText(
        image,
        text,
        (position[0] + 3, position[1] + 3),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        (0, 0, 0),
        4
    )
    
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        color,
        2
    )


def draw_finger_status(image: np.ndarray, fingers_up: List[bool], 
                      finger_names: List[str],
                      position: Tuple[int, int] = (20, 120),
                      font_scale: float = 0.6):
    y_offset = 0
    for i, (name, is_up) in enumerate(zip(finger_names, fingers_up)):
        status = "UP" if is_up else "DOWN"
        color = (0, 255, 0) if is_up else (100, 100, 100)
        text = f"{name}: {status}"
        
        cv2.putText(
            image,
            text,
            (position[0], position[1] + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2
        )
        y_offset += 25


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      label: str = "", 
                      color: Tuple[int, int, int] = (0, 255, 255),
                      thickness: int = 2):
    xmin, ymin, xmax, ymax = bbox
    
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    
    if label:
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            image,
            (xmin, ymin - text_height - 10),
            (xmin + text_width + 10, ymin),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (xmin + 5, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )


def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (10, 30)):
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


def draw_status_message(image: np.ndarray, message: str, 
                       position: Tuple[int, int] = (20, 450),
                       color: Tuple[int, int, int] = (255, 255, 0)):
    cv2.putText(
        image,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def draw_hand_summary(image: np.ndarray, hands_info: List[Dict], 
                     position: Tuple[int, int] = (10, 60)):
    if not hands_info:
        text = "No hands detected"
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 100, 100),
            2
        )
    else:
        y_offset = 0
        for i, hand_info in enumerate(hands_info):
            label = hand_info['label']
            finger_count = hand_info['finger_count']
            
            color = (0, 255, 0) if label == "Left" else (255, 0, 0)
            
            text = f"{label}: {finger_count} finger{'s' if finger_count != 1 else ''}"
            
            cv2.putText(
                image,
                text,
                (position[0] + 2, position[1] + y_offset + 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                (0, 0, 0),
                3
            )
            
            cv2.putText(
                image,
                text,
                (position[0], position[1] + y_offset),
                cv2.FONT_HERSHEY_DUPLEX,
                0.9,
                color,
                2
            )
            
            y_offset += 35
