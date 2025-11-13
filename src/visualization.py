"""
Visualization utilities for Monster Analyzer
Handles drawing overlays on frames
"""
import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
import mediapipe as mp


def draw_pose_skeleton(image: np.ndarray, pose_results, color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Draw pose skeleton on image
    
    Args:
        image: Image to draw on (BGR format)
        pose_results: MediaPipe pose results
        color: Color for skeleton (BGR)
    """
    if pose_results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        
        mp_drawing.draw_landmarks(
            image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
        )


def draw_bounding_box(image: np.ndarray, bbox: Tuple[int, int, int, int], 
                      label: str = "", score: float = 0.0, 
                      color: Tuple[int, int, int] = (0, 255, 255),
                      thickness: int = 2):
    """
    Draw bounding box with label
    
    Args:
        image: Image to draw on (BGR format)
        bbox: (xmin, ymin, xmax, ymax)
        label: Text label
        score: Confidence score
        color: Box color (BGR)
        thickness: Line thickness
    """
    xmin, ymin, xmax, ymax = bbox
    
    # Draw rectangle
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)
    
    # Draw label background
    if label:
        label_text = f"{label}: {score:.2%}" if score > 0 else label
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
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
            label_text,
            (xmin + 5, ymin - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )


def draw_flavor_prediction(image: np.ndarray, flavor: str, confidence: float,
                          emoji: str = "", position: Tuple[int, int] = (20, 50),
                          font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255)):
    """
    Draw flavor prediction text with emoji
    
    Args:
        image: Image to draw on (BGR format)
        flavor: Flavor name
        confidence: Confidence score (0-1)
        emoji: Emoji character
        position: (x, y) position for text
        font_scale: Font size scale
        color: Text color (BGR)
    """
    text = f"Guess: {flavor} ({confidence:.0%}) {emoji}"
    
    # Draw text shadow for better visibility
    cv2.putText(
        image,
        text,
        (position[0] + 2, position[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        3
    )
    
    # Draw main text
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        2
    )


def draw_confidence_bar(image: np.ndarray, confidence: float, 
                       position: Tuple[int, int] = (20, 100),
                       bar_width: int = 300, bar_height: int = 20,
                       bg_color: Tuple[int, int, int] = (50, 50, 50),
                       fg_color: Tuple[int, int, int] = (0, 255, 0)):
    """
    Draw confidence bar
    
    Args:
        image: Image to draw on (BGR format)
        confidence: Confidence score (0-1)
        position: (x, y) position for bar
        bar_width: Width of bar in pixels
        bar_height: Height of bar in pixels
        bg_color: Background color (BGR)
        fg_color: Foreground color (BGR)
    """
    x, y = position
    
    # Draw background
    cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), bg_color, -1)
    
    # Draw filled portion
    fill_width = int(bar_width * confidence)
    cv2.rectangle(image, (x, y), (x + fill_width, y + bar_height), fg_color, -1)
    
    # Draw border
    cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)


def draw_wrist_indicator(image: np.ndarray, wrist_pos: Tuple[int, int], 
                         color: Tuple[int, int, int] = (255, 0, 255),
                         radius: int = 8):
    """
    Draw circle at wrist position
    
    Args:
        image: Image to draw on (BGR format)
        wrist_pos: (x, y) position
        color: Circle color (BGR)
        radius: Circle radius
    """
    if wrist_pos is not None:
        cv2.circle(image, wrist_pos, radius, color, -1)
        cv2.circle(image, wrist_pos, radius + 2, (255, 255, 255), 2)


def draw_fps(image: np.ndarray, fps: float, position: Tuple[int, int] = (20, 30)):
    """
    Draw FPS counter
    
    Args:
        image: Image to draw on (BGR format)
        fps: Frames per second
        position: (x, y) position
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


def draw_status_message(image: np.ndarray, message: str, 
                       position: Tuple[int, int] = (20, 150),
                       color: Tuple[int, int, int] = (255, 255, 0)):
    """
    Draw status message
    
    Args:
        image: Image to draw on (BGR format)
        message: Status message text
        position: (x, y) position
        color: Text color (BGR)
    """
    cv2.putText(
        image,
        message,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )


def create_split_view(frame1: np.ndarray, frame2: np.ndarray, 
                     labels: Tuple[str, str] = ("Original", "Detection")) -> np.ndarray:
    """
    Create side-by-side split view of two frames
    
    Args:
        frame1: First frame
        frame2: Second frame
        labels: Labels for each frame
        
    Returns:
        Combined frame
    """
    # Ensure frames are same size
    h1, w1 = frame1.shape[:2]
    h2, w2 = frame2.shape[:2]
    
    if h1 != h2 or w1 != w2:
        frame2 = cv2.resize(frame2, (w1, h1))
    
    # Stack horizontally
    combined = np.hstack([frame1, frame2])
    
    # Add labels
    cv2.putText(combined, labels[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2)
    cv2.putText(combined, labels[1], (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, (255, 255, 255), 2)
    
    return combined
