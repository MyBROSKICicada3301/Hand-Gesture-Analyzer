"""
Model utilities for Monster Analyzer
Handles loading and inference for all ML models
"""
import tensorflow as tf
import numpy as np
import mediapipe as mp
from typing import Tuple, List, Dict, Optional
import cv2


class PoseEstimator:
    """Handles pose estimation using MediaPipe"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
    
    def detect_pose(self, image: np.ndarray) -> Optional[object]:
        """
        Detect pose in image
        
        Args:
            image: RGB image array
            
        Returns:
            Pose landmarks or None
        """
        results = self.pose.process(image)
        return results
    
    def get_wrist_positions(self, results) -> Dict[str, Tuple[int, int]]:
        """
        Extract wrist positions from pose results
        
        Args:
            results: MediaPipe pose results
            
        Returns:
            Dictionary with 'left_wrist' and 'right_wrist' positions
        """
        wrist_positions = {'left_wrist': None, 'right_wrist': None}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Left wrist (index 15)
            if len(landmarks) > 15:
                left_wrist = landmarks[15]
                wrist_positions['left_wrist'] = (
                    int(left_wrist.x * 640),  # Assume 640 width
                    int(left_wrist.y * 480)   # Assume 480 height
                )
            
            # Right wrist (index 16)
            if len(landmarks) > 16:
                right_wrist = landmarks[16]
                wrist_positions['right_wrist'] = (
                    int(right_wrist.x * 640),
                    int(right_wrist.y * 480)
                )
        
        return wrist_positions
    
    def close(self):
        """Release resources"""
        self.pose.close()


class ObjectDetector:
    """Handles object detection for cans using TFLite"""
    
    def __init__(self, model_path: str):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Object detection model loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load object detection model: {e}")
            print("  Note: You'll need to provide a TFLite object detection model")
    
    def detect(self, image: np.ndarray, confidence_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of detections with bounding boxes and scores
        """
        if self.interpreter is None:
            return []
        
        # Preprocess image
        input_shape = self.input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # Resize and normalize
        input_image = cv2.resize(image, (input_width, input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        
        # Convert to the expected dtype
        if self.input_details[0]['dtype'] == np.uint8:
            input_image = input_image.astype(np.uint8)
        else:
            input_image = input_image.astype(np.float32) / 255.0
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        
        # Get outputs (assumes standard TFLite object detection output format)
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]
        
        # Parse detections
        detections = []
        h, w = image.shape[:2]
        
        for i in range(len(scores)):
            if scores[i] > confidence_threshold:
                # Convert normalized coordinates to pixel coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                detection = {
                    'bbox': (
                        int(xmin * w),
                        int(ymin * h),
                        int(xmax * w),
                        int(ymax * h)
                    ),
                    'class_id': int(classes[i]),
                    'score': float(scores[i])
                }
                detections.append(detection)
        
        return detections


class FlavorClassifier:
    """Handles Monster flavor classification using TFLite"""
    
    def __init__(self, model_path: str, flavor_labels: List[str]):
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_path = model_path
        self.flavor_labels = flavor_labels
        self._load_model()
    
    def _load_model(self):
        """Load TFLite model"""
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"Flavor classifier loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load flavor classifier: {e}")
            print("  Note: You'll need to train and provide the Monster flavor classifier")
    
    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Classify Monster flavor from cropped can image
        
        Args:
            image: Cropped image of the can (BGR format)
            
        Returns:
            Tuple of (flavor_name, confidence)
        """
        if self.interpreter is None:
            return ("Model not loaded", 0.0)
        
        # Preprocess image
        input_shape = self.input_details[0]['shape']
        input_height, input_width = input_shape[1], input_shape[2]
        
        # Resize and normalize
        input_image = cv2.resize(image, (input_width, input_height))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.expand_dims(input_image, axis=0)
        
        # Convert to the expected dtype
        if self.input_details[0]['dtype'] == np.uint8:
            input_image = input_image.astype(np.uint8)
        else:
            input_image = input_image.astype(np.float32) / 255.0
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
        self.interpreter.invoke()
        
        # Get prediction
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Get top prediction
        predicted_idx = np.argmax(output_data)
        confidence = float(output_data[predicted_idx])
        
        # Map to flavor label
        if predicted_idx < len(self.flavor_labels):
            flavor = self.flavor_labels[predicted_idx]
        else:
            flavor = "Unknown"
        
        return flavor, confidence


def check_proximity(point1: Tuple[int, int], bbox: Tuple[int, int, int, int], 
                    threshold: int = 100) -> bool:
    """
    Check if a point is close to a bounding box
    
    Args:
        point1: (x, y) coordinates of the point (e.g., wrist)
        bbox: (xmin, ymin, xmax, ymax) bounding box
        threshold: Maximum distance in pixels
        
    Returns:
        True if point is within threshold distance of bbox
    """
    if point1 is None:
        return False
    
    px, py = point1
    xmin, ymin, xmax, ymax = bbox
    
    # Find closest point on bbox to the point
    closest_x = max(xmin, min(px, xmax))
    closest_y = max(ymin, min(py, ymax))
    
    # Calculate distance
    distance = np.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    return distance <= threshold
