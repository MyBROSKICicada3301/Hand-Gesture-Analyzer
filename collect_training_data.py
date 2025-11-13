"""
Data Collection Helper - Capture training images from webcam
"""
import cv2
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def collect_training_images(flavor_name: str, num_images: int = 50):
    """
    Collect training images from webcam
    
    Args:
        flavor_name: Name of the Monster flavor (e.g., "Monster_Ultra_Blue")
        num_images: Number of images to collect
    """
    # Create directory for this flavor
    flavor_dir = os.path.join(config.DATA_DIR, 'training', flavor_name)
    os.makedirs(flavor_dir, exist_ok=True)
    
    # Initialize webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print("✗ Failed to open webcam")
        return
    
    print("=" * 60)
    print(f"📸 Collecting images for: {flavor_name}")
    print("=" * 60)
    print(f"Target: {num_images} images")
    print(f"Output directory: {flavor_dir}")
    print("\nInstructions:")
    print("  - Hold the Monster can in different positions")
    print("  - Vary the angle, distance, and lighting")
    print("  - Press SPACE to capture an image")
    print("  - Press 'q' to quit")
    print("=" * 60 + "\n")
    
    image_count = 0
    
    while image_count < num_images:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display frame with overlay
        display_frame = frame.copy()
        
        # Add instruction text
        text = f"Captured: {image_count}/{num_images} - Press SPACE to capture"
        cv2.putText(display_frame, text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add crosshair for centering
        h, w = display_frame.shape[:2]
        cv2.line(display_frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
        cv2.line(display_frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)
        
        cv2.imshow('Data Collection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):
            # Capture image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(flavor_dir, f"{flavor_name}_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            image_count += 1
            print(f"✓ Captured image {image_count}/{num_images}: {filename}")
            
            # Brief pause to prevent accidental multiple captures
            cv2.waitKey(200)
        
        elif key == ord('q'):
            print("\nQuitting...")
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"✓ Collection complete: {image_count} images saved")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect training images from webcam')
    parser.add_argument('flavor', type=str, 
                       help='Flavor name (e.g., Monster_Ultra_Blue)')
    parser.add_argument('--num-images', type=int, default=50,
                       help='Number of images to collect (default: 50)')
    
    args = parser.parse_args()
    
    # Replace spaces with underscores
    flavor_name = args.flavor.replace(' ', '_')
    
    collect_training_images(flavor_name, args.num_images)
