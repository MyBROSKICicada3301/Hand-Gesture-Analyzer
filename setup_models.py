"""
Model Downloader - Download pre-trained models for Monster Analyzer
"""
import os
import sys
import urllib.request

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def download_file(url: str, destination: str):
    """Download a file with progress indication"""
    print(f"Downloading: {url}")
    print(f"Destination: {destination}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        print(f"\rProgress: {percent:.1f}%", end='')
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print("\n✓ Download complete")
        return True
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        return False


def download_pose_model():
    """Download MediaPipe Pose Landmarker model"""
    # Note: MediaPipe handles model downloads automatically
    # This is just for reference
    print("MediaPipe Pose model will be downloaded automatically when needed")
    return True


def download_sample_object_detection_model():
    """
    Download a sample object detection model
    Note: You may need to train your own for best results
    """
    print("\n" + "=" * 60)
    print("Object Detection Model")
    print("=" * 60)
    print("For best results, you should train your own object detection model")
    print("to specifically detect Monster Energy cans.")
    print("\nOptions:")
    print("1. Use Google Teachable Machine (https://teachablemachine.withgoogle.com/)")
    print("2. Use Edge Impulse (https://www.edgeimpulse.com/)")
    print("3. Train a custom model using TensorFlow")
    print("\nFor now, the application will use fallback color-based detection.")
    return True


def setup_models():
    """Set up all required models"""
    print("=" * 60)
    print("🔧 Monster Analyzer Model Setup")
    print("=" * 60)
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    
    # Check pose model
    print("\n1. Pose Estimation Model:")
    download_pose_model()
    
    # Check object detection model
    print("\n2. Object Detection Model:")
    if not os.path.exists(config.OBJECT_DETECTION_MODEL_PATH):
        download_sample_object_detection_model()
    else:
        print(f"✓ Model already exists: {config.OBJECT_DETECTION_MODEL_PATH}")
    
    # Check flavor classifier
    print("\n3. Flavor Classifier Model:")
    if not os.path.exists(config.FLAVOR_CLASSIFIER_PATH):
        print(f"✗ Model not found: {config.FLAVOR_CLASSIFIER_PATH}")
        print("\nTo create the flavor classifier:")
        print("  1. Collect training images using: python collect_training_data.py")
        print("  2. Train the model using: python train_classifier.py")
    else:
        print(f"✓ Model exists: {config.FLAVOR_CLASSIFIER_PATH}")
    
    print("\n" + "=" * 60)
    print("✓ Model setup complete")
    print("=" * 60)


if __name__ == "__main__":
    setup_models()
