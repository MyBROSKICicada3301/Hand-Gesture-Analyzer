"""
Quick Start Script - Get started with Monster Analyzer
"""
import os
import sys
import subprocess


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("✗ Python 3.8 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def install_dependencies():
    """Install required packages"""
    print_header("Installing Dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("✗ Failed to install dependencies")
        return False


def setup_directories():
    """Create necessary directories"""
    print_header("Setting Up Directories")
    
    directories = [
        "models",
        "data",
        "data/training",
        "logs",
        "src"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}/")
    
    return True


def check_webcam():
    """Check if webcam is accessible"""
    print_header("Checking Webcam")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Webcam not accessible")
            print("  Please check:")
            print("  - Webcam is connected")
            print("  - No other application is using the webcam")
            print("  - Webcam permissions are granted")
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print("✓ Webcam is accessible")
            return True
        else:
            print("✗ Failed to capture frame from webcam")
            return False
    except ImportError:
        print("⚠ OpenCV not installed yet - will check after installation")
        return True
    except Exception as e:
        print(f"⚠ Error checking webcam: {e}")
        return True


def print_next_steps():
    """Print instructions for next steps"""
    print_header("🎉 Setup Complete!")
    
    print("\nNext Steps:\n")
    
    print("1. Collect Training Data")
    print("   python collect_training_data.py \"Monster Ultra Blue\" --num-images 50")
    print("   (Repeat for each flavor you want to detect)\n")
    
    print("2. Train the Flavor Classifier")
    print("   python train_classifier.py --epochs 20\n")
    
    print("3. Run Monster Analyzer")
    print("   python monster_analyzer.py\n")
    
    print("Alternatively, you can run with limited functionality:")
    print("   python monster_analyzer.py")
    print("   (Will work without flavor classifier, using fallback detection)\n")
    
    print("For more information, see README.md")
    print("=" * 60)


def main():
    """Main setup function"""
    print_header("🔥 Monster Analyzer - Quick Start Setup")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Setting up directories", setup_directories),
        ("Checking webcam", check_webcam),
        ("Installing dependencies", install_dependencies),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n✗ Setup failed at: {step_name}")
            print("Please resolve the issue and run this script again.")
            sys.exit(1)
    
    print_next_steps()


if __name__ == "__main__":
    main()
