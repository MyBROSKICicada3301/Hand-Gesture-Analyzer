# Monster Analyzer - AI-Powered Flavor Detection

**Real-time Monster Energy can flavor detection using TensorFlow Lite, pose estimation, and computer vision**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red)

## Overview

Monster Analyzer is an interactive AI application that uses your laptop webcam to:

- Detect when you're holding a Monster Energy can using pose estimation
- Identify the specific flavor in real-time using a custom-trained classifier
- Display live visual overlays with predictions and confidence scores
- Optionally announce the flavor using text-to-speech

### Core Features

**Real-time Detection**

- Pose estimation to track hand positions
- Object detection for identifying cans
- Proximity detection between hands and cans

**Visual Overlays**

- Pose skeleton visualization
- Bounding boxes around detected cans
- Live flavor predictions with emojis
- Confidence bars showing prediction strength

**Custom ML Pipeline**

- TensorFlow Lite models for efficient inference
- Transfer learning for flavor classification
- Support for multiple Monster Energy flavors

**Optional Features**

- Text-to-speech announcements
- Detection logging and statistics
- Screenshot capture
- FPS monitoring

## Project Structure

```
Monster-Analyzer/
├── monster_analyzer.py          # Main application
├── train_classifier.py          # Model training script
├── collect_training_data.py     # Data collection helper
├── setup_models.py              # Model setup utility
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── src/
│   ├── models.py               # Model utilities (pose, detection, classification)
│   └── visualization.py        # Drawing and overlay functions
├── models/                      # TFLite models directory
│   ├── pose_landmark_lite.tflite
│   ├── detect.tflite           # Object detection model (optional)
│   └── monster_flavor_classifier.tflite
├── data/
│   └── training/               # Training images organized by flavor
│       ├── Monster_Energy_Original/
│       ├── Monster_Ultra_White/
│       ├── Monster_Ultra_Blue/
│       └── ...
└── logs/
    └── detections.csv          # Detection logs
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

### Installation

1. **Clone the repository**

   ```powershell
   git clone https://github.com/MyBROSKICicada3301/Monster-Analyzer.git
   cd Monster-Analyzer
   ```
2. **Install dependencies**

   ```powershell
   pip install -r requirements.txt
   ```
3. **Set up models**

   ```powershell
   python setup_models.py
   ```

### Running the Application

**Note:** The application will work with limited functionality without the flavor classifier. You'll need to train your own model for full flavor detection.

```powershell
python monster_analyzer.py
```

### Controls

- **`q`** - Quit the application
- **`s`** - Save screenshot
- **`r`** - Reset detection statistics

## Training Your Own Flavor Classifier

### Step 1: Collect Training Data

Use the data collection script to capture images from your webcam:

```powershell
# Collect images for each flavor you want to detect
python collect_training_data.py "Monster Ultra Blue" --num-images 50
python collect_training_data.py "Monster Ultra White" --num-images 50
python collect_training_data.py "Monster Energy Original" --num-images 50
```

**Tips for collecting good training data:**

- Take photos from different angles (front, side, angled)
- Vary the distance from the camera
- Use different lighting conditions
- Include partial views of the can
- Vary backgrounds (but keep the can clearly visible)
- Collect at least 50-100 images per flavor

### Step 2: Organize Your Data

Your training data should be organized as follows:

```
data/training/
├── Monster_Energy_Original/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
├── Monster_Ultra_White/
│   ├── image001.jpg
│   └── ...
└── Monster_Ultra_Blue/
    ├── image001.jpg
    └── ...
```

### Step 3: Train the Model

```powershell
python train_classifier.py --epochs 20 --batch-size 32
```

This will:

- Load your training images
- Apply data augmentation
- Train a MobileNetV2-based classifier
- Convert to TensorFlow Lite format
- Save the model to `models/monster_flavor_classifier.tflite`

**Training options:**

- `--data-dir` - Path to training data directory
- `--epochs` - Number of training epochs (default: 20)
- `--batch-size` - Batch size for training (default: 32)

## Configuration

Edit `config.py` to customize the application:

### Model Paths

```python
POSE_MODEL_PATH = 'models/pose_landmark_lite.tflite'
OBJECT_DETECTION_MODEL_PATH = 'models/detect.tflite'
FLAVOR_CLASSIFIER_PATH = 'models/monster_flavor_classifier.tflite'
```

### Webcam Settings

```python
CAMERA_INDEX = 0  # Change if you have multiple cameras
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
```

### Detection Settings

```python
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
HAND_CAN_PROXIMITY_THRESHOLD = 100  # Distance in pixels
```

### Monster Flavors

Add or remove flavors in the `MONSTER_FLAVORS` list:

```python
MONSTER_FLAVORS = [
    "Monster Energy Original",
    "Monster Ultra White",
    "Monster Ultra Blue",
    # Add your own...
]
```

### Optional Features

```python
ENABLE_TTS = False  # Text-to-speech
ENABLE_LOGGING = True  # Log detections to CSV
SHOW_POSE_SKELETON = True
SHOW_BOUNDING_BOXES = True
SHOW_CONFIDENCE_BAR = True
```

## 🎨 Customization

### Adding Custom Emojis

Edit the `FLAVOR_EMOJIS` dictionary in `config.py`:

```python
FLAVOR_EMOJIS = {
    "Monster Energy Original": "⚡",
    "Monster Ultra White": "❄️",
    "Monster Ultra Blue": "💙",
    "Your Custom Flavor": "🔥",
}
```

### Changing Colors

Colors are in BGR format for OpenCV:

```python
COLOR_SKELETON = (0, 255, 0)  # Green
COLOR_BBOX_CAN = (0, 255, 255)  # Yellow
COLOR_TEXT = (255, 255, 255)  # White
```

## 🔧 Advanced Usage

### Using Alternative Training Methods

#### Option 1: Google Teachable Machine

1. Visit [Teachable Machine](https://teachablemachine.withgoogle.com/)
2. Create an image project
3. Upload your training images
4. Train the model
5. Export as TensorFlow Lite
6. Place the model in `models/monster_flavor_classifier.tflite`

#### Option 2: Edge Impulse

1. Create a project on [Edge Impulse](https://www.edgeimpulse.com/)
2. Upload training images
3. Design and train your model
4. Export as TensorFlow Lite
5. Place the model in the models directory

### Custom Object Detection Model

For better can detection, train a custom object detection model:

1. Collect images of Monster cans in various settings
2. Annotate bounding boxes using tools like [LabelImg](https://github.com/heartexlabs/labelImg)
3. Train using TensorFlow Object Detection API
4. Convert to TFLite
5. Place in `models/detect.tflite`

## 📊 Detection Logging

When `ENABLE_LOGGING = True`, detections are logged to `logs/detections.csv`:

```csv
timestamp,flavor,confidence,hand_used
2025-11-13T10:30:45,Monster Ultra Blue,0.9234,right_wrist
2025-11-13T10:30:46,Monster Ultra Blue,0.9456,right_wrist
```

Analyze your logs to see:

- Which flavors you drink most often
- Detection accuracy over time
- Usage patterns

## 🐛 Troubleshooting

### Webcam not detected

- Check `config.CAMERA_INDEX` (try 0, 1, or 2)
- Ensure no other application is using the webcam
- Check webcam permissions

### Low FPS

- Reduce frame resolution in config
- Use a lighter model
- Close other applications
- Ensure you have a compatible GPU (optional but helps, cause more GPU=more 💪)

### Poor detection accuracy

- Collect more training data (100+ images per flavor)
- Ensure good lighting when collecting data
- Vary training data diversity
- Train for more epochs
- Adjust confidence thresholds

### Models not loading

- Run `python setup_models.py`
- Check model file paths in `config.py`
- Ensure TensorFlow is properly installed

## 🎯 Performance Tips

1. **Better Training Data** = Better Results

   - More images per flavor (100+ recommended)
   - Diverse angles and lighting
   - Clear, focused images
2. **Optimize for Speed**

   - Reduce input resolution
   - Use quantized models
   - Lower FPS target
3. **Improve Accuracy**

   - Fine-tune confidence thresholds
   - Adjust proximity threshold
   - Use better lighting

## 📝 TODO / Future Enhancements

- [ ] Multi-can detection (track multiple cans)
- [ ] Drinking detection (detect when taking a sip)
- [ ] Historical tracking dashboard
- [ ] Mobile app version
- [ ] Cloud-based model training
- [ ] Social media integration
- [ ] Calorie/caffeine tracking
- [ ] AR overlays with effects

## 🙏 Acknowledgments

- **TensorFlow** - Machine learning framework
- **MediaPipe** - Pose estimation
- **OpenCV** - Computer vision
- **Monster Energy** - For making awesome drinks worth detecting! 🥤

**Made with alot of monsters, by a monster addict⚡**
