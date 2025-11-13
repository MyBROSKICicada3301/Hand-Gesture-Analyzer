"""
Training Script for Monster Flavor Classifier
This script helps you train a TensorFlow Lite model to classify the Monster Energy flavors
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config


def create_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    """
    Create a transfer learning model for Monster flavor classification
    
    Args:
        num_classes: Number of flavor classes
        input_shape: Input image shape
        
    Returns:
        Keras model
    """
    # Use MobileNetV2 as base (efficient for TFLite)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model


def prepare_dataset(data_dir: str, img_size: tuple = (224, 224), batch_size: int = 32):
    """
    Prepare training and validation datasets
    
    Args:
        data_dir: Directory containing training data organized in subdirectories by class
        img_size: Target image size
        batch_size: Batch size
        
    Returns:
        Tuple of (train_ds, val_ds, class_names)
    """
    # Data augmentation for training
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Load training data
    train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    # Load validation data
    val_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    class_names = list(train_ds.class_indices.keys())
    
    return train_ds, val_ds, class_names


def convert_to_tflite(model: keras.Model, output_path: str, quantize: bool = True):
    """
    Convert Keras model to TensorFlow Lite
    
    Args:
        model: Trained Keras model
        output_path: Output path for .tflite file
        quantize: Whether to apply quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply dynamic range quantization for smaller model size
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved: {output_path}")
    print(f"  Model size: {os.path.getsize(output_path) / 1024:.2f} KB")


def train_flavor_classifier(data_dir: str, epochs: int = 20, batch_size: int = 32):
    """
    Main training function
    
    Args:
        data_dir: Directory with training data (organized by class folders)
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 60)
    print("Monster Flavor Classifier Training")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("\nPlease organize your training data as follows:")
        print("  data/training/")
        print("    ├── Monster_Energy_Original/")
        print("    │   ├── image1.jpg")
        print("    │   ├── image2.jpg")
        print("    │   └── ...")
        print("    ├── Monster_Ultra_White/")
        print("    │   ├── image1.jpg")
        print("    │   └── ...")
        print("    └── ...")
        return
    
    # Prepare datasets
    print("\nLoading training data...")
    train_ds, val_ds, class_names = prepare_dataset(data_dir, batch_size=batch_size)
    
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")
    print(f"  Training samples: {train_ds.samples}")
    print(f"  Validation samples: {val_ds.samples}")
    
    # Create model
    print("\nBuilding model...")
    model = create_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model created")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODEL_DIR, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    print(f"\nTraining for {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Evaluate
    print("\nEvaluation Results:")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    
    # Save class names
    class_names_path = os.path.join(config.MODEL_DIR, 'class_names.json')
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"Class names saved: {class_names_path}")
    
    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    convert_to_tflite(model, config.FLAVOR_CLASSIFIER_PATH, quantize=True)
    
    # Fine-tuning (optional)
    print("\nFine-tuning model...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"  Validation Loss: {val_loss:.4f}")
    print(f"  Validation Accuracy: {val_acc:.4f}")
    
    # Convert fine-tuned model to TFLite
    print("\nConverting fine-tuned model to TensorFlow Lite...")
    convert_to_tflite(model, config.FLAVOR_CLASSIFIER_PATH, quantize=True)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nModel saved to: {config.FLAVOR_CLASSIFIER_PATH}")
    print(f"Class names saved to: {class_names_path}")
    print("\nYou can now use this model with monster_analyzer.py")


def create_sample_dataset_structure():
    """
    Create sample dataset structure with instructions
    """
    training_dir = os.path.join(config.DATA_DIR, 'training')
    os.makedirs(training_dir, exist_ok=True)
    
    # Create sample class directories
    sample_classes = [
        "Monster_Energy_Original",
        "Monster_Ultra_White",
        "Monster_Ultra_Blue",
        "Monster_Ultra_Paradise"
    ]
    
    for class_name in sample_classes:
        class_dir = os.path.join(training_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
    
    # Create README
    readme_path = os.path.join(training_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("Monster Flavor Classifier Training Data\n")
        f.write("=" * 60 + "\n\n")
        f.write("Instructions:\n")
        f.write("1. Collect 50-100 images of each Monster flavor you want to detect\n")
        f.write("2. Organize images into subdirectories by flavor name\n")
        f.write("3. Each subdirectory should contain only images of that flavor\n")
        f.write("4. Images can be .jpg, .jpeg, or .png\n")
        f.write("5. Vary lighting, angles, and backgrounds for better results\n\n")
        f.write("Tips for collecting images:\n")
        f.write("- Take photos from different angles\n")
        f.write("- Use different lighting conditions\n")
        f.write("- Include partial views of the can\n")
        f.write("- Vary the distance from the camera\n")
        f.write("- Include different backgrounds\n\n")
        f.write("You can also use:\n")
        f.write("- Google Images (with proper licensing)\n")
        f.write("- Your own webcam to capture training images\n")
            f.write("- Use Data augmentation will be applied automatically during training\n")
    
    print(f"Sample dataset structure created: {training_dir}")
    print(f"  README: {readme_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Monster Flavor Classifier')
    parser.add_argument('--data-dir', type=str, 
                       default=os.path.join(config.DATA_DIR, 'training'),
                       help='Directory containing training data')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--create-structure', action='store_true',
                       help='Create sample dataset structure')
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_sample_dataset_structure()
    else:
        train_flavor_classifier(args.data_dir, args.epochs, args.batch_size)
