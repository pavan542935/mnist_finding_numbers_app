"""
Handwritten Digit Recognition using MNIST Dataset
Machine Learning Project

This project implements a neural network to classify handwritten digits (0-9)
using the famous MNIST dataset.
"""

# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Scikit-learn for metrics and preprocessing
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

# System and utility libraries
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette("husl")

print("All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Keras version: {keras.__version__}")

# ==========================================
# DATA LOADING AND EXPLORATION FUNCTIONS
# ==========================================

def load_mnist_data():
    """
    Load MNIST dataset from local CSV files
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test) - Training and testing data
    """
    print("Loading MNIST dataset from local files...")
    
    try:
        # Load training data
        print("📂 Loading training data...")
        train_path = "dataset/mnist_train.csv"
        if not os.path.exists(train_path):
            train_path = "mnist_train.csv"  # Fallback if in current directory
        
        train_data = pd.read_csv(train_path)
        print(f"   Training CSV loaded: {train_data.shape}")
        
        # Load test data
        print("📂 Loading test data...")
        test_path = "dataset/mnist_test.csv"
        if not os.path.exists(test_path):
            test_path = "mnist_test.csv"  # Fallback if in current directory
            
        test_data = pd.read_csv(test_path)
        print(f"   Test CSV loaded: {test_data.shape}")
        
        # Extract labels (first column) and features (remaining columns)
        y_train = train_data.iloc[:, 0].values
        X_train = train_data.iloc[:, 1:].values
        
        y_test = test_data.iloc[:, 0].values
        X_test = test_data.iloc[:, 1:].values
        
        # Reshape images from flat arrays to 28x28
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        
        # Convert to appropriate data types
        X_train = X_train.astype('uint8')
        X_test = X_test.astype('uint8')
        y_train = y_train.astype('int64')
        y_test = y_test.astype('int64')
        
        print("✅ Dataset loaded successfully from local files!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Classes: {np.unique(y_train)}")
        print(f"Pixel value range: [{X_train.min()}, {X_train.max()}]")
        
        return X_train, y_train, X_test, y_test
        
    except FileNotFoundError as e:
        print(f"❌ Error: Dataset files not found!")
        print(f"Expected files: dataset/mnist_train.csv and dataset/mnist_test.csv")
        print(f"Falling back to Keras dataset download...")
        
        # Fallback to original method
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        
        print("Dataset loaded from Keras successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
        return X_train, y_train, X_test, y_test
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print("Falling back to Keras dataset download...")
        
        # Fallback to original method
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        return X_train, y_train, X_test, y_test

def explore_dataset(X_train, y_train, X_test, y_test):
    """
    Explore and visualize the MNIST dataset
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
    """
    print("\n" + "="*50)
    print("DATASET EXPLORATION")
    print("="*50)
    
    # Basic information
    print(f"Image dimensions: {X_train.shape[1]} x {X_train.shape[2]} pixels")
    print(f"Pixel value range: {X_train.min()} to {X_train.max()}")
    print(f"Data type: {X_train.dtype}")
    
    # Class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution in training set:")
    for digit, count in zip(unique, counts):
        print(f"Digit {digit}: {count:,} samples ({count/len(y_train)*100:.1f}%)")
    
    # Visualize sample images
    plt.figure(figsize=(15, 6))
    for i in range(10):
        # Find first occurrence of each digit
        idx = np.where(y_train == i)[0][0]
        
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[idx], cmap='gray')
        plt.title(f'Digit: {i}', fontsize=12)
        plt.axis('off')
    
    plt.suptitle('Sample Images from Each Class', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Show random samples
    plt.figure(figsize=(15, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        random_idx = np.random.randint(0, len(X_train))
        plt.imshow(X_train[random_idx], cmap='gray')
        plt.title(f'Label: {y_train[random_idx]}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Random Sample of 25 Images', fontsize=16)
    plt.tight_layout()
    plt.show()

# ==========================================
# DATA PREPROCESSING FUNCTIONS
# ==========================================

def preprocess_data(X_train, y_train, X_test, y_test, normalize=True, reshape_for_cnn=True):
    """
    Preprocess the MNIST dataset for training
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels
        normalize (bool): Whether to normalize pixel values to [0,1]
        reshape_for_cnn (bool): Whether to reshape for CNN (add channel dimension)
    
    Returns:
        tuple: Preprocessed (X_train, y_train, X_test, y_test)
    """
    print("\n" + "="*50)
    print("DATA PREPROCESSING")
    print("="*50)
    
    # Store original shapes
    original_train_shape = X_train.shape
    original_test_shape = X_test.shape
    
    print(f"Original training data shape: {original_train_shape}")
    print(f"Original test data shape: {original_test_shape}")
    print(f"Original pixel value range: [{X_train.min()}, {X_train.max()}]")
    
    # Step 1: Normalize pixel values to [0, 1] range
    if normalize:
        print("\n1. Normalizing pixel values to [0, 1] range...")
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        print(f"   New pixel value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    
    # Step 2: Reshape data for CNN (add channel dimension)
    if reshape_for_cnn:
        print("\n2. Reshaping data for CNN (adding channel dimension)...")
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        print(f"   New training data shape: {X_train.shape}")
        print(f"   New test data shape: {X_test.shape}")
    
    # Step 3: Convert labels to categorical (one-hot encoding)
    print("\n3. Converting labels to categorical (one-hot encoding)...")
    print(f"   Original labels shape: {y_train.shape}")
    print(f"   Sample original labels: {y_train[:10]}")
    
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    print(f"   New labels shape: {y_train_categorical.shape}")
    print(f"   Sample one-hot labels: {y_train_categorical[0]}")
    
    # Step 4: Create validation set from training data
    print("\n4. Creating validation set (20% of training data)...")
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train_categorical, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_categorical.argmax(axis=1)
    )
    
    print(f"   Final training set: {X_train_final.shape}")
    print(f"   Validation set: {X_val.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # Summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"✓ Normalized pixel values: {normalize}")
    print(f"✓ Reshaped for CNN: {reshape_for_cnn}")
    print(f"✓ One-hot encoded labels: True")
    print(f"✓ Created validation set: True")
    print(f"\nFinal dataset sizes:")
    print(f"  Training: {X_train_final.shape[0]:,} samples")
    print(f"  Validation: {X_val.shape[0]:,} samples")
    print(f"  Test: {X_test.shape[0]:,} samples")
    
    return X_train_final, y_train_final, X_val, y_val, X_test, y_test_categorical

def visualize_preprocessing_effects(X_original, X_processed, sample_idx=0):
    """
    Visualize the effects of preprocessing on sample images
    
    Args:
        X_original: Original images
        X_processed: Preprocessed images
        sample_idx: Index of sample to visualize
    """
    print("\nVisualizing preprocessing effects...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Original image
    axes[0].imshow(X_original[sample_idx], cmap='gray')
    axes[0].set_title(f'Original Image\nShape: {X_original[sample_idx].shape}\nRange: [{X_original[sample_idx].min()}, {X_original[sample_idx].max()}]')
    axes[0].axis('off')
    
    # Processed image (remove channel dimension for display)
    processed_img = X_processed[sample_idx].squeeze() if len(X_processed[sample_idx].shape) == 3 else X_processed[sample_idx]
    axes[1].imshow(processed_img, cmap='gray')
    axes[1].set_title(f'Preprocessed Image\nShape: {X_processed[sample_idx].shape}\nRange: [{X_processed[sample_idx].min():.3f}, {X_processed[sample_idx].max():.3f}]')
    axes[1].axis('off')
    
    plt.suptitle('Before vs After Preprocessing', fontsize=14)
    plt.tight_layout()
    plt.show()

def analyze_class_distribution(y_train, y_val, y_test):
    """
    Analyze and visualize class distribution across train/val/test sets
    
    Args:
        y_train, y_val, y_test: One-hot encoded labels for each set
    """
    print("\nAnalyzing class distribution...")
    
    # Convert one-hot back to class labels for analysis
    y_train_labels = np.argmax(y_train, axis=1)
    y_val_labels = np.argmax(y_val, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Create distribution plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = [y_train_labels, y_val_labels, y_test_labels]
    titles = ['Training Set', 'Validation Set', 'Test Set']
    
    for i, (data, title) in enumerate(zip(datasets, titles)):
        unique, counts = np.unique(data, return_counts=True)
        axes[i].bar(unique, counts, alpha=0.7)
        axes[i].set_title(f'{title}\n({len(data):,} samples)')
        axes[i].set_xlabel('Digit Class')
        axes[i].set_ylabel('Number of Samples')
        axes[i].set_xticks(range(10))
        
        # Add count labels on bars
        for j, count in enumerate(counts):
            axes[i].text(j, count + 50, str(count), ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Class Distribution Across Datasets', fontsize=16)
    plt.tight_layout()
    plt.show()


# ==========================================
# MODEL BUILDING AND TRAINING FUNCTIONS
# ==========================================

def build_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Build a Convolutional Neural Network model for MNIST digit classification
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of output classes
    
    Returns:
        model: Compiled Keras model
    """
    print("Building CNN model...")
    
    model = models.Sequential([
        # First convolutional layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and add dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Model built successfully!")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=128):
    """
    Train the CNN model
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
        batch_size: Training batch size
    
    Returns:
        history: Training history
    """
    print(f"\nTraining model for {epochs} epochs...")
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.0001)
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def save_model(model, filepath='mnist_model.h5'):
    """
    Save the trained model
    
    Args:
        model: Trained Keras model
        filepath: Path to save the model
    """
    model.save(filepath)
    print(f"Model saved to {filepath}")

def load_trained_model(filepath='mnist_model.h5'):
    """
    Load a pre-trained model
    
    Args:
        filepath: Path to the saved model
    
    Returns:
        model: Loaded Keras model
    """
    try:
        model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
        return model
    except:
        print(f"Could not load model from {filepath}")
        return None


# ==========================================
# MAIN FUNCTION TO RUN THE CODE
# ==========================================

def main():
    """
    Main function to execute the MNIST project pipeline
    """
    print("\n" + "="*70)
    print("MNIST HANDWRITTEN DIGIT RECOGNITION PROJECT".center(70))
    print("="*70)
    
    # Step 1: Load the MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Step 2: Explore the dataset
    explore_dataset(X_train, y_train, X_test, y_test)
    
    # Store original data for visualization
    X_train_original = X_train.copy()
    
    # Step 3: Preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_train, y_train, X_test, y_test)
    
    # Step 4: Visualize preprocessing effects
    visualize_preprocessing_effects(X_train_original, X_train)
    
    # Step 5: Analyze class distribution
    analyze_class_distribution(y_train, y_val, y_test)
    
    # Step 6: Build and train the model
    model = build_cnn_model()
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=5)
    
    # Step 7: Save the trained model
    save_model(model, 'mnist_model.h5')
    
    # Step 8: Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!".center(70))
    print("="*70)
    
    return model, X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    # Execute the main function when script is run directly
    model, X_train, y_train, X_val, y_val, X_test, y_test = main()
