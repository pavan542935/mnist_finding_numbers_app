"""
Test script to verify model predictions work correctly
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from ml_project import load_mnist_data, preprocess_data

def test_model_predictions():
    """Test the model on known training samples"""
    print("🧪 Testing Model Predictions")
    print("=" * 50)
    
    # Load model
    try:
        model = keras.models.load_model('mnist_model.h5')
        print("✅ Model loaded successfully")
    except:
        print("❌ Could not load model. Please train first with: python ml_project.py")
        return
    
    # Load local dataset
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Preprocess exactly like during training
    X_train_final, y_train_final, X_val, y_val, X_test_processed, y_test_categorical = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # Test on a few samples
    print("\n📊 Testing on 5 random samples:")
    print("-" * 30)
    
    # Test samples
    test_indices = np.random.choice(len(X_test_processed), 5, replace=False)
    
    for i, idx in enumerate(test_indices):
        sample_image = X_test_processed[idx:idx+1]  # Keep batch dimension
        true_label = np.argmax(y_test_categorical[idx])
        
        # Make prediction
        prediction = model.predict(sample_image, verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        print(f"Sample {i+1}:")
        print(f"  True label: {true_label}")
        print(f"  Predicted: {predicted_label}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Correct: {'✅' if true_label == predicted_label else '❌'}")
        print()
        
        # Show image
        plt.figure(figsize=(3, 3))
        plt.imshow(sample_image[0, :, :, 0], cmap='gray')
        plt.title(f'True: {true_label}, Predicted: {predicted_label} ({confidence:.1f}%)')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'test_sample_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # Overall accuracy test
    print("🎯 Testing overall accuracy on test set:")
    test_predictions = model.predict(X_test_processed[:1000], verbose=0)  # Test on first 1000 samples
    test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(y_test_categorical[:1000], axis=1))
    
    print(f"Accuracy on 1000 test samples: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

if __name__ == "__main__":
    test_model_predictions()