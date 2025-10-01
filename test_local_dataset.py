"""
Quick test script to verify local MNIST dataset loading
"""

import matplotlib.pyplot as plt
import numpy as np
from ml_project import load_mnist_data

def test_local_dataset():
    print("🧪 Testing Local MNIST Dataset Loading")
    print("=" * 50)
    
    # Load the dataset
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Display sample images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('Sample Images from Local MNIST Dataset', fontsize=16)
    
    for i in range(10):
        row = i // 5
        col = i % 5
        
        # Find first occurrence of digit i
        idx = np.where(y_train == i)[0][0]
        
        axes[row, col].imshow(X_train[idx], cmap='gray')
        axes[row, col].set_title(f'Digit: {i}', fontsize=12)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('local_dataset_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Local dataset test completed!")
    print(f"📊 Dataset Statistics:")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Image shape: {X_train[0].shape}")
    print(f"   Pixel range: [{X_train.min()}, {X_train.max()}]")
    print(f"   Classes: {sorted(np.unique(y_train))}")
    
    # Show class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n📈 Class Distribution (Training Set):")
    for digit, count in zip(unique, counts):
        print(f"   Digit {digit}: {count:,} samples ({count/len(y_train)*100:.1f}%)")

if __name__ == "__main__":
    test_local_dataset()