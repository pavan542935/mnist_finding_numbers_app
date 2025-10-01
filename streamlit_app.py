"""
Streamlit Web Interface for MNIST Handwritten Digit Recognition
Upload an image and get digit prediction with confidence scores
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-section {
        padding: 2rem;
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        text-align: center;
        background-color: #f0f2f6;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #e8f4f8;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Load model function
@st.cache_resource
def load_model():
    """Load the trained MNIST model"""
    model_path = "mnist_model.h5"
    if os.path.exists(model_path):
        try:
            model = keras.models.load_model(model_path)
            return model, True
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, False
    else:
        return None, False

# Load local dataset for preview
@st.cache_data
def load_local_dataset_preview(num_samples=20):
    """Load a small sample from the local MNIST dataset for preview"""
    try:
        import pandas as pd
        
        # Load a small sample from training data
        train_path = "dataset/mnist_train.csv"
        if os.path.exists(train_path):
            # Read only first few rows for preview (faster loading)
            sample_data = pd.read_csv(train_path, nrows=1000)
            
            # Extract labels and features
            y_sample = sample_data.iloc[:, 0].values
            X_sample = sample_data.iloc[:, 1:].values
            
            # Reshape to 28x28 images
            X_sample = X_sample.reshape(-1, 28, 28)
            
            # Get diverse samples (one from each digit class)
            preview_images = []
            preview_labels = []
            
            for digit in range(10):
                digit_indices = np.where(y_sample == digit)[0]
                if len(digit_indices) > 0:
                    idx = digit_indices[0]
                    preview_images.append(X_sample[idx])
                    preview_labels.append(y_sample[idx])
            
            # Add some random samples
            random_indices = np.random.choice(len(X_sample), 
                                            min(num_samples - len(preview_images), len(X_sample)), 
                                            replace=False)
            
            for idx in random_indices:
                preview_images.append(X_sample[idx])
                preview_labels.append(y_sample[idx])
            
            return np.array(preview_images), np.array(preview_labels), True
        else:
            return None, None, False
            
    except Exception as e:
        st.error(f"Error loading dataset preview: {e}")
        return None, None, False

def display_dataset_preview():
    """Display a grid of sample images from the local dataset"""
    st.subheader("📂 Local Dataset Preview")
    
    preview_images, preview_labels, success = load_local_dataset_preview()
    
    if success and preview_images is not None:
        st.success(f"✅ Loaded {len(preview_images)} sample images from local dataset")
        
        # Create columns for the image grid
        cols_per_row = 5
        num_rows = (len(preview_images) + cols_per_row - 1) // cols_per_row
        
        for row in range(min(num_rows, 4)):  # Show max 4 rows
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                img_idx = row * cols_per_row + col_idx
                
                if img_idx < len(preview_images):
                    with cols[col_idx]:
                        # Create matplotlib figure for each image
                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(preview_images[img_idx], cmap='gray')
                        ax.set_title(f'Label: {preview_labels[img_idx]}', fontsize=10)
                        ax.axis('off')
                        
                        st.pyplot(fig)
                        plt.close(fig)
        
        # Show dataset statistics
        with st.expander("📊 Dataset Statistics"):
            unique_labels, counts = np.unique(preview_labels, return_counts=True)
            stats_df = {
                'Digit': unique_labels,
                'Count in Preview': counts,
                'Percentage': (counts / len(preview_labels) * 100).round(1)
            }
            st.dataframe(stats_df)
            
    else:
        st.warning("⚠️ Could not load dataset preview. Make sure dataset/mnist_train.csv exists.")

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction
    
    Args:
        image: PIL Image object
        
    Returns:
        processed_image: Preprocessed numpy array ready for prediction
    """
    # Convert to grayscale
    image = ImageOps.grayscale(image)
    
    # Resize to 28x28 pixels
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Check if image needs inversion (if background is bright, digit should be dark)
    # Calculate mean brightness to determine if inversion is needed
    mean_brightness = np.mean(image_array)
    
    if mean_brightness > 127:  # Bright background, dark digit -> invert to match MNIST format
        image_array = 255 - image_array
    
    # IMPORTANT: Make sure the image matches MNIST format exactly
    # MNIST digits are WHITE on BLACK background, values 0-255
    
    # Normalize pixel values to [0, 1] - EXACTLY like training data
    image_array = image_array.astype('float32') / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    image_array = image_array.reshape(1, 28, 28, 1)
    
    return image_array

def create_prediction_chart(predictions):
    """Create a bar chart of prediction probabilities"""
    digits = list(range(10))
    probabilities = predictions[0] * 100  # Convert to percentages
    
    # Create plotly bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=probabilities,
            marker_color=['#ff7f0e' if i == np.argmax(probabilities) else '#1f77b4' for i in range(10)],
            text=[f'{prob:.1f}%' for prob in probabilities],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Confidence for Each Digit",
        xaxis_title="Digit",
        yaxis_title="Confidence (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def debug_preprocessing_steps(original_image, processed_array):
    """
    Debug function to show all preprocessing steps
    """
    # Convert back for visualization
    processed_display = processed_array[0, :, :, 0]  # Remove batch and channel dims
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image\nMean: {np.mean(original_image):.1f}')
    axes[0].axis('off')
    
    # Resized to 28x28
    resized = original_image.resize((28, 28), Image.Resampling.LANCZOS)
    resized_array = np.array(resized)
    axes[1].imshow(resized_array, cmap='gray')
    axes[1].set_title(f'Resized 28x28\nMean: {np.mean(resized_array):.1f}')
    axes[1].axis('off')
    
    # Final processed
    axes[2].imshow(processed_display, cmap='gray')
    axes[2].set_title(f'Processed (Model Input)\nMean: {np.mean(processed_display):.3f}\nRange: [{processed_display.min():.3f}, {processed_display.max():.3f}]')
    axes[2].axis('off')
    
    plt.suptitle('Preprocessing Steps Visualization', fontsize=14)
    plt.tight_layout()
    
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Recognition</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("📋 About This App")
    st.sidebar.info(
        """
        This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset 
        to recognize handwritten digits (0-9).
        
        **How to use:**
        1. Upload an image of a handwritten digit
        2. The model will predict what digit it is
        3. View confidence scores for all digits
        
        **Tips for best results:**
        - Use clear, handwritten digits
        - White digits on dark background work best
        - Single digit per image
        """
    )
    
    # Dataset preview option in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📂 Dataset Options")
    show_dataset_preview = st.sidebar.checkbox(
        "Show Dataset Preview",
        help="Display sample images from the local MNIST training dataset"
    )
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("⚠️ Model not found! Please train the model first by running:")
        st.code("python ml_project.py")
        st.info("This will train the model and save it as 'mnist_model.h5'")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Show dataset preview if requested
    if show_dataset_preview:
        st.markdown("---")
        display_dataset_preview()
        st.markdown("---")
    
    # Initialize the image variable
    uploaded_file = None
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Upload an image containing a single handwritten digit"
        )
        
        # Sample images section
        st.subheader("📋 Or Try Sample Images")
        
        # Load sample images from local dataset
        sample_images, sample_labels, sample_success = load_local_dataset_preview(num_samples=10)
        
        if sample_success and sample_images is not None:
            sample_options = ["None"] + [f"Sample {i+1} (Digit {sample_labels[i]})" for i in range(min(10, len(sample_images)))]
            
            sample_choice = st.selectbox(
                "Choose a sample from local dataset:",
                sample_options,
                help="Select a sample image from your local MNIST dataset to test the model"
            )
            
            if sample_choice != "None":
                # Extract sample index
                sample_idx = int(sample_choice.split()[1]) - 1
                selected_image = sample_images[sample_idx]
                selected_label = sample_labels[sample_idx]
                
                # Display the selected sample
                st.write(f"**Selected Sample:** Digit {selected_label}")
                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(selected_image, cmap='gray')
                ax.set_title(f'Original Label: {selected_label}')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
                
                # Create a PIL image for processing
                # Convert to PIL format (scale to 0-255 and convert to uint8)
                pil_image = Image.fromarray((selected_image).astype('uint8'), mode='L')
                
                # Override uploaded_file with the sample
                uploaded_file = pil_image
        else:
            st.info("📝 No local dataset samples available. Make sure dataset/mnist_train.csv exists.")
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Check if we have either an uploaded file or a sample image
        if uploaded_file is not None:
            try:
                # Initialize image variable
                image = None
                
                # Handle both uploaded files and sample images
                if hasattr(uploaded_file, 'read'):  # It's an uploaded file
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                elif isinstance(uploaded_file, Image.Image):  # It's a PIL Image from sample selection
                    image = uploaded_file
                    st.image(image, caption="Selected Sample Image", use_column_width=True)
                else:
                    st.error("Invalid image format")
                    return
                
                # Ensure we have a valid image
                if image is None:
                    st.error("Could not load image")
                    return
                
                # Preprocess image
                processed_image = preprocess_image(image)
                
                # Display preprocessing steps for debugging
                st.subheader("🔄 Preprocessing Steps")
                debug_fig = debug_preprocessing_steps(image, processed_image)
                st.pyplot(debug_fig)
                plt.close(debug_fig)
                
                # Show preprocessing info
                with st.expander("🔍 Preprocessing Details"):
                    original_array = np.array(image.resize((28, 28)))
                    final_array = processed_image[0, :, :, 0]
                    
                    st.write(f"**Original image stats:**")
                    st.write(f"- Mean brightness: {np.mean(original_array):.1f}")
                    st.write(f"- Range: [{original_array.min()}, {original_array.max()}]")
                    
                    st.write(f"**Processed image stats:**")
                    st.write(f"- Mean: {np.mean(final_array):.3f}")
                    st.write(f"- Range: [{final_array.min():.3f}, {final_array.max():.3f}]")
                    st.write(f"- Shape: {processed_image.shape}")
                    
                    # Check if inversion happened
                    if np.mean(original_array) > 127:
                        st.info("🔄 Image was inverted to match MNIST format (white digits on black background)")
                    else:
                        st.info("✨ Image was kept as-is (already dark background, bright digits)")
                
                # Make prediction
                with st.spinner('Making prediction...'):
                    predictions = model.predict(processed_image, verbose=0)
                
                # Get predicted digit and confidence
                predicted_digit = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                # Display main prediction
                st.markdown(f"""
                <div class="prediction-box">
                    <h2 style="color: #1f77b4; margin: 0;">Predicted Digit: {predicted_digit}</h2>
                    <h3 style="color: #28a745; margin: 0;">Confidence: {confidence:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Display confidence chart
                st.subheader("📊 Confidence Scores")
                chart = create_prediction_chart(predictions)
                st.plotly_chart(chart, use_container_width=True)
                
                # Display raw predictions
                with st.expander("🔍 View Raw Predictions"):
                    pred_df = {
                        'Digit': list(range(10)),
                        'Probability': predictions[0],
                        'Percentage': predictions[0] * 100
                    }
                    st.dataframe(pred_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Please make sure you uploaded a valid image file.")
                st.info("Supported formats: PNG, JPG, JPEG, BMP, GIF")
                
                # Show detailed error for debugging
                with st.expander("🐛 Debug Info"):
                    st.text(f"Error type: {type(e).__name__}")
                    st.text(f"Error details: {str(e)}")
                    st.text(f"Uploaded file type: {type(uploaded_file)}")
                    if hasattr(uploaded_file, 'name'):
                        st.text(f"File name: {uploaded_file.name}")
        
        else:
            st.info("👆 Upload an image above to see the prediction")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            Built with ❤️ using Streamlit and TensorFlow<br>
            MNIST Handwritten Digit Recognition Project
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()