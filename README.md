# 🔢 MNIST Handwritten Digit Recognition Web App

A web application that recognizes handwritten digits (0-9) using a Convolutional Neural Network trained on the MNIST dataset.

## 🚀 How to Run the App

### **Option 1: Quick Launch (Recommended)**
```bash
python launch_app.py
```

### **Option 2: Using Batch File (Windows)**
Double-click `run_mnist_app.bat` or run:
```cmd
run_mnist_app.bat
```

### **Option 3: Using PowerShell**
```powershell
.\run_mnist_app.ps1
```

### **Option 4: Manual Method**
1. Open terminal/command prompt
2. Navigate to project folder: `cd "D:\warp\ml project"`
3. Activate environment: `mnist_env\Scripts\activate`
4. Run app: `streamlit run streamlit_app.py`

## 📁 Project Structure

```
D:\warp\ml project\
├── 📂 dataset\             # Local MNIST dataset (CSV files)
│   ├── mnist_train.csv     # Training data (60,000 samples)
│   └── mnist_test.csv      # Test data (10,000 samples)
├── 📂 mnist_env\           # Virtual environment
├── 🐍 ml_project.py        # Main ML pipeline & training
├── 🌐 streamlit_app.py     # Web interface
├── 🚀 launch_app.py        # Simple launcher (recommended)
├── 📄 run_mnist_app.bat    # Windows batch launcher
├── 📄 run_mnist_app.ps1    # PowerShell launcher
├── 🔗 create_shortcut.py   # Desktop shortcut creator
├── 🧪 test_local_dataset.py # Local dataset test script
├── 🤖 mnist_model.h5       # Trained model (99.06% accuracy)
└── 📖 README.md           # This file
```

## 🎯 Usage Instructions

1. **Launch the app** using any of the methods above
2. **Open your browser** to `http://localhost:8501` (opens automatically)
3. **Upload an image** of a handwritten digit (0-9)
4. **View the prediction** with confidence scores
5. **See the interactive chart** showing probabilities for all digits

## 🔧 First-Time Setup (Already Done!)

✅ Virtual environment created  
✅ All packages installed  
✅ Model trained (99.06% accuracy)  
✅ Web interface ready  

## 🎮 Tips for Best Results

- Use **clear, handwritten digits**
- **White digits on dark background** work best
- **Single digit per image**
- Common formats: PNG, JPG, JPEG, BMP, GIF

## 🛠️ Troubleshooting

### App won't start?
- Make sure you're in the project directory
- Try: `python launch_app.py` (most reliable method)

### Model not found error?
- Run: `python ml_project.py` to retrain the model

### Import errors?
- Activate environment: `mnist_env\Scripts\activate`
- Reinstall packages: `pip install -r requirements.txt`

### Browser doesn't open?
- Manually go to: `http://localhost:8501`

## 🔄 Creating a Desktop Shortcut

1. Install requirements: `pip install pywin32 winshell`
2. Run: `python create_shortcut.py`
3. Find shortcut on desktop: "MNIST Digit Recognition"

## 📊 Model Performance

- **Accuracy:** 99.06% on test data
- **Architecture:** Convolutional Neural Network (CNN)
- **Training Data:** 48,000 samples
- **Validation Data:** 12,000 samples  
- **Test Data:** 10,000 samples

## 💡 Features

✅ **Drag & drop image upload**  
✅ **Real-time prediction**  
✅ **Confidence scores for all digits**  
✅ **Interactive visualization**  
✅ **Image preprocessing display**  
✅ **Professional web interface**  
✅ **Automatic model loading**  

---

**Built with ❤️ using TensorFlow, Keras, and Streamlit**