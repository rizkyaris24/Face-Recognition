# Enhanced Face Detection with Python and OpenCV

This project implements real-time face detection and age estimation using Python and OpenCV with webcam input. It combines traditional Haar Cascade detection with modern deep neural network (DNN) models for enhanced accuracy and additional features.

## Features

- **🎯 Real-time face detection** using webcam
- **🧠 DNN-based face detection** with fallback to Haar Cascade
- **👶 Age estimation** - Predicts age ranges (0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100)
- **👤 Gender detection** - Classifies as Male/Female
- **📊 Face counting** and numbering
- **📏 Distance estimation** (Close/Medium/Far)
- **📸 Screenshot capture** functionality
- **🎮 Interactive controls** (keyboard shortcuts)
- **🔧 Robust error handling** and graceful degradation
- **🖥️ Cross-platform compatibility** (Windows, macOS, Linux)

## Requirements

- Python 3.6 or higher
- Webcam (built-in or external)

## Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd Face-Recognition
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Or install OpenCV directly:
   ```bash
   pip install opencv-python
   ```

## Usage

1. **Run the enhanced face detection application**
   ```bash
   python app.py
   ```

2. **Interactive Controls**
   - **'q'** - Quit the application
   - **'s'** - Save screenshot with timestamp
   
3. **Application Features**
   - **Face Detection**: Green rectangles around detected faces
   - **Face Numbering**: Each face is labeled (Face 1, Face 2, etc.)
   - **Face Counter**: Live count of detected faces displayed
   - **Distance Estimation**: Shows if faces are Close/Medium/Far
   - **Age Estimation**: Shows age ranges like (25-32), (15-20), etc.
   - **Gender Detection**: Shows Male/Female classification
   - **Frame Counter**: Displays current frame number
   - **Screenshots**: Saved with timestamp in current directory

## How It Works

### Detection Methods
The application uses a dual-approach system:

1. **DNN Face Detection** (Primary): Uses OpenCV's deep neural network models for more accurate detection
2. **Haar Cascade** (Fallback): Traditional cascade classifier as backup when DNN models aren't available

### Processing Pipeline
1. **Model Loading**: Loads DNN models (if available) and age estimation models
2. **Webcam Capture**: Captures real-time video frames from the default webcam
3. **Face Detection**: Applies DNN or Haar Cascade to identify faces in each frame
4. **Feature Extraction**: For each detected face:
   - Calculates face size for distance estimation
   - Extracts face region for age prediction (if enabled)
5. **Visual Overlay**: Draws rectangles, labels, and information on the frame
6. **Display**: Shows the enhanced video feed with all annotations

### Age Estimation (Optional)
When age models are available, the system:
- Extracts detected face regions
- Preprocesses faces for the age estimation model
- Predicts age ranges: (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100)
- Displays age predictions above detected faces

## Technical Details

### Parameters Used

- **scaleFactor=1.1**: Reduces image size by 10% at each scale
- **minNeighbors=5**: Minimum number of neighbor rectangles for face validation
- **minSize=(30, 30)**: Minimum face size in pixels

### Key Functions

- `cv2.CascadeClassifier()`: Loads the face detection model
- `cv2.VideoCapture()`: Accesses the webcam
- `cv2.cvtColor()`: Converts color frames to grayscale
- `detectMultiScale()`: Detects faces at multiple scales
- `cv2.rectangle()`: Draws rectangles around detected faces

## Age Detection Setup (Optional)

To enable age estimation, you need to download additional model files:

### Required Files
Place these files in the `models/` directory:
- `age_deploy.prototxt` - Age model architecture
- `age_net.caffemodel` - Pre-trained age model weights

### Download Sources
Due to file size limitations, age models need to be downloaded manually:
1. **age_deploy.prototxt**: [Download from GitHub](https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/age_deploy.prototxt)
2. **age_net.caffemodel**: Search for "OpenCV age detection caffemodel" or use models from research papers

### Verification
When models are properly installed, the application will show:
```
Age detection enabled
```

## Troubleshooting

### Common Issues

1. **"Error: Could not open webcam"**
   - Check if webcam is connected and not being used by another application
   - Try changing the camera index from `0` to `1` or `2`

2. **"Age detection model files not found"**
   - Download the required age model files (see Age Detection Setup)
   - Ensure files are placed in the `models/` directory
   - Verify file names match exactly

3. **No faces detected**
   - Ensure good lighting conditions
   - Position your face clearly in front of the camera
   - Try both DNN and Haar Cascade modes

4. **Poor performance**
   - Close other applications using the webcam
   - Reduce video resolution if needed
   - Use Haar Cascade mode if DNN is too slow

## Customization

You can modify the detection parameters in `face_detection.py`:

```python
faces = face_cascade.detectMultiScale(
    gray, 
    scaleFactor=1.1,    # Adjust for detection sensitivity
    minNeighbors=5,     # Adjust for false positive filtering
    minSize=(30, 30)    # Adjust minimum face size
)
```

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Based on the tutorial from [GeeksforGeeks](https://www.geeksforgeeks.org/python/face-detection-using-python-and-opencv-with-webcam/)
- Uses OpenCV's Haar Cascade Classifier
- Built with Python and OpenCV

## Contributing

Feel free to fork this project and submit pull requests for improvements!
