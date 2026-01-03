import cv2
import numpy as np
import os
import time
from datetime import datetime

def load_face_model():
    """Load the pre-trained face detection model."""
    # Load face detection model
    face_proto = "models/opencv_face_detector.pbtxt"
    face_model = "models/opencv_face_detector_uint8.pb"
    
    if not os.path.exists(face_proto) or not os.path.exists(face_model):
        print("Face detection model files not found. Using Haar Cascade instead.")
        return None
    
    face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
    return face_net

def load_age_model():
    """Load the age prediction model if available."""
    age_proto = "models/age_deploy.prototxt"
    age_model = "models/age_net.caffemodel"
    
    if not os.path.exists(age_proto) or not os.path.exists(age_model):
        print("Age detection model files not found. Age prediction disabled.")
        return None
    
    try:
        age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
        return age_net
    except Exception as e:
        print(f"Error loading age model: {e}")
        return None

def load_gender_model():
    """Load the gender prediction model if available."""
    gender_proto = "models/gender_deploy.prototxt"
    gender_model = "models/gender_net.caffemodel"
    
    if not os.path.exists(gender_proto) or not os.path.exists(gender_model):
        print("Gender detection model files not found. Gender prediction disabled.")
        return None
    
    try:
        gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)
        return gender_net
    except Exception as e:
        print(f"Error loading gender model: {e}")
        return None

def detect_faces(net, frame, conf_threshold=0.7):
    """Detect faces in the frame using DNN model."""
    frame_height, frame_width = frame.shape[:2]
    
    # Create blob from frame
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    
    return face_boxes

def predict_age(face, age_net):
    """Predict age for a detected face."""
    # Define age ranges as used in the pre-trained model
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    
    # Model mean values for preprocessing
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    # Prepare face image for age prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = age_list[age_preds[0].argmax()]
    
    return age

def predict_gender(face, gender_net):
    """Predict gender for a detected face."""
    # Define gender categories as used in the pre-trained model
    gender_list = ['Male', 'Female']
    
    # Model mean values for preprocessing (same as age model)
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    
    # Prepare face image for gender prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    
    return gender

def detect_faces_haar(frame):
    """Fallback face detection using Haar Cascade."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    face_boxes = []
    for (x, y, w, h) in faces:
        face_boxes.append([x, y, x + w, y + h])
    
    return face_boxes

def estimate_face_size(face_box):
    """Estimate relative face size (distance approximation)."""
    x1, y1, x2, y2 = face_box
    width = x2 - x1
    height = y2 - y1
    size = (width + height) // 2
    
    if size > 200:
        return "Close"
    elif size > 100:
        return "Medium"
    else:
        return "Far"

def main():
    """
    Enhanced face detection with age and gender prediction.
    Press 'q' to quit, 's' to save screenshot.
    """
    print("Loading models...")
    
    # Load face detection model
    face_net = load_face_model()
    use_dnn = face_net is not None
    
    # Load age model (optional)
    age_net = load_age_model()
    use_age = age_net is not None
    
    # Load gender model (optional)
    gender_net = load_gender_model()
    use_gender = gender_net is not None
    
    print("\n🔧 Model Status:")
    if use_dnn:
        print("✅ Using DNN face detection model")
    else:
        print("⚠️ Using Haar Cascade face detection")
    
    if use_age:
        print("✅ Age detection enabled")
    else:
        print("❌ Age detection disabled")
        
    if use_gender:
        print("✅ Gender detection enabled")
    else:
        print("❌ Gender detection disabled")
    
    # Try to open webcam - try multiple camera indices
    cap = None
    camera_index = 0
    
    print("\n📹 Attempting to access camera...")
    for i in range(3):  # Try camera indices 0, 1, 2
        print(f"   Trying camera index {i}...")
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            # Test if we can actually read a frame
            test_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, test_frame = test_cap.read()
            if ret and test_frame is not None:
                cap = test_cap
                camera_index = i
                print(f"   ✅ Camera {i} is working!")
                break
            else:
                test_cap.release()
        else:
            if test_cap:
                test_cap.release()
    
    if cap is None:
        print("❌ Error: Could not open any camera.")
        print("   Please check:")
        print("   - Camera permissions are granted")
        print("   - Camera is not being used by another application")
        print("   - Camera is properly connected")
        return
    
    # Configure camera properties for better compatibility
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Give camera a moment to initialize
    print("   Initializing camera...")
    time.sleep(0.5)
    
    # Warm up the camera by reading a few frames
    for _ in range(5):
        cap.read()
    
    # Determine window title early
    window_title = "Enhanced Face Detection"
    features = []
    if use_age:
        features.append("Age")
    if use_gender:
        features.append("Gender")
    if features:
        window_title += f" with {' & '.join(features)} Prediction"
    
    # Create window BEFORE the loop (important for macOS)
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, 800, 600)
    cv2.moveWindow(window_title, 100, 100)
    
    print("\n✅ Face detection started!")
    print(f"🖼️  Camera window: '{window_title}'")
    print("Controls:")
    print("- Press 'q' in the camera window to quit")
    print("- Press 's' in the camera window to save screenshot")
    print("- Or press Ctrl+C in terminal to quit")
    print("- 💡 If window is behind other apps, check your Dock or use Cmd+Tab")
    
    frame_count = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    
    try:
        while True:
            # Capture frame from webcam
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n❌ Error: Could not read frame after {max_consecutive_errors} attempts.")
                    print("   Camera may have been disconnected or is in use.")
                    break
                time.sleep(0.1)  # Brief pause before retry
                continue
            
            consecutive_errors = 0  # Reset error counter on success
            
            frame_count += 1
        
            # Detect faces using appropriate method
            if use_dnn:
                face_boxes = detect_faces(face_net, frame)
            else:
                face_boxes = detect_faces_haar(frame)
            
            # Process each detected face
            face_count = len(face_boxes)
            for i, (x1, y1, x2, y2) in enumerate(face_boxes):
                # Draw rectangle around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add face number
                cv2.putText(frame, f"Face {i+1}", (x1, y1-40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Estimate distance/size
                distance = estimate_face_size([x1, y1, x2, y2])
                cv2.putText(frame, f"Distance: {distance}", (x1, y1-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Age and Gender prediction if available
                if use_age or use_gender:
                    face = frame[max(0, y1-20):min(y2+20, frame.shape[0]-1), 
                                max(0, x1-20):min(x2+20, frame.shape[1]-1)]
                    
                    if face.size > 0:
                        predictions = []
                        
                        # Age prediction
                        if use_age:
                            try:
                                age = predict_age(face, age_net)
                                predictions.append(f"Age: {age}")
                            except Exception as e:
                                print(f"Error predicting age: {e}")
                        
                        # Gender prediction
                        if use_gender:
                            try:
                                gender = predict_gender(face, gender_net)
                                predictions.append(f"Gender: {gender}")
                            except Exception as e:
                                print(f"Error predicting gender: {e}")
                        
                        # Display predictions
                        for idx, prediction in enumerate(predictions):
                            y_offset = y2 + 20 + (idx * 25)
                            cv2.putText(frame, prediction, (x1, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display face count and frame info
            cv2.putText(frame, f"Faces: {face_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow(window_title, frame)
            
            # Handle key presses and process window events (needed for display on macOS)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                print("\n🛑 Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved as {filename}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
    finally:
        # Release the webcam and close all windows
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("✅ Face detection stopped. Camera released.")

if __name__ == "__main__":
    main()
