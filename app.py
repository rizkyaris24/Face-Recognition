import cv2
import numpy as np
import os
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
    
    # Open webcam (0 refers to the default webcam)
    cap = cv2.VideoCapture(0)
    
    # Check if webcam is accessible
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("\nFace detection started!")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
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
        window_title = "Enhanced Face Detection"
        features = []
        if use_age:
            features.append("Age")
        if use_gender:
            features.append("Gender")
        if features:
            window_title += f" with {' & '.join(features)} Prediction"
        
        cv2.imshow(window_title, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved as {filename}")
    
    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()
    print("Face detection stopped.")

if __name__ == "__main__":
    main()
