import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
import os
from collections import deque, Counter

import mediapipe as mp

# --- MobileNetV2 Model Definition ---
def get_mobilenet_model(num_classes):
    model = models.mobilenet_v2(weights=None) # Our own trained weights
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# --- Main Application ---
def main():
    # 1. Load Model
    model_path = 'asl_mobilenet_v2.pth'
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Please run training first.")
        return

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    data_dir = 'data/train'
    if os.path.exists(data_dir):
        classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    else:
        classes = sorted(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                          'del', 'nothing', 'space'])
    
    NUM_CLASSES = len(classes)
    print(f"Classes: {classes}")

    print("Loading MobileNetV2 model...")
    model = get_mobilenet_model(num_classes=NUM_CLASSES)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return
        
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # 2. Setup Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # 3. Setup MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False, # Faster for video
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 4. Prediction Smoothing Setup
    prediction_history = deque(maxlen=10) 
    
    flip_image = True
    
    typed_text = ""
    last_action_time = 0
    COOLDOWN_SECONDS = 0.5
    
    print("Controls:")
    print("  'q': Quit")
    print("  'f': Toggle Image Flipping (Mirror)")
    print("  'SPACE': Append current prediction to text")
    print("  'BACKSPACE': Delete last character")
    print("  'c': Clear text")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if flip_image:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        
        # Define ROI (Region of Interest)
        roi_size = 300
        x1, y1 = int(w * 0.6), int(h * 0.1)
        x2, y2 = x1 + roi_size, y1 + roi_size
        
        # Ensure ROI is within frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        
        predicted_label = "Waiting..."
        conf_score = 0.0
        img_processed_display = np.zeros((roi_size, roi_size, 3), dtype=np.uint8) # Placeholder

        if roi.size != 0:
            # --- Preprocessing with MediaPipe (Background Removal) ---
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = hands.process(roi_rgb)
            
            img_final = None
            
            if results.multi_hand_landmarks:
                mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                landmarks = results.multi_hand_landmarks[0]
                h_roi, w_roi, _ = roi.shape
                points = []
                for lm in landmarks.landmark:
                    points.append((int(lm.x * w_roi), int(lm.y * h_roi)))
                
                hull = cv2.convexHull(np.array(points))
                cv2.fillConvexPoly(mask, hull, 255)
                
                kernel = np.ones((20, 20), np.uint8)
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                img_masked = cv2.bitwise_and(roi, roi, mask=mask)
                
                x, y, w_box, h_box = cv2.boundingRect(hull)
                pad = 20
                x = max(0, x - pad)
                y = max(0, y - pad)
                w_box = min(w_roi - x, w_box + 2*pad)
                h_box = min(h_roi - y, h_box + 2*pad)
                
                img_cropped = img_masked[y:y+h_box, x:x+w_box]
                
                if img_cropped.size != 0:
                    img_final = cv2.resize(img_cropped, (128, 128))
            

            if img_final is None:
                 img_final = cv2.resize(roi, (128, 128))

            # --- Standard Preprocessing ---
            # 1. Grayscale
            gray = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
            
            # 2. CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            equalized = clahe.apply(gray)
            
            # 3. Merge to 3 channels
            img_processed = cv2.merge([equalized, equalized, equalized])
            img_processed_display = cv2.resize(img_processed, (roi_size, roi_size))
            
            # 4. To Tensor
            img_tensor = transforms.ToTensor()(img_processed)
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
            
            # --- Prediction ---
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                raw_label = classes[predicted.item()]
                raw_conf = confidence.item()
                prediction_history.append(raw_label)
                most_common = Counter(prediction_history).most_common(1)
                predicted_label = most_common[0][0]
                conf_score = raw_conf

        # --- UI Construction ---
        color = (0, 255, 0)
        CONFIDENCE_THRESHOLD = 0.7
        
        if conf_score < CONFIDENCE_THRESHOLD:
            color = (0, 0, 255) # Red
            display_text = f"Unsure ({conf_score*100:.1f}%)"
        else:
            display_text = f"{predicted_label} ({conf_score*100:.1f}%)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, "Webcam Feed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        side_panel_width = roi_size + 40
        side_panel = np.zeros((h, side_panel_width, 3), dtype=np.uint8)

        y_offset = (h - roi_size) // 2
        x_offset = 20
        side_panel[y_offset:y_offset+roi_size, x_offset:x_offset+roi_size] = img_processed_display
        
        cv2.putText(side_panel, "Model Input", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(side_panel, "(Gray + CLAHE)", (x_offset, y_offset + roi_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        main_display = np.hstack((frame, side_panel))
        
        text_area_height = 100
        text_area = np.zeros((text_area_height, main_display.shape[1], 3), dtype=np.uint8)
        
        cv2.putText(text_area, f"Typed: {typed_text}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(text_area, "Controls: [Space] Add Letter | [Backspace] Delete | [C] Clear", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        final_ui = np.vstack((main_display, text_area))

        cv2.imshow('ASL Recognition System', final_ui)

        # --- Key Handling ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_image = not flip_image
        elif key == ord('c'):
            typed_text = ""
        elif key == 8: # Backspace
            typed_text = typed_text[:-1]
        elif key == 32: # Space
            # Logic to append character
            if conf_score > CONFIDENCE_THRESHOLD:
                if predicted_label == 'space':
                    typed_text += " "
                elif predicted_label == 'del':
                    typed_text = typed_text[:-1]
                elif predicted_label != 'nothing':
                    typed_text += predicted_label
            else:
                print("Confidence too low to type.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
