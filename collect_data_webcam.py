import cv2
import os
import time

# Configuration
DATA_DIR = 'data/my_data'
ROI_SIZE = 300

def create_dirs(label):
    path = os.path.join(DATA_DIR, label)
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    current_label = 'A'
    save_dir = create_dirs(current_label)
    count = len(os.listdir(save_dir))
    
    auto_save = False
    last_save_time = 0
    save_interval = 0.1 # Seconds

    print(f"Current Class: {current_label}")
    print("Controls:")
    print("  'a'-'z': Change class to A-Z")
    print("  '0': Change class to 'nothing'")
    print("  '1': Change class to 'space'")
    print("  '2': Change class to 'del'")
    print("  's': SAVE single image")
    print("  'c': Toggle AUTO SAVE (Hold pose and move hand slightly)")
    print("  'q': Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # ROI
        x1, y1 = int(w * 0.6), int(h * 0.1)
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE
        
        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Auto Save Logic
        current_time = time.time()
        if auto_save and (current_time - last_save_time > save_interval):
            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                timestamp = int(current_time * 1000)
                filename = f"{current_label}_{timestamp}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, roi)
                count += 1
                last_save_time = current_time
                print(f"Auto-Saved {filepath}")

        # Info
        cv2.putText(frame, f"Class: {current_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Saved: {count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if auto_save:
            cv2.putText(frame, "AUTO SAVE ON", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to save, 'c' for auto", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        cv2.imshow('Data Collection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            auto_save = not auto_save
        elif key == ord('s'):
            # Save ROI
            roi = frame[y1:y2, x1:x2]
            if roi.size != 0:
                timestamp = int(time.time() * 1000)
                filename = f"{current_label}_{timestamp}.jpg"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, roi)
                count += 1
                print(f"Saved {filepath}")
        elif ord('a') <= key <= ord('z'):
            current_label = chr(key).upper()
            save_dir = create_dirs(current_label)
            count = len(os.listdir(save_dir))
            print(f"Switched to class: {current_label}")
        elif key == ord('0'):
            current_label = 'nothing'
            save_dir = create_dirs(current_label)
            count = len(os.listdir(save_dir))
            print(f"Switched to class: {current_label}")
        elif key == ord('1'):
            current_label = 'space'
            save_dir = create_dirs(current_label)
            count = len(os.listdir(save_dir))
            print(f"Switched to class: {current_label}")
        elif key == ord('2'):
            current_label = 'del'
            save_dir = create_dirs(current_label)
            count = len(os.listdir(save_dir))
            print(f"Switched to class: {current_label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
