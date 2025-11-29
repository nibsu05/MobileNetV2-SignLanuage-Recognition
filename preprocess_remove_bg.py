import cv2
import mediapipe as mp
import os
import numpy as np
from tqdm import tqdm

# Configuration
INPUT_DIR = 'data/my_data'
OUTPUT_DIR = 'data/my_data_clean'
IMG_SIZE = 128

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def process_and_save(input_path, output_path):
    image = cv2.imread(input_path)
    if image is None:
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        h, w, _ = image.shape
        
        landmarks = results.multi_hand_landmarks[0]
        
        points = []
        for lm in landmarks.landmark:
            points.append((int(lm.x * w), int(lm.y * h)))
        
        hull = cv2.convexHull(np.array(points))
        
        cv2.fillConvexPoly(mask, hull, 255)
        
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        img_masked = cv2.bitwise_and(image, image, mask=mask)
        
        x, y, w_box, h_box = cv2.boundingRect(hull)
        
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w_box = min(w - x, w_box + 2*pad)
        h_box = min(h - y, h_box + 2*pad)
        
        img_cropped = img_masked[y:y+h_box, x:x+w_box]
        
        if img_cropped.size != 0:
            img_final = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(output_path, img_final)
            return True
            
    return False

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory {INPUT_DIR} not found.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    classes = [d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))]
    print(f"Found classes: {classes}")

    total_processed = 0
    total_failed = 0

    for cls in classes:
        src_cls_dir = os.path.join(INPUT_DIR, cls)
        dst_cls_dir = os.path.join(OUTPUT_DIR, cls)
        
        if not os.path.exists(dst_cls_dir):
            os.makedirs(dst_cls_dir)
            
        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Processing class '{cls}'...")
        
        processed_files = []
        
        if cls == 'nothing':
            for f in tqdm(files, desc=f"Copying {cls}"):
                src_path = os.path.join(src_cls_dir, f)
                dst_path = os.path.join(dst_cls_dir, f)
                
                img = cv2.imread(src_path)
                if img is not None:
                    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(dst_path, img_resized)
                    processed_files.append(dst_path)
                    total_processed += 1
        else:
            for f in tqdm(files, desc=f"Segmenting {cls}"):
                src_path = os.path.join(src_cls_dir, f)
                dst_path = os.path.join(dst_cls_dir, f)
                
                success = process_and_save(src_path, dst_path)
                if success:
                    total_processed += 1
                    processed_files.append(dst_path)
                else:
                    total_failed += 1

        # --- Data Augmentation ---
        current_count = len(processed_files)
        target_count = 500
        
        if current_count > 0 and current_count < target_count:
            needed = target_count - current_count
            print(f"  -> Class '{cls}' has {current_count} images. Generating {needed} augmented images...")
            
            for i in range(needed):
                src_img_path = np.random.choice(processed_files)
                img = cv2.imread(src_img_path)
                
                if img is None: continue
                
                rows, cols, _ = img.shape
                
                # 1. Random Rotation (-15 to 15 degrees)
                angle = np.random.uniform(-15, 15)
                M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                img_aug = cv2.warpAffine(img, M, (cols, rows))
                
                # 2. Random Brightness (0.8 to 1.2)
                alpha = np.random.uniform(0.8, 1.2)
                img_aug = cv2.convertScaleAbs(img_aug, alpha=alpha, beta=0)
                
                aug_filename = f"aug_{i}_{os.path.basename(src_img_path)}"
                aug_path = os.path.join(dst_cls_dir, aug_filename)
                cv2.imwrite(aug_path, img_aug)
            
            print(f"  -> Finished augmentation for '{cls}'. Total: {target_count}")

    print(f"\nProcessing Complete.")
    print(f"Successfully processed: {total_processed}")
    print(f"Failed (No hand detected): {total_failed}")
    print(f"Clean data saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
