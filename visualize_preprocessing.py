import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import math

DATA_DIR = 'data/train'

def get_all_image_paths(data_dir):
    """Helper to get all image paths and their labels."""
    image_paths = []
    labels = []
    if not os.path.exists(data_dir):
        return [], []
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(cls_dir, img_name))
                labels.append(cls)
    return image_paths, labels, classes

def plot_class_distribution(labels, classes):
    """Plots a bar chart showing the number of images per class."""
    counts = {cls: labels.count(cls) for cls in classes}
    
    plt.figure(figsize=(15, 6))
    plt.bar(counts.keys(), counts.values(), color='skyblue', edgecolor='black')
    plt.title('Class Distribution (Augmented Dataset)', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('class_distribution.png')
    print("Saved 'class_distribution.png'")
    # plt.show()

def plot_sample_batch(image_paths, labels, rows=5, cols=5):
    """Plots a grid of random images from the dataset."""
    if not image_paths:
        print("No images found.")
        return

    indices = random.sample(range(len(image_paths)), min(len(image_paths), rows * cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    fig.suptitle(f'Random Sample Batch ({rows}x{cols}) - Processed Data', fontsize=16)
    
    for i, idx in enumerate(indices):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        img_path = image_paths[idx]
        label = labels[idx]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis('off')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('sample_batch.png')
    print("Saved 'sample_batch.png'")
    # plt.show()

def visualize_preprocessing_comparison(data_dir):
    """Visualizes the preprocessing pipeline: Input -> Gray -> CLAHE."""
    classes_to_visualize = ['A', 'B', 'C', 'L', 'Y'] # Visualize a few distinct classes
    sample_images = []
    
    available_classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes_to_use = [c for c in classes_to_visualize if c in available_classes]
    if not classes_to_use and available_classes:
        classes_to_use = random.sample(available_classes, min(3, len(available_classes)))
    
    for cls in classes_to_use:
        cls_dir = os.path.join(data_dir, cls)
        images = [img for img in os.listdir(cls_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if images:
            img_name = random.choice(images)
            sample_images.append((cls, os.path.join(cls_dir, img_name)))
            
    if not sample_images:
        return

    # 4 Columns: Input, Gray, CLAHE, Histogram
    fig, axes = plt.subplots(len(sample_images), 4, figsize=(20, 5 * len(sample_images)))
    if len(sample_images) == 1:
        axes = np.expand_dims(axes, axis=0)
        
    fig.suptitle('Preprocessing Pipeline: BG Removed -> Grayscale -> CLAHE', fontsize=16)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    for i, (cls, img_path) in enumerate(sample_images):
        # 1. Input (Already BG Removed)
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 3. CLAHE
        equalized = clahe.apply(gray)
        
        # Plot 1: Input
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"Input (BG Removed) - {cls}")
        axes[i, 0].axis('off')
        
        # Plot 2: Grayscale
        axes[i, 1].imshow(gray, cmap='gray')
        axes[i, 1].set_title("Grayscale")
        axes[i, 1].axis('off')
        
        # Plot 3: CLAHE
        axes[i, 2].imshow(equalized, cmap='gray')
        axes[i, 2].set_title("CLAHE (Final Input)")
        axes[i, 2].axis('off')
        
        # Plot 4: Histogram of CLAHE
        axes[i, 3].hist(equalized.ravel(), 256, [0, 256], color='black', alpha=0.7)
        axes[i, 3].set_title("CLAHE Histogram")
        axes[i, 3].set_xlim([0, 256])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('preprocessing_comparison.png')
    print("Saved 'preprocessing_comparison.png'")

if __name__ == "__main__":
    print(f"Scanning data in {DATA_DIR}...")
    image_paths, labels, classes = get_all_image_paths(DATA_DIR)
    
    if image_paths:
        print(f"Found {len(image_paths)} images in {len(classes)} classes.")
        
        print("Generating Class Distribution Chart...")
        plot_class_distribution(labels, classes)
        
        print("Generating Sample Batch Visualization...")
        plot_sample_batch(image_paths, labels, rows=5, cols=5)
        
        print("Generating Preprocessing Comparison...")
        visualize_preprocessing_comparison(DATA_DIR)
        
        print("\n" + "="*80)
        print("GIẢI THÍCH CÁC BIỂU ĐỒ:")
        print("="*80)
        print("1. class_distribution.png (Phân phối lớp):")
        print("   - Kiểm tra xem dữ liệu đã cân bằng chưa (mục tiêu ~500 ảnh/lớp).")
        print("   - Nếu các cột đều nhau, việc Augmentation đã thành công.")
        print("\n2. sample_batch.png (Mẫu dữ liệu):")
        print("   - Xem ngẫu nhiên 25 ảnh từ tập train.")
        print("   - Kiểm tra chất lượng ảnh sau khi xóa nền (nền đen, tay rõ).")
        print("\n3. preprocessing_comparison.png (Quy trình tiền xử lý):")
        print("   - Cột 1: Ảnh đầu vào (đã xóa nền).")
        print("   - Cột 2: Ảnh xám (Grayscale) - Loại bỏ màu sắc.")
        print("   - Cột 3: CLAHE - Cân bằng sáng, làm rõ chi tiết tay.")
        print("   - Cột 4: Biểu đồ Histogram của ảnh CLAHE - Phân bố độ sáng.")
        print("="*80)
    else:
        print(f"No images found in {DATA_DIR}")
