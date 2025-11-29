import os
import shutil
import random
import math

def split_and_organize_dataset(source_dir, dest_base_dir, max_per_class=500, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        print("Error: Ratios must sum to 1.0")
        return

    train_dir = os.path.join(dest_base_dir, 'train')
    valid_dir = os.path.join(dest_base_dir, 'valid')
    test_dir = os.path.join(dest_base_dir, 'test')

    for d in [train_dir, valid_dir, test_dir]:
        if os.path.exists(d):
            print(f"Removing existing directory: {d}")
            shutil.rmtree(d)
        os.makedirs(d)

    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    print(f"Found {len(classes)} classes in {source_dir}")

    for cls in classes:
        cls_source_dir = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        random.shuffle(files)
        
        if len(files) > max_per_class:
            print(f"Class {cls}: Truncating {len(files)} -> {max_per_class}")
            files = files[:max_per_class]
        
        total_files = len(files)
        n_train = int(total_files * train_ratio)
        n_valid = int(total_files * valid_ratio)
        n_test = total_files - n_train - n_valid

        train_files = files[:n_train]
        valid_files = files[n_train:n_train+n_valid]
        test_files = files[n_train+n_valid:]

        print(f"Class {cls}: {total_files} images -> Train: {len(train_files)}, Valid: {len(valid_files)}, Test: {len(test_files)}")

        def copy_files(file_list, destination):
            cls_dest = os.path.join(destination, cls)
            os.makedirs(cls_dest, exist_ok=True)
            for f in file_list:
                shutil.copy2(os.path.join(cls_source_dir, f), os.path.join(cls_dest, f))

        copy_files(train_files, train_dir)
        copy_files(valid_files, valid_dir)
        copy_files(test_files, test_dir)

    print("\nData split and organization completed successfully!")
    print(f"Train data: {train_dir}")
    print(f"Valid data: {valid_dir}")
    print(f"Test data: {test_dir}")

if __name__ == "__main__":
    SOURCE_DIR = os.path.join(os.getcwd(), 'data', 'my_data_clean')
    DEST_BASE_DIR = os.path.join(os.getcwd(), 'data')
    
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist. Please run data collection first.")
    else:
        split_and_organize_dataset(SOURCE_DIR, DEST_BASE_DIR)
