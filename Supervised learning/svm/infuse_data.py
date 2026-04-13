import os
import shutil
import glob

# Paths
SOURCE_1 = "/Users/daakka/.cache/kagglehub/datasets/ishanishah8/indian-classical-mudras-classification/versions/1"
SOURCE_2 = "/Users/daakka/.cache/kagglehub/datasets/demon2angel/asanyukta-kathak-mudra/versions/1"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Dataset 2 Class Mapping (from data.yaml)
DS2_CLASSES = ['Hamsapaksha', 'aral', 'ardhachandra', 'ardhpataka', 'bhramara', 'chandrakala', 'chatur', 'hansaasya', 'kangul', 'kapitth', 'kartarimukh', 'katak', 'mayur', 'mrighasheesh', 'mukul', 'mushti', 'padamkosh', 'pataka', 'sarpsheesh', 'shikhar', 'shuktund', 'sinhamukh', 'soochi', 'tamrachud', 'tripataka', 'trishool']

def infuse_dataset_1():
    print("Infusing Dataset 1 (Indian Classical Mudras)...")
    # Dataset 1 has train/test with class folders
    for split in ['train', 'test']:
        split_path = os.path.join(SOURCE_1, split)
        if not os.path.exists(split_path): continue
        
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path): continue
            
            target_class = class_name.replace(" ", "_").lower()
            target_dir = os.path.join(DATA_DIR, target_class)
            os.makedirs(target_dir, exist_ok=True)
            
            for img in os.listdir(class_path):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    shutil.copy2(os.path.join(class_path, img), os.path.join(target_dir, f"ds1_{split}_{img}"))

def infuse_dataset_2():
    print("Infusing Dataset 2 (Asanyukta Kathak Mudra)...")
    # Dataset 2 is in YOLO format: split/images/ and split/labels/
    for split in ['train', 'valid', 'test']:
        img_dir = os.path.join(SOURCE_2, split, 'images')
        lbl_dir = os.path.join(SOURCE_2, split, 'labels')
        if not os.path.exists(img_dir): continue
        
        for img_name in os.listdir(img_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')): continue
            
            # Find label
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # Take the first object's class ID
                        class_id = int(lines[0].split()[0])
                        if class_id < len(DS2_CLASSES):
                            class_name = DS2_CLASSES[class_id].lower()
                            target_dir = os.path.join(DATA_DIR, class_name)
                            os.makedirs(target_dir, exist_ok=True)
                            shutil.copy2(os.path.join(img_dir, img_name), os.path.join(target_dir, f"ds2_{split}_{img_name}"))

def main():
    if os.path.exists(DATA_DIR):
        print(f"Cleaning existing data directory: {DATA_DIR}")
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR)
    
    infuse_dataset_1()
    infuse_dataset_2()
    print("\nData infusion complete. Unified dataset created in 'data/' folder.")

if __name__ == "__main__":
    main()
