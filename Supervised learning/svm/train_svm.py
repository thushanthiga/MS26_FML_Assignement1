import os
import numpy as np
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_mudra_data(data_dir, img_size=(128, 128)):
    features = []
    labels = []
    class_names = []
    
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"Warning: Data directory '{data_dir}' is empty or does not exist.")
        return None, None, None

    # Get class folders, filtering for directories only
    class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    
    for class_folder in class_folders:
        class_names.append(class_folder)
        class_path = os.path.join(data_dir, class_folder)
        print(f"Loading class: {class_folder}")
        
        count = 0
        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            img_path = os.path.join(class_path, img_file)
            try:
                # Read and resize
                img = cv2.imread(img_path)
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, img_size)
                
                # Extract HOG features
                fd = hog(img, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False)
                
                features.append(fd)
                labels.append(len(class_names) - 1)
                count += 1
            except Exception as e:
                print(f"  Error loading {img_file}: {e}")
        print(f"  Total images loaded for {class_folder}: {count}")
                    
    return np.array(features), np.array(labels), class_names

def main():
    DATA_PATH = "data"
    print("Loading data and extracting HOG features...")
    X, y, class_names = load_mudra_data(DATA_PATH)

    if X is None or len(X) == 0:
        print("Error: No images were loaded. Check the 'data' folder.")
        return

    print(f"\nTotal images loaded: {len(X)}")
    print(f"Feature vector shape: {X.shape[1]}")

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")

    # Train SVM Model
    print("Training SVM model (this may take a few minutes)...")
    clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluation
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    report = classification_report(y_test, y_pred, target_names=class_names)
    print("\nClassification Report:\n", report)

    # Save results to a file
    with open("classification_results.txt", "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print("\nResults saved to 'classification_results.txt'.")

if __name__ == "__main__":
    main()
