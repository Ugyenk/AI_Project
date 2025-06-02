import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# dataset directory
base_dir = "data/temp"  # your folder

# Classes
classes = ['jump', 'nothing']

def load_and_crop_images():
    images = []
    labels = []

    for cls in classes:
        class_path = os.path.join(base_dir, cls)
        label = classes.index(cls)

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Failed to load {img_path}")
                continue

            cropped_img = img[78:125, 38:102] 

            images.append(cropped_img)
            labels.append(label)

    # Calculate new dimensions after cropping
    cropped_height, cropped_width = 125 - 78, 102 - 38  
    images = np.array(images).reshape(-1, cropped_height, cropped_width, 1)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images from {base_dir}")
    print(f"Cropped image dimensions: {cropped_height}x{cropped_width} pixels")
    return images, labels

if __name__ == "__main__":
    X, y = load_and_crop_images()
    np.save("data/temp/X_cropped.npy", X)
    np.save("data/temp/y_cropped.npy", y)

    plt.imshow(X[0].reshape(47, 64), cmap='gray')  
    plt.title(f"Label: {classes[y[0]]}")
    plt.show()