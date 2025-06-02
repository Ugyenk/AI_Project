import os
from PIL import Image
import numpy as np
import cv2

# Source and destination base directories
SOURCE_BASE = "data/temp_balanced"          
DEST_BASE = "data/processed_data"  

# Classes 
classes = ["0", "1"] 

# Create destination folders
for cls in classes:
    os.makedirs(os.path.join(DEST_BASE, cls), exist_ok=True)

def process_and_save_images():
    for cls in classes:
        src_folder = os.path.join(SOURCE_BASE, cls)
        dest_folder = os.path.join(DEST_BASE, cls)

        for filename in os.listdir(src_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(src_folder, filename)
                dest_path = os.path.join(dest_folder, filename)

                img = Image.open(src_path).convert('L')

                img = img.resize((384, 216))

                img_np = np.array(img)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                img_np = clahe.apply(img_np)

                img_processed = Image.fromarray(img_np)

                img_processed.save(dest_path)

        print(f"Processed class '{cls}' images saved to {dest_folder}")

if __name__ == "__main__":
    process_and_save_images()
