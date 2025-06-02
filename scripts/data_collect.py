import keyboard
from PIL import ImageGrab
import time
import os
from datetime import datetime

# Base directory 
BASE_PATH = "data/temp"

# Create class directories 
for class_name in ["jump", "nothing"]:
    os.makedirs(os.path.join(BASE_PATH, class_name), exist_ok=True)

def run(save_path):
    keys = []
    imgs = []

    print("Press SPACE to start capturing...")
    while not keyboard.is_pressed('space'):
        time.sleep(0.1)

    print("Capturing... Press Q to stop")

    prev = time.time()
    count = 0

    while True:
        img = ImageGrab.grab().resize((384, 216)).convert(mode='L')
        imgs.append(img)
        now = time.time()
        print(f"Time since last capture: {now - prev:.2f}s, Frames captured: {count}", end='\r')
        prev = now

        if keyboard.is_pressed('space') or keyboard.is_pressed('up'):
            keys.append("jump")
        else:
            keys.append("nothing")

        count += 1

        if keyboard.is_pressed('q'):
            print("\nStopping capture...")
            break

        time.sleep(0.05)  # small delay to avoid duplicates

    print(f"\nTotal images captured: {len(imgs)}")
    save(imgs, keys, save_path)

def save(imgs, keys, save_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, (key, img) in enumerate(zip(keys, imgs), start=1):
        class_dir = os.path.join(save_path, key)
        filename = f"{timestamp}_{i:04d}.png"
        filepath = os.path.join(class_dir, filename)
        img.save(filepath, 'PNG')
    print(f"Saved {len(imgs)} images to {save_path}")

if __name__ == '__main__':
    print("Dino Dataset Creation Tool")
    print("Controls:")
    print("- SPACE/UP: Capture jump image")
    print("- No key pressed: Capture 'nothing' image")
    print("- Q: Stop capturing")

    run(BASE_PATH)
