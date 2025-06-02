import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import ImageGrab
import pyautogui
import time
import keyboard

# DinoNet2 model  
class DinoNet2(nn.Module):
    def __init__(self):
        super(DinoNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 10 * 14, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DinoNet2().to(device)
model.load_state_dict(torch.load("DinoNet2_lr0.001_bs32_lr0.001_epochs10.pth", map_location=device)) 
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((47, 64)),
    transforms.ToTensor(),
])

CROP_REGION = (180, 360, 980, 600)  

def grab_game_region():
    return ImageGrab.grab(bbox=CROP_REGION)

def predict_action(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted].item()
        print(f"Predicted class: {predicted}, Confidence: {confidence:.2f}")
    
    if predicted == 1 and confidence > 0.8:
        return 1
    else:
        return 0

def press_jump():
    pyautogui.press('up')

def main_loop():
    print("Press SPACE to start/stop the Dino AI. Press ESC to exit.")
    running = False
    jumped = False

    while True:
        if keyboard.is_pressed('esc'):
            print("Exiting...")
            break

        if keyboard.is_pressed('space'):
            running = not running
            print("Dino AI running:", running)
            time.sleep(0.1)

        if running:
            img = grab_game_region()
            action = predict_action(img)

            if action == 1 and not jumped:
                press_jump()
                jumped = True

            if action == 0 and jumped:
                jumped = False

            time.sleep(0.1)
        else: 
            time.sleep(0.1)

if __name__ == "__main__":
    main_loop()
 