
#  Chrome Dino AI: Self-Playing Agent

##  Project Description and Objectives

This project develops an AI agent that can autonomously play the Chrome Dinosaur game. It involves capturing gameplay data, preprocessing images, training convolutional neural networks (CNNs) to recognize obstacles (e.g., cactus, bird high, bird low), and deploying the best-performing model in real-time using `pyautogui`.

### Key Objectives:
- Collect and label gameplay data (frames and actions)
- Preprocess images and extract obstacles
- Train and evaluate multiple CNN models (DinoNet1, DinoNet2, DinoNet3)
- Perform hyperparameter tuning and visualize results using TensorBoard
- Deploy a fast and accurate self-playing Dino agent

---

## ⚙️ Installation and Setup Instructions

### Clone the repository
```bash
git clone https://github.com/yourusername/Chrome-Dino-AI.git
cd Chrome-Dino-AI
```

### Create a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

---

##  Usage Examples and Expected Outputs

### 1. Data Collection (manual gameplay & screenshots)
```bash
python data_collect.py
```
- Captures screenshots and key presses during gameplay
- Saves frames to `frames/` folder

### 2. Dataset Balancing
```bash
python balancing_dataset.py
```
- Balances the dataset across `cactus`, `bird_high`, `bird_low`

### 3. Preprocessing and Obstacle Extraction
```bash
python preprocess.py
```
- Crops obstacles from frames using OpenCV
- Saves to `data/processed_images/obstacles/`

### 4. Model Training with Hyperparameter Tuning
```bash
python model.py
```
- Trains DinoNet1, DinoNet2, and DinoNet3
- Logs performance in `hyperparameter_tuning_results.json`

### 5. Visualize Training in TensorBoard
```bash
tensorboard --logdir runs
```
- Open [http://localhost:6006](http://localhost:6006) to view loss/accuracy plots

### 6. Deploy the AI Agent
```bash
python DINO_Agent.py
```
- Starts real-time gameplay automation

---

##  Data Preparation Guidelines

1. **Manual Gameplay**  
   Use `data_collect.py` to collect ~1000 frames while playing manually. Keys are logged.

2. **Frame Processing**  
   Convert images to grayscale, resize to uniform size, normalize pixel values.

3. **Obstacle Extraction**  
   Automatically crop obstacle regions using bounding boxes.

4. **Directory Structure**
```
data/
├── frames/                    # Raw screenshots
├── temp/                      # Temporary cropped numpy arrays
├── processed_images/
│   └── obstacles/
│       ├── cactus/
│       ├── bird_high/
│       └── bird_low/
```

---

##  Model Architecture Details

###  DinoNet1
- 3 Convolutional Layers
- 4 Fully Connected Layers
- Deeper architecture with high capacity

###  DinoNet2 (Deployed Model)
- 3 Convolutional Layers
- 3 Fully Connected Layers
- Balanced in performance and speed

###  DinoNet3
- 2 Convolutional Layers
- 2 Fully Connected Layers
- Lightweight and fast

> All models use ReLU activation and CrossEntropy loss.

---

##  Performance Metrics and Evaluation Results

After hyperparameter tuning on a balanced dataset:

| Model     | Accuracy (%) | Parameters | Notes                     |
|-----------|--------------|------------|---------------------------|
| DinoNet1  | 86       | High       | Deep and expressive       |
| **DinoNet2**  | **89.6**         | Moderate   | Best trade-off (Deployed) |
| DinoNet3  | 84.5     | Low        | Fast but lower accuracy   |

- **DinoNet2** is deployed due to high accuracy and efficient inference.

---

##  requirements.txt

```
numpy
torch
torchvision
opencv-python
Pillow
pyautogui
matplotlib
tensorboard
```

To install:
```bash
pip install -r requirements.txt
```

---

##  License

MIT License. See `LICENSE` file for details.
