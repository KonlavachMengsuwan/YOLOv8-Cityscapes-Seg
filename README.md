# YOLOv8-Cityscapes-Seg
```
import os
import json
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

# ============================== #
#  ✅ 1. SETUP ENVIRONMENT       #
# ============================== #

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# Create directories if not exist
CITYSCAPES_ROOT = "cityscapes"
IMAGE_DIR = os.path.join(CITYSCAPES_ROOT, "leftImg8bit")
ANNOTATION_DIR = os.path.join(CITYSCAPES_ROOT, "gtFine")
YOLO_LABELS_DIR = os.path.join(CITYSCAPES_ROOT, "labels_yolo")

os.makedirs(YOLO_LABELS_DIR, exist_ok=True)

# Define Cityscapes classes
CLASS_NAMES = ["road", "sidewalk", "building", "wall", "fence", "pole",
               "traffic light", "traffic sign", "vegetation", "terrain",
               "sky", "person", "rider", "car", "truck", "bus", "train",
               "motorcycle", "bicycle"]
CLASS_DICT = {name: i for i, name in enumerate(CLASS_NAMES)}

# ============================== #
#  ✅ 2. CONVERT CITYSCAPES TO YOLO FORMAT  #
# ============================== #

def convert_cityscapes_to_yolo():
    """ Convert Cityscapes annotation to YOLO segmentation format. """
    for split in ["train", "val"]:
        img_split_dir = os.path.join(IMAGE_DIR, split)
        ann_split_dir = os.path.join(ANNOTATION_DIR, split)
        yolo_split_dir = os.path.join(YOLO_LABELS_DIR, split)
        os.makedirs(yolo_split_dir, exist_ok=True)

        for city in tqdm(os.listdir(img_split_dir), desc=f"Processing {split} set"):
            img_files = os.listdir(os.path.join(img_split_dir, city))
            for img_file in img_files:
                img_path = os.path.join(img_split_dir, city, img_file)
                ann_file = img_file.replace("leftImg8bit.png", "gtFine_polygons.json")
                ann_path = os.path.join(ann_split_dir, city, ann_file)
                label_path = os.path.join(yolo_split_dir, img_file.replace(".png", ".txt"))

                if not os.path.exists(ann_path):
                    continue  # Skip if annotation is missing

                with open(ann_path, "r") as f:
                    ann_data = json.load(f)

                img = cv2.imread(img_path)
                img_h, img_w = img.shape[:2]

                with open(label_path, "w") as f:
                    for obj in ann_data["objects"]:
                        label = obj["label"]
                        if label in CLASS_DICT:
                            class_id = CLASS_DICT[label]
                            polygon = np.array(obj["polygon"], dtype=np.float32)
                            polygon[:, 0] /= img_w  # Normalize x-coordinates
                            polygon[:, 1] /= img_h  # Normalize y-coordinates
                            poly_str = " ".join(map(str, polygon.flatten().tolist()))
                            f.write(f"{class_id} {poly_str}\n")

# Convert dataset if not already converted
if not os.path.exists(YOLO_LABELS_DIR):
    convert_cityscapes_to_yolo()
    print("✅ Cityscapes dataset converted to YOLO format.")

# ============================== #
#  ✅ 3. CREATE data.yaml FILE   #
# ============================== #

data_yaml_path = os.path.join(CITYSCAPES_ROOT, "data.yaml")

if not os.path.exists(data_yaml_path):
    with open(data_yaml_path, "w") as f:
        f.write(f"""path: {CITYSCAPES_ROOT}
train: leftImg8bit/train
val: leftImg8bit/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
""")
    print("✅ data.yaml created successfully.")

# ============================== #
#  ✅ 4. TRAIN YOLOv8 ON CITYSCAPES   #
# ============================== #

# Load YOLOv8 segmentation model
model = YOLO("yolov8n-seg.pt")  # Use 's', 'm', 'l' for larger models

# Train the model
model.train(data=data_yaml_path, epochs=20, batch=8, imgsz=640, device=device)

# Save the best trained model path
trained_model_path = "runs/segment/train/weights/best.pt"
print(f"✅ Training completed. Best model saved at: {trained_model_path}")

# ============================== #
#  ✅ 5. RUN INFERENCE ON CUSTOM IMAGES   #
# ============================== #

def run_inference(image_path):
    """ Run YOLOv8 segmentation inference on a given image. """
    model = YOLO(trained_model_path)
    results = model(image_path)

    for result in results:
        annotated_img = result.plot()
        plt.imshow(annotated_img)
        plt.axis('off')
        plt.show()

# Example: Run inference on your custom image
custom_image_path = "path/to/your/image.jpg"  # Change this to your image
if os.path.exists(custom_image_path):
    print(f"✅ Running inference on: {custom_image_path}")
    run_inference(custom_image_path)
else:
    print("⚠️ Custom image not found. Please update 'custom_image_path'.")

# ============================== #
#  ✅ 6. RUN INFERENCE ON VIDEO (OPTIONAL)  #
# ============================== #

def run_video_inference(video_path):
    """ Run YOLOv8 segmentation inference on a video. """
    model.predict(source=video_path, save=True, device=device)

# Example: Run inference on a video file
video_path = "path/to/your/video.mp4"  # Change this to your video file
if os.path.exists(video_path):
    print(f"✅ Running inference on: {video_path}")
    run_video_inference(video_path)
else:
    print("⚠️ Video file not found. Please update 'video_path'.")

```

📌 How This Script Works
✔ 1. Checks CUDA & Sets Up Environment (Fast execution)
✔ 2. Converts Cityscapes dataset to YOLO segmentation format
✔ 3. Creates data.yaml file automatically
✔ 4. Trains YOLOv8 segmentation model with optimized settings
✔ 5. Runs inference on custom images (Shows results)
✔ 6. Runs inference on videos (Saves results)

🚀 How to Use This Script
1️⃣ Download & Extract Cityscapes Dataset
- Download leftImg8bit_trainvaltest.zip
- Download gtFine_trainvaltest.zip
- Extract both into cityscapes/
- 
2️⃣ Run the script on your PC with CUDA

```
python train_yolo_cityscapes.py
```

3️⃣ Check results in runs/segment/train/
4️⃣ Modify custom_image_path to test your images
5️⃣ Modify video_path to test on a video

🔥 Optimizations in This Script
✅ Efficient Dataset Conversion (Runs only once)
✅ Auto-detects CUDA (Runs smoothly on GPU)
✅ Uses Fast Training Settings (batch=8, imgsz=640, epochs=20)
✅ Runs Inference on Images & Videos
