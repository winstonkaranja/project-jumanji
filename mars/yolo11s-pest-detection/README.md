---
license: mit
datasets:
  - IP102
library_name: ultralytics
tags:
  - object-detection
  - YOLO11s
  - pests
  - agriculture
  - ip102
model-index:
  - name: IP102 Pest Detector (YOLO11 Small)
    results:
      - task: 
            type: object-detection
        dataset:
            name: IP102
            type: pest-detection
        metrics:
          - type: mAP@0.5
            value: 0.941
          - type: mAP@0.5:0.95
            value: 0.838
          - type: Precision
            value: 0.923
          - type: Recall
            value: 0.907

---




# 🐞 IP102 Pest Detector — YOLO11 Small

A custom YOLO11 object detection model trained on the **IP102** dataset — designed for pest detection in precision agriculture.

> **Model Purpose:** Detect and classify 102 pest species in real-time field conditions using computer vision.

---

## 💡 Model Details

- **Model:** YOLO11 Small  
- **Dataset:** IP102 (Balanced, 14K+ images)  
- **Image Sizes:** Trained on 640x640 and 896x896  
- **Classes:** 102 pest species  
- **Framework:** Ultralytics YOLO11s  
- **Hardware:** NVIDIA A100 GPU  
- **Epochs:** 77  
- **License:** MIT License  

---

## 🧪 Performance

| Metric              | Train Set | Validation Set |
|----------------------|-----------|-----------------|
| Precision            | 0.912     | 0.744           |
| Recall               | 0.923     | 0.789           |
| mAP@0.5              | 0.941     | 0.815           |
| mAP@0.5:0.95         | 0.838     | 0.605           |

---

---
## 🐜 Class List 
The model detects 102 agricultural pests, including:

rice leaf roller

paddy stem maggot

brown plant hopper

aphids

mole cricket

blister beetle
...and many more!

(See pests.yaml for the full class list.)
--- 

## ⚖️ License
This project is released under the MIT License — free for personal and commercial use.


---

## 📚 Citation
If you use this model in research or production, please cite the IP102 dataset:

Wu, S., Zhan, C., et al.
"IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition."
CVPR, 2019.

---

## 💬 Questions?
Open an issue or reach me on Hugging Face Discussions.


---

## 📦 Usage

```python
from ultralytics import YOLO

# Load model
model = YOLO("path/to/best.pt")

# Run inference
results = model.predict("your_image.jpg", imgsz=640)

# Display results
results.show()
