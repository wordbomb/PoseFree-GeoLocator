# PoseFree-GeoLocator

**Time Series for Flying Object Geographic Coordinate Recognition without Attitude Measurement and Reference Objects**

This repository provides the codebase, dataset, and documentation for our work on localizing UAVs in 3D space using only time-series visual data — without relying on attitude sensors or spatial references.

---

## 📁 Repository Contents (GitHub)

### 🔹 `datasets/detections/`
YOLOv12 detection results for one UAV flight recorded by three cameras on 2024-06-19 (`0619`).

- `0619-1.txt` — Detection output from Camera 1  
- `0619-2.txt` — Detection output from Camera 2  
- `0619-3.txt` — Detection output from Camera 3  

Each file contains one bounding box per frame in YOLO format:

    <class_id> <x_center> <y_center>    # all normalized

---

### 🔹 `datasets/trajectory/0619_fps6_rtk.txt`
This file contains the **ground-truth 3D trajectory** of the UAV collected via GNSS during the 0619 experiment.

Format (one line per frame, at 6 FPS):

    X Y Z

Each row corresponds to one timestamp.

---

## 📁 Video Dataset (Baidu Netdisk)

The full UAV detection dataset and video recordings are hosted on Baidu Netdisk:

🔗 **[Download (Baidu Netdisk)](https://pan.baidu.com/s/1iZbWehjIdt9_IRope5tQ3g)**  

This includes:

- Raw synchronized video sequences (3 views × 3 trajectories)
- YOLOv12 detection dataset: training/test images and annotations

💡 If the link becomes invalid, please contact the author for alternative access.

---

## 🧪 Numerical Simulation
This module tests the proposed 3D localization method using synthetic UAV trajectories and simulated camera inputs.

📁 `Numerical simulation/`  
• `main.ipynb` — Simulates trajectory, performs spatial reconstruction, and plots results  
• `core.py` — Implements triangulation and optimization logic  
• `visualization.py` — 3D plotting utilities for error analysis and result comparison  

Highlights:  
• Multi-camera views synthesized from ideal positions  
• Evaluates spatial accuracy under varying camera configurations and noise  
• Helps verify the algorithm’s robustness before real-world deployment  

---

## 🛩️ UAV Experiment

Real-world validation using multi-camera video recordings and YOLOv12-based detection of UAVs.

📁 `UAV experiment/UAV_3D_recognition/`

### 🟦 YOLOv12 Detection

• 📂 `YOLOv12/` — Forked and adapted YOLOv12 architecture  
• `train.ipynb` / `run_train.sh` — Scripts for model training  
• `predict.ipynb` — Inference with custom-trained weights  
• `runs_UAV/` — Logs and visualizations of training runs  
• `datasets/` — Over 48,000 labeled UAV images, annotated under diverse backgrounds  

Key Features:  
• UAV images collected under challenging conditions (twilight, backlight, buildings, etc.)  
• Manual annotations ensure high-quality bounding boxes  
• 80/20 train-test split with consistent format  

---

### 🔺 3D Reconstruction from Multi-View Detection

📁 `reconstruction/`  
• `main123-0619.ipynb` — Performs 3D trajectory reconstruction from YOLOv12 detections  
• `core.py` — Computes triangulated positions and applies refinement  
• `transformation.py` — Handles projection between camera and world coordinates  
• `visualization.py` — 3D scatter/line plots to evaluate reconstruction vs. ground truth  
• `data/` — Contains calibration files for camera intrinsics and extrinsics (e.g., `cam1_0619.json`)  

Pipeline:  
1. Detect UAV in each video frame using YOLOv12  
2. Map 2D detections to 3D coordinates through calibrated projection  
3. Apply bundle adjustment for trajectory smoothing  
4. Compare reconstructed trajectory with GNSS ground truth  

---

## 📄 Citation

If you use this dataset or code in your research, please cite:

    @article{Yi2025PoseFreeGeoLocator,
      title   = {Time Series for Flying Object Geographic Coordinate Recognition without Attitude Measurement and Reference Objects},
      author  = {Junfan Yi, Ke-ke Shang and Michael Small},
      journal = {In submission},
      year    = {2025}
    }

---

## 🛡 License

This dataset and code are released for **non-commercial academic use only**.  
For commercial use or redistribution, please contact the author.