# PoseFree-GeoLocator

**Time Series for Flying Object Geographic Coordinate Recognition without Attitude Measurement and Reference Objects**

This repository provides code, ground-truth data, and detection results for a UAV 3D localization experiment based on visual time-series input. It accompanies the work:

> **Time Series for Flying Object Geographic Coordinate Recognition without Attitude Measurement and Reference Objects**  

---

## 📁 Repository Contents (GitHub)

### 🔹 `detections/`
YOLOv12 detection results for one UAV flight recorded by three cameras on 2024-06-19 (`0619`).

- `0619-1.txt` — Detection output from Camera 1  
- `0619-2.txt` — Detection output from Camera 2  
- `0619-3.txt` — Detection output from Camera 3  

Each file contains one bounding box per frame in YOLO format:

<class_id> <x_center> <y_center>    # all normalized

---

### 🔹 `trajectory/0619_fps6_rtk.txt`
This file contains the **ground-truth 3D trajectory** of the UAV collected via RTK-GNSS during the 0619 experiment.

Format (one line per frame, at 30 FPS):

X Y Z

---

### 🔹 `Numerical simulation/`
Python scripts for 3D trajectory reconstruction and evaluation:

- `core.py`: Core functions for geometric processing
- `main.ipynb`: Main workflow (can be run in Jupyter)
- `visualization.py`: 3D plotting and reconstruction result visualization

---

## 📦 Full Dataset (百度网盘)

The full UAV detection dataset and video recordings are hosted on Baidu Netdisk:

🔗 **[📥 Download (百度网盘)](https://pan.baidu.com/s/1iZbWehjIdt9_IRope5tQ3g)**  

This includes:

- 📹 Raw synchronized video sequences (3 views × 3 trajectories)
- 🖼️ YOLOv12 detection dataset: training/test images and annotations

> 💡 If the link becomes invalid, please contact the author for alternative access.

---

## 📄 Citation

If you use this dataset or code in your research, please cite:

@article{Yi2025PoseFreeGeoLocator,
title   = {Time Series for Flying Object Geographic Coordinate Recognition without Attitude Measurement and Reference Objects},
author  = {Junfan Yi， Ke-ke Shang and Michael Small},
journal = {In submission},
year    = {2025}
}
## 🔖 License

This dataset and code are released for **non-commercial academic use only**.  
For commercial use or redistribution, please contact the author.