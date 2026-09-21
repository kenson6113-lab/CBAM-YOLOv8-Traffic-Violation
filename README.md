# An Approach to Real-Time Detection of Motorcycle Road Violations with a CBAM-Integrated YOLOv8 Framework

**Author:** Kent Anthony P. Lumanog  
**Institution:** Ateneo de Davao University  
**Repository URL:** https://github.com/kenson6113-lab/CBAM-YOLOv8-Traffic-Violation  

---

## 1. Hardware & Software Environment

* **Hardware Specs:** Google Colab Environment (NVIDIA Tesla T4 GPU, 16GB VRAM, 12.7GB System RAM)
* **Operating System:** Ubuntu 22.04 LTS (x86_64)
* **Programming Language:** Python 3.10.12
* **Key Libraries:**
  * `torch==2.1.2`
  * `torchvision==0.16.2`
  * `ultralytics==8.1.0`
  * `opencv-python==4.8.1.78`
  * `matplotlib==3.7.1`

---

## 2. Dataset Setup Instructions

To avoid repository size inflation, raw images are hosted separately and decoupled from Git source code.

1. Download the `master_traffic_violation_dataset` archive from the project storage/Kaggle link.
2. Extract the directory into your working root (e.g., `/content/master_traffic_violation_dataset`).
3. Ensure the folder structure matches:

```text
master_traffic_violation_dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
