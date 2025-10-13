# 🏀 Simple-NBA-Segmentation-Annotation-Tool

## Overview

**Simple-NBA-Segmentation-Annotation-Tool** is a research tool designed to automatically detect, segment, and annotate basketball players in video clips. It provides a simple **Tkinter-based GUI** that allows you to:

- Load a basketball video  
- Select a player to track  
- Run segmentation using **Segment Anything 2 (SAM2)**  
- Save the processed output video and auto-generated annotations (in JSON format)

This project builds upon techniques described in Roboflow’s guide:  
🔗 [Identify Basketball Players with AI](https://blog.roboflow.com/identify-basketball-players/)

---

## ⚙️ Features

- **Player Detection** using Roboflow’s basketball player detection model  
- **Segmentation** powered by **Segment Anything 2 (Real-Time)**  
- **GUI Interface** for easy interaction  
- **JSON Annotation Export** in COCO-like format  
- **Supports CUDA acceleration** for GPU-optimized inference  

---

## 🧩 GUI Layout

The interface is intentionally minimal for research and annotation workflow simplicity.

| Control | Description |
|----------|-------------|
| 🎥 **Select Input Video** | Choose an `.mp4` video to process |
| 🧍 **Select Player** | Choose which player to annotate (detected automatically) |
| ▶️ **Run Segmentation** | Start segmentation and annotation |
| 💾 **Save Output** | Save processed frames and annotation data |
| 📁 **Output Folder** | Where all results are saved |

---

## 🖥️ Example Output

After processing a video, the tool automatically produces:
- `output_video.mp4` — segmented and annotated video  
- `annotations.json` — JSON file containing frame-by-frame mask coordinates, labels, and bounding boxes  

---

## 🧠 Model Details

This tool integrates:
- **Roboflow Detection Model** for basketball player detection  
- **Meta’s Segment Anything 2 (SAM2)** for segmentation  
- **OpenCV** and **PyTorch** for image and video processing  

---

## 🧪 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Simple-NBA-Segmentation-Annotation-Tool.git
cd Simple-NBA-Segmentation-Annotation-Tool
