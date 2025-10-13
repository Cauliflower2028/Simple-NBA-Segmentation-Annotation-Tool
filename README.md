# NBA-Segmentation-Annotation-Tool

A minimal, user-friendly tool for segmenting basketball players in video clips and generating annotation masks, designed for research and annotation workflows.  
 **Supports GPU acceleration and is intended for use on OSC (Ohio Supercomputer Center) Linux clusters.**

 This project builds upon techniques described in Roboflow’s guide:  
[Identify Basketball Players with AI](https://blog.roboflow.com/identify-basketball-players/)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [OSC Setup](#osc-setup)
- [Installation](#installation)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [API Keys](#api-keys)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This tool provides a simple GUI for segmenting basketball players in video clips, using state-of-the-art models and libraries.  
 It leverages [Roboflow](https://blog.roboflow.com/identify-basketball-players/) for player detection and [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time) for segmentation.  
 Annotations are saved in COCO format for downstream tasks.

## Features

- Select input video and output folder via GUI
- Specify player name and motion class (e.g., "3 point", "free-throw")
- Visualize original and segmented video side-by-side
- Save output video and annotation masks
- Fast, GPU-accelerated processing

## Prerequisites

- **OSC Account** and access to the Ascend cluster
- **Linux environment** (OSC cluster recommended)
- **GPU support** (CUDA 12.x)
- **Conda** installed

## OSC Setup

1.  **Read the OSC setup guide** (see pinned messages in Slack).
2.  **Configure SSH**:  
     Edit your SSH config file (`~/.ssh/config`) and add:
    ```
    Host osc-ascend
    	  HostName ascend-login01.hpc.osc.edu
    	  User <your_username>
    ```
3.  **Connect to OSC Ascend cluster** using VS Code SSH extension or terminal:
    ```bash
    ssh osc-ascend
    ```
    Your home directory should be `/users/PAS3184/<your_username>`.

## Installation

### 1. Clone Required Repositories

- Clone this project and required segmentation tools:
  ```bash
  git clone https://github.com/Cauliflower2028/Simple-NBA-Segmentation-Annotation-Tool.git
  git clone https://github.com/Gy920/segment-anything-2-real-time.git
  ```

### 2. X-AnyLabeling Setup (Optional/Recommended)

Follow the two-part installation guide for [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling):

- [Part 1: Interactive Video Object Segmentation](https://github.com/CVHub520/X-AnyLabeling/blob/main/examples/interactive_video_object_segmentation/README.md)
- [Part 2: Get Started](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md)

**Important steps:**

- Install ONNX Runtime GPU:
  ```bash
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  ```
- Install requirements:
  ```bash
  pip install -r requirements-gpu.txt
  ```
- Set environment variable:
  ```bash
  export PYTHONPATH=/path/to/X-AnyLabeling
  ```
- If `segment-anything-2/sam2/_C.so` is missing, copy it from `/fs/scratch/PAS3184/baicheng/`.

## Environment Setup

1.  **Create and activate Conda environment:**

    ```bash
    conda create -n venv python=3.10 -y
    conda activate venv
    ```

2.  **Upgrade pip:**

    ```bash
    pip install --upgrade pip
    ```

3.  **Load required modules (every new terminal session):**
    ```bash
    module load cuda/12.6.2
    module load cudnn/8.9.7.29-12
    module load ffmpeg/6.1.1
    export CC=gcc
    export CXX=g++
    ```

## Dependencies

Install the following Python packages in your environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install numpy wheel ninja pycocotools Pillow
pip install gdown inference-gpu
pip install git+https://github.com/roboflow/supervision.git
pip install git+https://github.com/roboflow/sports.git@feat/basketball
pip install transformers num2words
pip install flash-attn --no-build-isolation
```

### segment-anything-2-real-time

```bash
cd segment-anything-2-real-time
pip install -e .
python setup.py build_ext --inplace
(cd checkpoints && bash download_ckpts.sh)
cd ..
```

## API Keys

Create a `.env` file in the project root with the following contents:

```
HF_TOKEN=your_huggingface_token_here
ROBOFLOW_API_KEY=your_roboflow_key_here
```

- Get `HF_TOKEN` from [HuggingFace profile](https://huggingface.co/settings/profile)
- Get `ROBOFLOW_API_KEY` from [Roboflow API settings](https://app.roboflow.com/settings/api)

## Installation & Setup (Step-by-Step)

**Follow these steps exactly to set up your environment and install all dependencies.**

### 1. Clone Required Repositories

```bash
git clone https://github.com/Cauliflower2028/Simple-NBA-Segmentation-Annotation-Tool.git
git clone https://github.com/Gy920/segment-anything-2-real-time.git
```

### 2. Create and Activate Conda Environment

```bash
conda create -n venv python=3.10 -y
conda activate venv
```

### 3. Upgrade pip

```bash
pip install --upgrade pip
```

### 4. Load CUDA, cuDNN, and ffmpeg Modules (Run these every time you open a new terminal)

```bash
module load cuda/12.6.2
module load cudnn/8.9.7.29-12
module load ffmpeg/6.1.1
```

### 5. Set GCC as the Compiler

```bash
export CC=gcc
export CXX=g++
```

### 6. Install PyTorch (GPU) Packages

**Important:** You must install torch, torchvision, and torchaudio with the custom index URL for CUDA support. Do this before installing other requirements:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### 7. Install Other Python Dependencies

All other dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### 8. Install segment-anything-2-real-time

```bash
cd segment-anything-2-real-time
pip install -e .
python setup.py build_ext --inplace
(cd checkpoints && bash download_ckpts.sh)
cd ..
```

### 9. (Optional) Install Additional Dependencies

If you need to install any extra packages not listed in `requirements.txt`, refer to the installation.txt for specific commands.

### 9. Set Environment Variable for ONNX Runtime

```bash
export ONNXRUNTIME_EXECUTION_PROVIDERS="[CUDAExecutionProvider]"
```

### 10. Set Up API Keys

Create a file named `.env` in the project root with the following contents:

```
HF_TOKEN=your_huggingface_token_here
ROBOFLOW_API_KEY=your_roboflow_key_here
```

- Get `HF_TOKEN` from [HuggingFace profile](https://huggingface.co/settings/profile)
- Get `ROBOFLOW_API_KEY` from [Roboflow API settings](https://app.roboflow.com/settings/api)

---

### (Optional) X-AnyLabeling Setup

Follow the two-part installation guide for [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling):

- [Part 1: Interactive Video Object Segmentation](https://github.com/CVHub520/X-AnyLabeling/blob/main/examples/interactive_video_object_segmentation/README.md)
- [Part 2: Get Started](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md)

**Important steps:**

- Install ONNX Runtime GPU:
  ```bash
  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  ```
- Install requirements:
  ```bash
  pip install -r requirements-gpu.txt
  ```
- Set environment variable:
  ```bash
  export PYTHONPATH=/path/to/X-AnyLabeling
  ```
- If `segment-anything-2/sam2/_C.so` is missing, copy it from `/fs/scratch/PAS3184/baicheng/`.
- [Roboflow Basketball Player Detection](https://blog.roboflow.com/identify-basketball-players/)
- [segment-anything-2-real-time](https://github.com/Gy920/segment-anything-2-real-time)
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling)
- [OSC Documentation](https://www.osc.edu/documentation)

## Troubleshooting

- **Modules not found:** Make sure you have loaded CUDA, cuDNN, and ffmpeg modules in every new terminal session.
- **Missing _C.so:** If you get errors about missing `_C.so`, copy it from `/fs/scratch/PAS3184/baicheng/` to the correct directory.
- **API errors:** Double-check your `.env` file for correct HuggingFace and Roboflow tokens.
- **GUI issues:** If the GUI does not display, ensure you have X11 forwarding enabled when using SSH.
- **PyTorch install issues:** Always use the custom index URL for torch, torchvision, and torchaudio as described above.
