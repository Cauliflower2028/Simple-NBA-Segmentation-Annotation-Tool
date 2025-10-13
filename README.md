# NBA-Segmentation-Annotation-Tool

A minimal, user-friendly tool for segmenting basketball players in video clips and generating annotation masks, designed for research and annotation workflows.  
 **Supports GPU acceleration and is intended for use on OSC (Ohio Supercomputer Center) Linux clusters.**

 This project builds upon techniques described in Roboflow’s guide:  
[Identify Basketball Players with AI](https://blog.roboflow.com/identify-basketball-players/)

## Prerequisites 

### 1. Have the Ohio Supercomputer Virtual Machine set up before setup.

See this [documentation](https://docs.google.com/document/d/18efM3UhXIMKOZ-e1weG6fw2CS5-esf2s--cmgjhlOIs/edit?usp=sharing) for setup.

### 2. System Requirements

- Linux environment (OSC Ascend cluster)

- CUDA 12.6.2 support

- GPU access (required for SAM2 installation and runtime)

- Python 3.10

- Conda package manager


# Installation:

### 1. CREATE AND ACTIVATE VENV
```bash
conda create -n venv python=3.10 -y  
conda activate venv
```

### 2. UPGRADE PIP
```bash
pip install --upgrade pip
```

### 3. Load Required Modules
**IMPORTANT**: You must run this command every time you open a new terminal session:
```bash
module load ffmpeg/6.1.1
```  

### 4. Set GCC as the compiler
```bash
export CC=gcc  
export CXX=g++
```

### 5. Installing dependencies  
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126  
```

# 6. Install other dependencies
```bash
pip install -r requirements.txt
```

### 7. FOR INSTALLING SAM2 REAL-TIME - RUN THE FOLLOWING COMMANDS  
**IMPORTANT**: MAKE SURE TO REQUEST GPU BEFORE INSTALLING OR IT WILL NOT WORK!
```bash
git clone https://github.com/Gy920/segment-anything-2-real-time.git    
cd ./segment-anything-2-real-time  
pip install -e .  
python setup.py build_ext --inplace  
(cd checkpoints && bash download_ckpts.sh)  
cd ..
```

### 8. Install Additional Dependencies
```bash
pip install -r requirements.txt
pip install git+https://github.com/roboflow/supervision.git  
pip install git+https://github.com/roboflow/sports.git@feat/basketball
```

### 9. Set Environment Variable
```bash
export ONNXRUNTIME_EXECUTION_PROVIDERS="[CUDAExecutionProvider]"
```  

### 10. Configure API Keys

Create a .env file in your project root directory:
```bash
touch .env
```
Add the following contents to the .env file:
```bash
HF_TOKEN=your_huggingface_token_here  
ROBOFLOW_API_KEY=your_roboflow_key_here
```

GET HF_TOKEN FROM https://huggingface.co/settings/profile  
GET ROBOFLOW_API_KEY FROM https://app.roboflow.com/settings/api

## Troubleshooting 
**The biggest problem is installing SAM2 make sure to request GPU before installing!!**
