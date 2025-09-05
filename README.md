# iPhone FastVLM Recorder

Record video from your **iPhone** on macOS with:
- **MediaPipe** hand + face (or face mesh) overlays
- **Continuous FastVLM** (Apple MLX) scene captions at the bottom of the video
- One-key **Space** to start/stop recording, **Q** to quit
- Optional **auto-transcription** (`.txt` + `.srt`) via **mlx-whisper**

---

## Features
- Open iPhone camera via **Continuity Camera** (AVFoundation)
- Real-time overlays: MediaPipe Hands + Face / FaceMesh
- Live caption bar describing Faces/Hands/Gestures
- **FastVLM** captions that continuously describe the scene
- Videos + transcripts save into the **current working directory**

---

## Requirements

- macOS with Continuity Camera enabled (iPhone nearby, unlocked)
- Apple Silicon (M1/M2/M3/M4 with unified memory recommended)
- [Conda / Miniforge](https://github.com/conda-forge/miniforge)  
- [Homebrew](https://brew.sh) (for `ffmpeg` + `wget`)  
- Git (to clone repos)  
- Stable internet (to download Apple’s FastVLM models)

---

## Install

You have three ways to set up dependencies:

---

### **Option A: Setup script (recommended)**

Run everything in one go:

```bash
chmod +x setup.sh
./setup.sh
```

### **Option B: environment.yml (conda one-liner)**

Create the conda environment via the YAML:

```bash
conda env create -f environment.yml
```

This installs Python 3.11, NumPy, OpenCV (conda-forge), mediapipe, mlx-whisper, and FastVLM (pip).

You still need to install model checkpoints somewhere on disk and point the app to them via env vars (see below).

### **Option C: Manual install (for full control)**

```bash
# Conda environment
conda create -n mp python=3.11 -y
conda activate mp
conda install -c conda-forge -y opencv numpy

# Pip packages
pip install mediapipe==0.10.14 mlx-whisper

# Homebrew for system tools
brew install ffmpeg wget

# FastVLM (clone anywhere you like)
git clone https://github.com/apple/ml-fastvlm.git /path/to/ml-fastvlm
pip install -e /path/to/ml-fastvlm

# Download model checkpoints
cd /path/to/ml-fastvlm
bash get_models.sh
```


## Run
Set environment variables so paths are portable (recommended):

```bash
export FASTVLM_REPO=/path/to/ml-fastvlm
export FASTVLM_MODEL=/path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3
```

Then run (assuming iPhone is `--cam-index 1`):

```bash
python iphonerecording.py \
  --cam-index 1 \
  --width 1920 --height 1080 --fps 30 \
  --face-mesh \
  --show-fps \
  --auto-transcribe
```

Alternatively, pass paths explicitly without env vars:

```bash
python iphonerecording.py \
  --cam-index 1 \
  --fastvlm-repo /path/to/ml-fastvlm \
  --fastvlm-model /path/to/ml-fastvlm/checkpoints/llava-fastvithd_0.5b_stage3
```

## Controls
	•	Space: start/stop recording
	•	Q / Esc: quit

## Output
	•	Videos: saved to the current working directory
	•	Filenames like: iphone_record_2025-09-05_15-12-33.mp4
	•	Transcripts: iphone_record_*.txt + iphone_record_*.srt

⸻

## Continuity Camera Checklist
	•	macOS Ventura+ and iOS 16+
	•	Same Apple ID on Mac & iPhone
	•	Wi-Fi + Bluetooth ON on both
	•	iPhone unlocked and nearby
	•	On iPhone: Settings → General → AirPlay & Handoff → Continuity Camera = ON
	•	On macOS: System Settings → Privacy & Security → Camera → allow Terminal/VS Code

    Check camera index with:
    ffmpeg -f avfoundation -list_devices true -i ""

   ## Troubleshooting
	•	iPhone not detected → toggle Continuity Camera OFF/ON on iPhone, restart FaceTime once, close other camera apps.
	•	Permission issues → tccutil reset Camera and rerun.
	•	No models found → set FASTVLM_MODEL to your checkpoint dir and re-run bash get_models.sh in your FastVLM clone.
	•	Performance → use 0.5b model for speed, lower --vlm-maxside (e.g., 384).
	•	Rotated video → add --rotate90.
