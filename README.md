<div align="center">

# EXON FACE

### Advanced AI Face Swapping & Enhancement Framework

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Gradio-orange?style=for-the-badge&logo=gradio)](https://gradio.app/)
[![Accelerator](https://img.shields.io/badge/Accelerator-CUDA%2011.8%2B-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

_High-Performance, Ethics-First face manipulation tool powered by ONNX Runtime and InsightFace._

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Tech Stack](#technology-stack) • [Disclaimer](#ethical-disclaimer-and-legal-notice)

</div>

---

## 📖 Overview

**EXON FACE** is a state-of-the-art implementation of face swapping technology, engineered for high fidelity, performance, and user control. Unlike basic implementations, EXON focuses on professional workflows, offering granular masking controls (BiSeNet), optimized FP16 inference pipelines, and a suite of post-processing enhancers to deliver cinema-quality results.

Built with a modular architecture, it leverages **ONNX Runtime** for hardware-agnostic acceleration (NVIDIA CUDA support comes pre-tuned) and a reactive **Gradio** interface for seamless interaction.

##  Features

###  Core Capabilities

- **High-Fidelity Face Swapping**: Utilizes the InsightFace model suite for robust face detection and feature mapping.
- **Precision Masking**: Integrated **BiSeNet** segmentation allows for pixel-perfect inclusion/exclusion of facial features (eyes, mouth, skin) to preserve original expressions or fix occlusion artifacts.
- **Smart Enhancers**: Includes **CodeFormer** and **GFPGAN** pipelines for restoring facial details in low-resolution sources.
- **Performance Profiles**: One-click presets (**Cinema**, **Balanced**, **Fast**) to automatically tune execution threads and FP16 precision.

### 🛠 Technical Highlights

- **Faceset Management**: Create, save, and reuse custom collections of faces for batch processing.
- **Visual Face Selector**: Gallery-based interface to select specific target faces in multi-person scenes.
- **Lossless Output**: Support for high-bitrate video output and lossless image formats.
- **CLI & GUI**: Flexible usage for both casual users and automation scripts.

##  Installation

### Prerequisites

- **Operating System**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GeForce GTX 1000 series or newer (Recommended for optimal performance).
- **Drivers**: Latest NVIDIA Studio or Game Ready Drivers.

### Quick Start (Recommended)

We provide a one-click installer to handle all dependencies, including Python environments and CUDA libraries.

1.  **Clone the Repository**:

    ```bash
    git clone https://github.com/bmontes93/EXON-FACE.git
    cd EXON-FACE
    ```

2.  **Run the Installer**:
    Double-click `install.bat`. This script will:

    - Verify your system requirements.
    - Set up a localized Python environment.
    - Download necessary AI models (if available).
    - Install PyTorch with CUDA acceleration.

3.  **Launch**:
    Double-click `start_exon.bat` to open the Web UI.

### Manual Installation (Developers)

If you prefer managing your own environment:

```bash
# 1. Create a virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install Dependencies (Ensure CUDA toolkit 11.8 is installed)
pip install -r requirements.txt

# 3. Run the application
python run.py
```

##  Usage

1.  **Select Source**: Upload an image containing the face you want to copy.
2.  **Select Target**: Upload the image or video where you want to paste the face.
3.  **Refine**:
    - Use the **Face Management** tab to analyze the target media and select specific faces to swap.
    - Enable **Face Enhancer** (CodeFormer) if the source face is low quality.
    - Adjust **Masking** settings if you need to preserve the original eyes or mouth.
4.  **Process**: Click "Start" and watch the preview generation. Output is saved to the `output/` directory by default.

## 🏗 Technology Stack

- **Backend**: Python 3.10
- **Inference Engine**: ONNX Runtime (CUDA/DirectML/CPU execution providers)
- **Computer Vision**: OpenCV, InsightFace
- **Face Segmentation**: BiSeNet (Bilateral Segmentation Network)
- **GUI**: Gradio

##  Ethical Disclaimer and Legal Notice

**THIS SOFTWARE IS PROVIDED FOR EDUCATIONAL, ARTISTIC, AND RESEARCH PURPOSES ONLY.**

The developers of EXON FACE condemn the use of this technology for:

1.  Creating non-consensual sexual content (NCII).
2.  Identity theft, fraud, or impersonation.
3.  Spreading misinformation or defamation.

**User Responsibility**: By using this software, you agree that you retain full legal responsibility for the content you create. The authors assume no liability for misuse. We strongly recommend watermarking AI-generated content to maintain transparency.

##  Credits & Acknowledgements

This project builds upon the shoulders of giants in the open-source community:

- **InsightFace**: For the foundational face analysis models.
- **Roop**: For the original inspiration on single-image face swapping.
- **HuggingFace**: For hosting model weights and datasets.

---

<div align="center">
    Created and Maintained by <b>bryanmontesdev</b><br>
    <i>Pushing the boundaries of creative AI.</i>
</div>
