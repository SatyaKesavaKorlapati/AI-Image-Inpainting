# AI Image Inpainting & Object Removal System

An AI-powered image inpainting and object removal application built using LaMa and Gradio. The system enables mask-guided object removal and scene restoration through deep-learning-based image completion.

The project provides an interactive Gradio interface for removing unwanted objects from images while preserving visual consistency in the surrounding scene.

---

## Overview

This project combines:

- LaMa-based image inpainting
- Mask-guided object removal
- Deep-learning image restoration
- Interactive Gradio UI
- Modular inference pipeline
- Local CPU/GPU execution

The application allows users to:
- upload an image
- mark unwanted objects
- generate restored outputs using AI-based inpainting

---

## Features

- AI-powered object removal
- LaMa image inpainting pipeline
- Interactive Gradio web interface
- Local inference support
- CPU and GPU compatibility
- Modular backend architecture
- Lightweight deployment setup
- Automatic model handling utilities

---

## Technologies Used

### AI / Deep Learning
- PyTorch
- LaMa
- OpenCV

### Frontend
- Gradio

### Backend
- Python

---

## Project Structure

```text
AI-Image-Inpainting/
│
├── gradio_lama_app.py
├── README.md
├── requirements.txt
├── LICENSE
│
└── engine/
    ├── api.py
    ├── const.py
    ├── download.py
    ├── helper.py
    ├── installer.py
    ├── model_manager.py
    ├── runtime.py
    ├── schema.py
    │
    ├── file_manager/
    │
    └── model/
        ├── base.py
        ├── lama.py
        └── utils.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/SatyaKesavaKorlapati/AI-Image-Inpainting.git
cd AI-Image-Inpainting
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the Gradio application:

```bash
python gradio_lama_app.py
```

After launching, open the local Gradio URL displayed in the terminal.

---

## Workflow

1. Upload an image
2. Draw a mask over unwanted regions
3. Run AI inpainting
4. Generate restored image output

The system fills masked regions using deep-learning-based scene understanding and texture synthesis.

---

## Core Components

### LaMa Inpainting Model

The application uses LaMa for:
- object removal
- image restoration
- mask-guided scene completion

### Gradio Interface

The frontend provides:
- image upload
- mask drawing
- interactive inference
- restored image preview

### Modular Backend

The backend contains:
- model management
- runtime configuration
- inference utilities
- file management helpers

---

## Use Cases

- Object removal
- Photo cleanup
- Scene restoration
- Background refinement
- Image editing workflows
- AI-assisted content generation

---

## Future Improvements

- Drag-and-drop mask editor
- Multi-model selection
- HD inpainting support
- Real-time preview generation
- Batch image processing
- Diffusion-based refinement

---

