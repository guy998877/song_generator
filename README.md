# Song Generator

Python project for training a song/lyrics generation model and then generating new text from it.

This repo is organized as a lightweight pipeline:
1) preprocess / clean text data  
2) train a model  
3) generate new samples from a prompt or seed

---

## Project structure

- `data_processing.py`  
  Utilities for loading, cleaning, and preparing the dataset for training.

- `model.py`  
  Model definition (the architecture lives here).

- `train.py`  
  Training script: builds the dataset, trains the model, and saves artifacts (weights, tokenizer/vocab, etc.).

- `generate.py`  
  Inference script: loads the saved artifacts and generates new text.

- `main.py`  
  Convenience entry point (typically used to run train/generate from one place).

- `requirements.txt`  
  Python dependencies.

---

## Setup

### 1) Clone and install
```bash
git clone https://github.com/guy998877/song_generator.git
cd song_generator
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
