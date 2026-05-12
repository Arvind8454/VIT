# Vision Transformer (ViT) Based Image Recognition with Explainable AI and Self-Supervised Pretraining

## Project Overview
This project provides a research-grade, end-to-end image recognition pipeline based on Vision Transformers (ViT). It includes self-supervised pretraining (MAE), supervised fine-tuning, explainable AI visualizations (Grad-CAM and attention rollout), and a Streamlit user interface for interactive inference.

## Problem Statement
Build an image recognition system that learns robust visual representations using self-supervised pretraining, then fine-tunes for supervised classification. Provide transparency via explainable AI methods, and expose the system through a lightweight web UI.

## Model Architecture
- Vision Transformer (ViT)
- Patch embedding + Transformer encoder blocks + classification head
- Pretrained initialization from MAE (Masked Autoencoder)

## Self-Supervised Learning Approach
We use Masked Autoencoder (MAE) pretraining to learn general visual representations by masking patches and reconstructing missing content. The pretrained encoder weights are then used to initialize the supervised ViT classifier.

## Explainable AI Methods
- Grad-CAM (applied to patch embedding projection)
- Attention rollout visualization across transformer layers
- Heatmap overlay on input image

## Installation
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## What You Need
- Python 3.10+ (Windows, macOS, or Linux)
- Optional: NVIDIA GPU with CUDA for faster training
- ~5-10 GB free disk space for datasets and checkpoints

## Dataset Setup
Default dataset: CIFAR-10 (auto-downloaded)

```bash
# data will be downloaded into vit_explainable_ai_project/data
```

## Self-Supervised Pretraining
```bash
python -m self_supervised.mae_pretrain \
  --dataset cifar10 \
  --data_dir data \
  --output_dir experiments/checkpoints \
  --epochs 10 \
  --batch_size 64
```

Fast smoke test:
```bash
python -m self_supervised.mae_pretrain \
  --dataset cifar10 \
  --data_dir datasets \
  --output_dir experiments/checkpoints \
  --epochs 1 \
  --batch_size 32 \
  --num_workers 0 \
  --max_steps 10
```

## Supervised Training
```bash
python -m training.train \
  --dataset cifar10 \
  --data_dir data \
  --output_dir experiments/checkpoints \
  --epochs 10 \
  --batch_size 64 \
  --use_mae_weights experiments/checkpoints/mae_pretrained
```

Fast smoke test:
```bash
python -m training.train \
  --dataset cifar10 \
  --data_dir datasets \
  --output_dir experiments/checkpoints \
  --epochs 1 \
  --batch_size 16 \
  --num_workers 0 \
  --use_mae_weights experiments/checkpoints/mae_pretrained \
  --max_steps 2 \
  --max_val_steps 2 \
  --skip_test
```

## Running Inference
```bash
python -m inference.predict \
  --model_dir experiments/checkpoints/best_model \
  --image_path path\to\image.jpg
```

## Running the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

## Running the Full Flask + Streamlit App
```bash
python app.py
```
Flask runs at `http://localhost:5000` and Streamlit runs at `http://localhost:8501`.

## Public Access with NPort
### 1) Install NPort
```bash
npm install -g nport
```

### 2) Verify installation
```bash
nport --version
```

### 3) Start Flask + tunnel
```bash
python scripts/start_tunnel.py
```

### 4) Manual tunnel command
```bash
nport 5000
```

## Example Outputs
- Predicted label and confidence score
- Grad-CAM heatmap overlay
- Attention rollout visualization

## Notes
- Mixed precision and GPU training are supported via flags in training scripts.
- Confusion matrix is saved to experiments/logs/ during training.
- Class names are stored at `experiments/checkpoints/best_model/class_names.txt` for inference.
- Check CUDA availability:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```
