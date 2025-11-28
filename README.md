# PCAM Baseline Model Training

## Overview
This is a baseline model for the PCAM (PatchCamelyon) dataset using ResNet50. The goal is to build a simple, working model that we can later integrate into a full ML pipeline with Airflow, MLflow, and Kubernetes.

## Dataset
The PCAM dataset is a binary classification task on 96x96 histopathology image patches. The task is to identify whether a patch contains tumor tissue or not.

- Train samples: 131,072
- Val samples: 32,768
- Test samples: 32,768

For quick iteration, we use a subset of 10,000 training samples.

## Model Architecture
- **Base model**: ResNet50 (pretrained weights not used initially)
- **Input**: 96x96 RGB images
- **Output**: 2 classes (tumor / non-tumor)
- **Training**: Binary cross-entropy loss with Adam optimizer

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

This will:
1. Download and load the PCAM dataset
2. Create a 10k sample subset for faster training
3. Train ResNet50 for 5 epochs
4. Evaluate on the full test set
5. Save the trained model
6. Generate training history plots

## Results
After 5 epochs on 10k training samples:
- Train Accuracy: ~82%
- Val Accuracy: ~83%
- Test Accuracy: ~83%

## What's Next
This baseline is the foundation. Next steps:
1. Add MLflow to log experiments and metrics
2. Create separate data, model, and training modules
3. Build an Airflow DAG to orchestrate the pipeline
4. Deploy the model using KServe
5. Scale to full dataset and optimize hyperparameters

## Files
- `pcam_baseline.py` - Main training script
- `pcam_baseline_model.pth` - Saved model weights
- `training_history.png` - Loss and accuracy plots
