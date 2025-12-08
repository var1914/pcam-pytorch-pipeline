# PyTorch ML Pipeline Framework

A production-ready ML pipeline framework for training PyTorch models on any image classification dataset. Built with Apache Airflow for orchestration, MLflow for experiment tracking, and MinIO for object storage.

## What is this?

This is a complete, working ML pipeline that you can use to train models on any PyTorch-compatible image dataset. Whether you're working with CIFAR-10, ImageNet, medical images, or your own custom dataset, this framework handles the entire workflow from data loading to model deployment.

The pipeline is designed to be modular and easy to customize. You can swap datasets, models, optimizers, and configurations without changing the core pipeline logic.

## Key Features

- **Dataset Agnostic**: Works with any `torchvision.datasets` or custom PyTorch `Dataset`
- **Modular Design**: Easily swap models, optimizers, loss functions
- **Experiment Tracking**: Built-in MLflow integration for tracking experiments
- **Cloud Storage**: MinIO S3-compatible storage for datasets and models
- **Pipeline Orchestration**: Apache Airflow DAGs for production workflows
- **Comprehensive Evaluation**: Automatic metrics, visualizations, and reports
- **Production Ready**: Model versioning, registry, and deployment tools

## How It Works

The pipeline has four main stages:

```
Data Preparation → Model Training → Model Evaluation → Model Deployment
```

Each stage is independent and communicates through MinIO storage and XCom. Data never gets passed directly between tasks, only file paths.

## Supported Datasets

This framework works with any dataset from `torchvision.datasets` including:
- CIFAR-10, CIFAR-100
- ImageNet
- MNIST, Fashion-MNIST
- SVHN
- PCAM (PatchCamelyon)
- STL10
- Places365
- And many more...

You can also easily add your own custom datasets by implementing PyTorch's `Dataset` interface.

## Project Structure

```
pytorch-ml-pipeline/
├── src/
│   ├── data/
│   │   ├── data_loading.py          # Dataset loading and parquet conversion
│   │   └── preprocessing.py         # Data validation and augmentation
│   ├── models/
│   │   └── model.py                 # Model architecture wrapper
│   ├── training/
│   │   └── training.py              # Training loop with MLflow and MinIO
│   ├── evaluation/
│   │   └── evaluation.py            # Metrics, confusion matrix, ROC curves
│   ├── deployment/
│   │   └── deployment.py            # Model deployment and versioning
│   ├── minio/
│   │   └── minio_init.py            # MinIO client setup
│   └── airflow/
│       └── dags.py                  # Airflow DAG definition
├── data/                            # Downloaded datasets go here
├── checkpoints/                     # Model checkpoints
├── requirements.txt
└── README.md
```

## Installation

### Requirements
- Python 3.8 or higher
- Docker (for running MinIO, MLflow, and Airflow)

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

The requirements.txt includes:
```
torch==2.8.0
torchvision==0.23.0
apache-airflow==3.1.2
pandas==2.3.3
h5py==3.15.1
matplotlib==3.10.7
mlflow==3.5.1
minio==7.2.20
scikit-learn
seaborn
```

## Quick Start

### Example 1: Train on CIFAR-10

Here's a simple example to train a ResNet18 model on CIFAR-10:

```python
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

from src.data.data_loading import DataDownloader
from src.training.training import ModelTrainer

# Define data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load CIFAR-10 dataset
downloader = DataDownloader()
train_dataset = downloader.load_data(
    datasets.CIFAR10, './data', split='train', download=True, transform=transform
)
val_dataset = downloader.load_data(
    datasets.CIFAR10, './data', split='test', download=True, transform=transform
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Create model
model = models.resnet18(num_classes=10)

# Configure MinIO and MLflow
minio_config = {
    'endpoint': 'localhost:9000',
    'access_key': 'admin',
    'secret_key': 'admin123',
    'secure': False,
    'bucket_name': 'ml-models'
}

# Train the model
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    minio_config=minio_config,
    mlflow_tracking_uri='http://localhost:5000',
    mlflow_experiment_name='cifar10_resnet18',
    lr=0.001
)

history = trainer.train(num_epochs=10, early_stopping_patience=3)
print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
```

### Example 2: Using Custom Datasets

You can use your own dataset by implementing PyTorch's Dataset class:

```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyCustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        label = int(self.image_files[idx].split('_')[0])  # Your label logic

        if self.transform:
            image = self.transform(image)

        return image, label

# Use it with the pipeline
custom_dataset = MyCustomDataset('./my_images', transform=transform)
train_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Training works the same way
trainer = ModelTrainer(model, train_loader, val_loader, ...)
history = trainer.train(num_epochs=20)
```

## Infrastructure Setup

You need to run three services before using the pipeline:

### 1. MinIO (Object Storage)

MinIO stores your datasets (as parquet files) and trained models.

```bash
docker run -d \
  -p 9000:9000 \
  -p 9001:9001 \
  --name minio \
  -e "MINIO_ROOT_USER=admin" \
  -e "MINIO_ROOT_PASSWORD=admin123" \
  minio/minio server /data --console-address ":9001"
```

Access the MinIO console at http://localhost:9001 with username `admin` and password `admin123`.

### 2. MLflow (Experiment Tracking)

MLflow tracks your experiments, metrics, and model artifacts.

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlflow-artifacts \
  --host 0.0.0.0 \
  --port 5000
```

Access the MLflow UI at http://localhost:5000

### 3. Apache Airflow (Optional, for pipeline orchestration)

Airflow orchestrates the entire pipeline workflow.

```bash
export AIRFLOW_HOME=$(pwd)
airflow db init
airflow users create \
  --username admin \
  --password admin \
  --role Admin \
  --email admin@example.com

airflow webserver -p 8080 &
airflow scheduler &
```

Access the Airflow UI at http://localhost:8080 with username `admin` and password `admin`.

## Main Components

### Data Loading

The `DataDownloader` class handles dataset loading and conversion to parquet format:

```python
from src.data.data_loading import DataDownloader

downloader = DataDownloader()

# Load any torchvision dataset
dataset = downloader.load_data(
    dataset_class=datasets.MNIST,
    root_path='./data',
    split='train',
    download=True,
    transform=your_transform
)

# Get dataset statistics
size, distribution = downloader.get_data_stats()

# Convert batches to parquet and upload to MinIO
downloader.convert_to_parquet_batches(dataloader, output_dir, bucket_name='dataset')
```

### Data Preprocessing

The `DataPreprocessor` validates your data and provides augmentation options:

```python
from src.data.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()

# Validate dataset quality
validation = preprocessor.validate_dataset(dataset)

# Check if classes are balanced
balance = preprocessor.check_class_balance(dataset)

# Get augmentation transforms
transform = preprocessor.get_data_augmentation_transforms(
    image_size=224,
    augmentation_level='medium'  # Options: 'light', 'medium', 'heavy'
)

# Create weighted sampler for imbalanced datasets
sampler = preprocessor.create_weighted_sampler(dataset)
```

### Model Training

The `ModelTrainer` handles the training loop with automatic logging to MLflow and saving to MinIO:

```python
from src.training.training import ModelTrainer

trainer = ModelTrainer(
    model=your_model,
    train_loader=train_loader,
    val_loader=val_loader,
    minio_config=minio_config,
    mlflow_tracking_uri='http://localhost:5000',
    mlflow_experiment_name='my_experiment',
    lr=0.001,
    checkpoint_dir='./checkpoints'
)

history = trainer.train(
    num_epochs=20,
    early_stopping_patience=5,
    save_best_only=True,
    log_to_mlflow=True
)
```

### Model Evaluation

The `ModelEvaluator` generates comprehensive metrics and visualizations:

```python
from src.evaluation.evaluation import ModelEvaluator

evaluator = ModelEvaluator(
    model=trained_model,
    minio_config=minio_config,
    mlflow_tracking_uri='http://localhost:5000'
)

# Evaluate and log everything to MLflow
metrics = evaluator.evaluate(
    test_loader=test_loader,
    log_to_mlflow=True,
    save_visualizations=True
)

# Get detailed classification report
report = evaluator.get_classification_report(test_loader)

# Get confusion matrix
confusion_matrix = evaluator.get_confusion_matrix(test_loader)
```

### Model Deployment

The `ModelDeployer` handles model versioning and deployment:

```python
from src.deployment.deployment import ModelDeployer

deployer = ModelDeployer(
    minio_config=minio_config,
    mlflow_tracking_uri='http://localhost:5000'
)

# Deploy a new model version
deployment_info = deployer.deploy_model(
    model=model,
    model_name='my_classifier',
    version='v1.0.0',
    metadata={'accuracy': 0.95, 'dataset': 'CIFAR10'},
    register_to_mlflow=True
)

# Load a deployed model
loaded_model = deployer.load_model_from_minio(
    model_name='my_classifier',
    version='v1.0.0'
)
```

## Using the Airflow Pipeline

The Airflow DAG orchestrates the entire workflow automatically. To configure it for your dataset:

1. Edit `src/airflow/dags.py` configuration (around line 34):

```python
# Change these for your dataset
DATASET = datasets.CIFAR10  # Your dataset class
NUM_CLASSES = 10            # Number of output classes
IMAGE_SIZE = 32             # Image size
DATA_ROOT = '../data'       # Where to store data
```

2. Copy the DAG file to Airflow:
```bash
cp src/airflow/dags.py $AIRFLOW_HOME/dags/
```

3. Open Airflow UI at http://localhost:8080

4. Find the `pcam_ml_pipeline` DAG and toggle it ON

5. Click "Trigger DAG" to start the pipeline

The pipeline runs these four tasks in sequence:
1. **data_preparation**: Downloads dataset, validates quality, converts to parquet, uploads to MinIO
2. **model_training**: Trains the model, logs metrics to MLflow, saves checkpoints
3. **model_evaluation**: Generates evaluation metrics, confusion matrix, ROC curves
4. **model_deployment**: Deploys model to MinIO production storage and registers in MLflow

## Monitoring

### MLflow Dashboard

Open http://localhost:5000 to view:
- All experiment runs
- Training metrics (loss, accuracy)
- Model parameters and hyperparameters
- Saved model artifacts

You can compare different runs side by side and see which configurations work best.

### MinIO Console

Open http://localhost:9001 (login: admin/admin123) to view:
- Stored datasets in parquet format
- Trained model files
- Model metadata

### Airflow Dashboard

Open http://localhost:8080 to monitor:
- DAG execution status
- Task logs and errors
- Task duration and performance
- Inter-task communication (XCom)

## Customization

### Using Different Models

You can use any PyTorch model architecture:

```python
from torchvision import models

# ResNet family
model = models.resnet18(num_classes=10)
model = models.resnet50(num_classes=10)

# EfficientNet
model = models.efficientnet_b0(num_classes=10)

# Vision Transformer
model = models.vit_b_16(num_classes=10)

# MobileNet
model = models.mobilenet_v3_large(num_classes=10)

# The training code remains the same
trainer = ModelTrainer(model, train_loader, val_loader, ...)
```

### Custom Loss Functions and Optimizers

```python
import torch.nn as nn
import torch.optim as optim

# Custom loss
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Custom optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Pass to trainer
trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    ...
)
```

## Common Issues

### Dataset download fails

Some datasets are large and may timeout. Increase the socket timeout:

```python
import socket
socket.setdefaulttimeout(600)  # 10 minutes
```

### Out of memory errors

Reduce the batch size:

```python
train_loader = DataLoader(dataset, batch_size=16)  # Instead of 32 or 64
```

### Training is slow

Increase the number of data loading workers:

```python
train_loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,      # More parallel workers
    pin_memory=True     # Faster GPU transfer
)
```

### MinIO connection fails

Check if MinIO is running:

```bash
docker ps | grep minio
docker logs minio
docker restart minio  # If needed
```

### MLflow UI not accessible

Check if MLflow server is running:

```bash
ps aux | grep mlflow
# Restart if needed
mlflow server --host 0.0.0.0 --port 5000
```

## What's Next

This framework is a starting point. Here are some ideas for extending it:

- Add Docker Compose for one-command infrastructure setup
- Implement hyperparameter tuning with Optuna or Ray Tune
- Add distributed training support
- Create a model serving API with FastAPI
- Deploy to Kubernetes
- Add CI/CD pipeline for automated testing
- Implement A/B testing framework
- Add model monitoring and drift detection
- Support for object detection and segmentation tasks
- Integrate data versioning with DVC

## Contributing

This is an open framework designed to be extensible. Feel free to contribute by:
- Adding support for new datasets
- Implementing new model architectures
- Improving documentation
- Reporting bugs or suggesting features

## License

MIT License

## Acknowledgments

This framework builds on:
- PyTorch and Torchvision for deep learning
- Apache Airflow for workflow orchestration
- MLflow for experiment tracking
- MinIO for object storage
