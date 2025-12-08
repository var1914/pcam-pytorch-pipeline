from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn as nn
import torch.optim as optim

import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

import os
import sys
import json
import logging

logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.data.data_loading import DataDownloader
from src.data.preprocessing import DataPreprocessor
from src.models.model import PCamModel
from src.training.training import ModelTrainer
from src.evaluation.evaluation import ModelEvaluator
from src.deployment.deployment import ModelDeployer

# MinIO and MLflow configuration
MINIO_CONFIG = {
    'endpoint': 'minio:9000',
    'access_key': 'admin',
    'secret_key': 'admin123',
    'secure': False,
    'bucket_name': 'ml-models'
}

MLFLOW_TRACKING_URI = 'http://mlflow:5000'

def data_preparation(**context):
    """
    Comprehensive data preparation task that:
    1. Downloads raw PCAM dataset
    2. Validates data quality
    3. Checks class balance
    4. Converts to parquet format
    5. Uploads to MinIO
    """
    try:
        download_data = DataDownloader()
        preprocessor = DataPreprocessor()
        data_root = '../data'

        # Step 1: Download and load raw datasets
        logger.info("Step 1: Downloading and loading raw datasets...")
        basic_transform = transforms.Compose([transforms.ToTensor()])

        train_dataset = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='train',
            download=True,
            transform=basic_transform
        )

        val_dataset = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='val',
            download=True,
            transform=basic_transform
        )

        # Step 2: Get dataset statistics
        logger.info("Step 2: Computing dataset statistics...")
        train_size, train_label_dist = download_data.get_data_stats()
        logger.info(f"Train dataset size: {train_size}")
        logger.info(f"Train label distribution: {train_label_dist}")

        # Step 3: Validate datasets
        logger.info("Step 3: Validating dataset quality...")
        train_validation = preprocessor.validate_dataset(train_dataset)
        val_validation = preprocessor.validate_dataset(val_dataset)

        # Step 4: Check class balance
        logger.info("Step 4: Checking class balance...")
        train_balance = preprocessor.check_class_balance(train_dataset)

        # Step 5: Create data loaders for parquet conversion
        logger.info("Step 5: Creating dataloaders for parquet conversion...")
        normalized_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset_normalized = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='train',
            download=False,
            transform=normalized_transform
        )

        val_dataset_normalized = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='val',
            download=False,
            transform=normalized_transform
        )

        train_loader = DataLoader(
            train_dataset_normalized,
            batch_size=32,
            shuffle=False,
        )

        val_loader = DataLoader(
            val_dataset_normalized,
            batch_size=32,
            shuffle=False,
        )

        # Step 6: Convert to parquet and upload to MinIO
        logger.info("Step 6: Converting to parquet and uploading to MinIO...")
        parquet_output_dir = '../data/parquet'

        download_data.convert_to_parquet_batches(
            train_loader,
            os.path.join(parquet_output_dir, 'train'),
            bucket_name='dataset'
        )

        download_data.convert_to_parquet_batches(
            val_loader,
            os.path.join(parquet_output_dir, 'val'),
            bucket_name='dataset'
        )

        # Push results to XCom (only paths, not data)
        context['task_instance'].xcom_push(key='data_root', value=data_root)
        context['task_instance'].xcom_push(key='parquet_train_dir', value=os.path.join(parquet_output_dir, 'train'))
        context['task_instance'].xcom_push(key='parquet_val_dir', value=os.path.join(parquet_output_dir, 'val'))
        context['task_instance'].xcom_push(key='train_size', value=train_size)
        context['task_instance'].xcom_push(key='train_balance', value=train_balance)

        logger.info("✓ Data preparation completed successfully!")

    except Exception as e:
        logger.error(f"Error in data preparation: {str(e)}")
        raise


def model_training(**context):
    """Train the model on prepared data."""
    try:
        # Pull data root from XCom
        data_root = context['task_instance'].xcom_pull(
            task_ids='data_preparation',
            key='data_root'
        )

        logger.info("Loading datasets for training...")
        download_data = DataDownloader()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='train',
            download=False,
            transform=transform
        )

        val_dataset = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='val',
            download=False,
            transform=transform
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Create model
        model = PCamModel(num_classes=2, pretrained=False)
        logger.info(f"Model created: ResNet50 with {model.num_classes} output classes")
        logger.info(f"Total parameters: {model.get_num_params():,}")
        logger.info(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")

        # Initialize trainer
        model_trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name='pcam_resnet50_model',
            lr=0.001,
            checkpoint_dir='./checkpoints'
        )

        # Train
        model_metrics = model_trainer.train(
            num_epochs=5,
            early_stopping_patience=3,
            save_best_only=True,
            log_to_mlflow=True
        )

        # Push paths to XCom
        context['task_instance'].xcom_push(key='best_model_path', value='./checkpoints/best_model.pt')
        context['task_instance'].xcom_push(key='data_root', value=data_root)

        logger.info("✓ Model training completed successfully!")

    except Exception as e:
        logger.error(f"Error in model training: {str(e)}")
        raise


def model_evaluation(**context):
    """Evaluate the trained model."""
    try:
        # Pull paths from XCom
        best_model_path = context['task_instance'].xcom_pull(
            task_ids='model_training',
            key='best_model_path'
        )
        data_root = context['task_instance'].xcom_pull(
            task_ids='model_training',
            key='data_root'
        )

        # Load model
        logger.info("Loading trained model...")
        model = PCamModel(num_classes=2, pretrained=False)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load validation data
        logger.info("Loading validation data...")
        download_data = DataDownloader()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        val_dataset = download_data.load_data(
            datasets.PCAM,
            data_root,
            split='val',
            download=False,
            transform=transform
        )
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Evaluate
        logger.info("Evaluating model...")
        evaluator = ModelEvaluator(
            model=model,
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name='pcam_evaluation'
        )

        eval_metrics = evaluator.evaluate(
            test_loader=val_loader,
            log_to_mlflow=True,
            save_visualizations=True
        )

        classification_rep = evaluator.get_classification_report(val_loader)
        logger.info(f"\nClassification Report:\n{classification_rep}")

        context['task_instance'].xcom_push(key='eval_metrics', value=eval_metrics)

        logger.info("✓ Model evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def model_deployment(**context):
    """Deploy model to production."""
    try:
        # Pull paths and metrics from XCom
        best_model_path = context['task_instance'].xcom_pull(
            task_ids='model_training',
            key='best_model_path'
        )
        eval_metrics = context['task_instance'].xcom_pull(
            task_ids='model_evaluation',
            key='eval_metrics'
        )

        # Load model
        logger.info("Loading model for deployment...")
        model = PCamModel(num_classes=2, pretrained=False)
        checkpoint = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        # Deploy
        logger.info("Deploying model...")
        deployer = ModelDeployer(
            minio_config=MINIO_CONFIG,
            mlflow_tracking_uri=MLFLOW_TRACKING_URI,
            mlflow_experiment_name='pcam_deployment'
        )

        version = datetime.now().strftime('%Y%m%d_%H%M%S')
        deployment_info = deployer.deploy_model(
            model=model,
            model_name='pcam_resnet50',
            version=version,
            metadata={
                'eval_metrics': eval_metrics,
                'deployment_date': datetime.now().isoformat(),
                'model_architecture': 'ResNet50',
                'num_classes': 2
            },
            register_to_mlflow=True
        )

        logger.info(f"Model deployed: {deployment_info}")
        context['task_instance'].xcom_push(key='deployment_info', value=deployment_info)

        logger.info("✓ Model deployment completed successfully!")

    except Exception as e:
        logger.error(f"Error in model deployment: {str(e)}")
        raise


# Define DAG
default_args = {
    'owner': 'varunrajput',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 8),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pcam_ml_pipeline',
    description='PCAM ML Pipeline with PyTorch, MLflow, and MinIO',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

# Define tasks
task_data_preparation = PythonOperator(
    task_id='data_preparation',
    python_callable=data_preparation,
    dag=dag,
    provide_context=True
)

task_model_training = PythonOperator(
    task_id='model_training',
    python_callable=model_training,
    dag=dag,
    provide_context=True
)

task_model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation,
    dag=dag,
    provide_context=True
)

task_model_deployment = PythonOperator(
    task_id='model_deployment',
    python_callable=model_deployment,
    dag=dag,
    provide_context=True
)

# Define task dependencies
task_data_preparation >> task_model_training >> task_model_evaluation >> task_model_deployment 