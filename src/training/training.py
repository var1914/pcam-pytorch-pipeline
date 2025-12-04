from typing import Any, Callable, Dict, Optional, Tuple, Type
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

from mlflow.tracking import MlflowClient



class TrainModel():
    """
    Trainer class for PyTorch models with built-in validation and metric tracking.
    
    Handles the complete training loop including forward pass, backpropagation,
    validation, and metric logging for supervised learning tasks.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[str] = None,
        lr: float = 0.001,
        checkpoint_dir: Optional[Path] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            criterion: Loss function (default: CrossEntropyLoss).
            optimizer: Optimizer instance (default: Adam with specified lr).
            device: Device to train on. Auto-detects if None.
            lr: Learning rate for optimizer (ignored if optimizer provided).
            checkpoint_dir: Directory to save model checkpoints.
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = lr
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize criterion and optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.Adam(
            self.model.parameters(), 
            lr=self.lr
        )

        # Move model to device
        self.model.to(self.device)
        
        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        
    def _train_epoch(self) -> Tuple[float, float]:
        """
        Execute one training epoch.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_bar = tqdm(self.train_loader, desc="Training", leave=False)
        

            
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)


            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = running_loss / len(self.train_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy

    @torch.no_grad()
    def _validate_epoch(self) -> Tuple[float, float]:
        """
        Execute one validation epoch.
        
        Returns:
            Tuple of (average_loss, accuracy).
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
        
        for inputs, labels in val_bar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            val_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
        
        avg_loss = running_loss / len(self.val_loader.dataset)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        early_stopping_patience: Optional[int] = None,
        save_best_only: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train.
            early_stopping_patience: Stop if validation loss doesn't improve
                                    for this many epochs. None disables.
            save_best_only: Only save checkpoints when validation improves.
            
        Returns:
            Dictionary containing training history with keys:
            'train_loss', 'train_acc', 'val_loss', 'val_acc'.
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 60)
            
            # Training phase
            train_loss, train_acc = self._train_epoch()
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch()
            
            # Store metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
            
            # Save checkpoint
            if self.checkpoint_dir:
                is_best = val_loss < best_val_loss
                
                if is_best:
                    best_val_loss = val_loss
                    patience_counter = 0
                    print(f"✓ New best validation loss: {val_loss:.4f}")
                
                if not save_best_only or is_best:
                    self._save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping check
            if early_stopping_patience:
                if val_loss >= best_val_loss:
                    patience_counter += 1
                    print(f"No improvement for {patience_counter} epoch(s)")
                    
                    if patience_counter >= early_stopping_patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        break
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    def _save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        is_best: bool = False
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> Dict:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Checkpoint dictionary.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch'] + 1}")
        print(f"Validation loss: {checkpoint['val_loss']:.4f}")
        
        return checkpoint


