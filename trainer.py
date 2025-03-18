import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, bce_weight=1, 
                 criterion=None, optimizer=None, scheduler=None, device=None):
        """
        Trainer class for time series classification models
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Use GPU if available
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([bce_weight]).to(self.device))
        
        # Default to Adam if not provided
        self.optimizer = optimizer if optimizer is not None else optim.Adam(model.parameters(), lr=0.001)
        
        # Optional scheduler
        self.scheduler = scheduler
        
        # Store training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train(self, num_epochs, clip_grad_norm=None):
        """
        Train the model
        
        Args:
            num_epochs (int): Number of epochs to train
            clip_grad_norm (float, optional): Max norm for gradient clipping
            
        Returns:
            dict: Training history
        """
        print(f"Training on {self.device}")
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_predictions = []
            train_targets = []
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for features, targets in progress_bar:
                features, targets = features.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                
                # Optional gradient clipping
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                
                # Update parameters
                self.optimizer.step()
                
                # Store loss
                train_loss += loss.item() * features.size(0)
                
                # Store predictions and targets for metrics
                predictions = (outputs > 0.5).float()
                train_predictions.extend(predictions.cpu().numpy())
                train_targets.extend(targets.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item())
            
            # Compute average training loss
            train_loss /= len(self.train_loader.dataset)
            train_acc = accuracy_score(train_targets, train_predictions)
            
            # Validation phase
            val_loss, val_acc, val_predictions, val_targets = self.evaluate(self.val_loader)
            
            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Store metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                
            # Print epoch results
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
        return self.history
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a given data loader
        
        Args:
            data_loader: DataLoader to evaluate on
            
        Returns:
            tuple: (loss, accuracy, predictions, targets)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in data_loader:
                features, targets = features.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                
                # Store loss
                total_loss += loss.item() * features.size(0)
                
                # Store predictions and targets for metrics
                predictions = (outputs > 0.5).float()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Compute average loss and accuracy
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = accuracy_score(all_targets, all_predictions)
        
        return avg_loss, accuracy, all_predictions, all_targets
    
    def test(self, detailed=True):
        """
        Test the model on the test set
        
        Args:
            detailed (bool): Whether to print detailed metrics
            
        Returns:
            dict: Test results
        """
        test_loss, test_acc, predictions, targets = self.evaluate(self.test_loader)
        
        results = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'predictions': predictions,
            'targets': targets
        }
        
        if detailed:
            precision = precision_score(targets, predictions, zero_division=0)
            recall = recall_score(targets, predictions, zero_division=0)
            f1 = f1_score(targets, predictions, zero_division=0)
            mcc = matthews_corrcoef(targets, predictions)
            
            results.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mcc': mcc
            })
            
            print("\nTest Results:")
            print(f"Loss: {test_loss:.4f}")
            print(f"Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            print("\nClassification Report:")
            print(classification_report(targets, predictions, zero_division=0))
        
        return results
    
    def plot_history(self):
        """
        Plot the training history
        """
        if not self.history['train_loss']:
            print("No training history to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Validation')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Validation')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path):
        """
        Save the model to the given path
        
        Args:
            path (str): Path to save the model
        """
        self.model.save(path)
