"""
ResNet18 Ensemble Training with K-Fold Cross-Validation on Combined Train+Val Sets
and Final Test Set Evaluation for Retinal Image Classification

This script:
1. Combines train.csv and val.csv for k-fold cross-validation
2. Trains an ensemble of ResNet18 models using k-fold cross-validation
3. Evaluates the trained ensemble on the separate test.csv
No data augmentation is applied.
"""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()
load_dotenv(override=True)


# Configuration
class Config:
    data_dir: Path = Path(os.getenv("PAPILA_PATH", "."))
    train_csv = "split/new_train.csv"  # Path to training dataset CSV
    val_csv = "split/new_val.csv"  # Path to validation dataset CSV
    test_csv = "split/new_test.csv"  # Path to test dataset CSV
    img_dir = "FundusImages/"  # Directory containing the images
    num_classes = 2  # Binary classification (0, 1)
    num_folds = 3  # Number of folds for cross-validation
    epochs = 10  # Number of training epochs
    batch_size = 64  # Batch size for training/validation
    learning_rate = 1e-4  # Learning rate
    weight_decay = 1e-5  # Weight decay for regularization
    img_size = 224  # Image size for ResNet18
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "saved_models/"  # Directory to save models


config = Config()

# Create directory for saving models if it doesn't exist
Path(config.model_save_path).mkdir(parents=True, exist_ok=True)


# Custom Dataset class for Retinal Images
class RetinalImageDataset(Dataset):
    def __init__(self, dataframe, data_dir: Path, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = data_dir / img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.img_dir / self.dataframe.iloc[idx]["Path"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.dataframe.iloc[idx]["Diagnosis"]
        return image, label


# Define data transformations (simplified, no augmentation)
data_transforms = transforms.Compose(
    [
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Function to create a ResNet18 model with pre-trained weights
def create_model():
    model = models.resnet18(weights="IMAGENET1K_V1")
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, config.num_classes)
    return model


# Training function for one epoch
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true = []
    y_pred = []

    for batch_inputs, batch_labels in dataloader:
        inputs = batch_inputs.to(device)
        labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(y_true, y_pred)

    return epoch_loss, epoch_acc


# Validation/Test function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for batch_inputs, batch_labels in dataloader:
            inputs = batch_inputs.to(device)
            labels = batch_labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Get probability scores for ROC AUC
            scores = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()

            running_loss += loss.item() * inputs.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(scores)

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5,
    }

    return epoch_loss, metrics, y_true, y_scores


# Function to prepare data (combine train and val, load test)
def prepare_data():
    # Load datasets
    df_train = pd.read_csv(config.data_dir / config.train_csv)
    df_val = pd.read_csv(config.data_dir / config.val_csv)
    df_test = pd.read_csv(config.data_dir / config.test_csv)
    # Combine train and val for k-fold cross-validation
    df_combined = pd.concat([df_train, df_val], ignore_index=True)
    return df_combined, df_test


# Function to create test dataloader
def create_test_dataloader(df_test: pd.DataFrame):
    test_dataset = RetinalImageDataset(
        dataframe=df_test,
        data_dir=config.data_dir,
        img_dir=config.img_dir,
        transform=data_transforms,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    return test_loader


# Main training function with K-Fold Cross-Validation
def train_kfold_ensemble():
    # Prepare data
    df_combined, df_test = prepare_data()

    # Create test dataloader
    test_loader = create_test_dataloader(df_test)

    # Initialize K-Fold
    kfold = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=42)

    # Initialize lists to store results
    fold_results = []
    ensemble_models = []

    # Iterate through folds
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(df_combined, df_combined["Diagnosis"])
    ):
        print(f"\nTraining Fold {fold+1}/{config.num_folds}")

        # Split data
        train_df = df_combined.iloc[train_idx].reset_index(drop=True)
        val_df = df_combined.iloc[val_idx].reset_index(drop=True)

        # Create datasets
        train_dataset = RetinalImageDataset(
            dataframe=train_df,
            data_dir=config.data_dir,
            img_dir=config.img_dir,
            transform=data_transforms,
        )

        val_dataset = RetinalImageDataset(
            dataframe=val_df,
            data_dir=config.data_dir,
            img_dir=config.img_dir,
            transform=data_transforms,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )

        val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4
        )

        # Create model
        model = create_model()
        model = model.to(config.device)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Initialize best model weights and metrics
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_auc = 0.0
        best_epoch = 0

        # Training loop
        for epoch in range(config.epochs):
            print(f"Epoch {epoch+1}/{config.epochs}")

            # Train for one epoch
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, config.device
            )

            # Validate
            val_loss, val_metrics, _, _ = evaluate(
                model, val_loader, criterion, config.device
            )

            # Update learning rate
            scheduler.step(val_loss)

            # Print epoch results
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val AUC: {val_metrics['auc']:.4f}"
            )

            # Save best model
            if val_metrics["auc"] > best_val_auc:
                best_val_auc = val_metrics["auc"]
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"New best model saved with AUC: {best_val_auc:.4f}")

        # Load best model weights
        model.load_state_dict(best_model_wts)

        # Save the model
        model_save_file = Path(config.model_save_path) / f"resnet18_fold_{fold+1}.pth"
        torch.save(model.state_dict(), model_save_file)
        print(f"Best model from epoch {best_epoch+1} saved to {model_save_file}")

        # Final validation
        _, final_metrics, y_true, y_scores = evaluate(
            model, val_loader, criterion, config.device
        )

        # Store results
        fold_results.append(
            {
                "fold": fold + 1,
                "metrics": final_metrics,
                "val_indices": val_idx,
            }
        )

        # Add model to ensemble
        ensemble_models.append(model)

    return ensemble_models, fold_results, test_loader


# Function to make ensemble predictions
def ensemble_predict(models, dataloader, device):
    all_scores = []
    y_true = []

    # Make predictions with each model
    for model in models:
        model.eval()
        fold_scores = []

        with torch.no_grad():
            for batch_inputs, batch_labels in dataloader:
                inputs = batch_inputs.to(device)

                outputs = model(inputs)
                scores = nn.Softmax(dim=1)(outputs)[:, 1].cpu().numpy()
                fold_scores.extend(scores)

                if len(all_scores) == 0:
                    y_true.extend(batch_labels.numpy())

        all_scores.append(fold_scores)

    # Average predictions from all models
    ensemble_scores = np.mean(np.array(all_scores), axis=0)
    ensemble_preds = (ensemble_scores > 0.5).astype(int)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, ensemble_preds),
        "precision": precision_score(y_true, ensemble_preds, zero_division=0),
        "recall": recall_score(y_true, ensemble_preds, zero_division=0),
        "f1": f1_score(y_true, ensemble_preds, zero_division=0),
        "auc": roc_auc_score(y_true, ensemble_scores)
        if len(np.unique(y_true)) > 1
        else 0.5,
    }

    return metrics


# Function to load trained models
def load_ensemble_models(num_folds=5):
    models = []
    for i in range(num_folds):
        model_path = Path(config.model_save_path) / f"resnet18_fold_{i+1}.pth"
        model = create_model()
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model = model.to(config.device)
        model.eval()
        models.append(model)
    return models


if __name__ == "__main__":
    print(f"Using device: {config.device}")

    # Start training timer
    start_time = time.time()

    # Train ensemble with k-fold cross-validation on combined train+val
    print(
        "\nTraining ensemble using k-fold cross-validation on combined train+val datasets..."
    )
    ensemble_models, fold_results, test_loader = train_kfold_ensemble()

    # Calculate training time
    training_time = time.time() - start_time
    print(f"\nTotal training time: {training_time/60:.2f} minutes")

    # Calculate average performance across folds
    print("\nAverage Performance Across All Folds:")
    avg_acc = np.mean([fold["metrics"]["accuracy"] for fold in fold_results])
    avg_precision = np.mean([fold["metrics"]["precision"] for fold in fold_results])
    avg_recall = np.mean([fold["metrics"]["recall"] for fold in fold_results])
    avg_f1 = np.mean([fold["metrics"]["f1"] for fold in fold_results])
    avg_auc = np.mean([fold["metrics"]["auc"] for fold in fold_results])

    print(f"Accuracy: {avg_acc:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"AUC: {avg_auc:.4f}")

    # Evaluate ensemble on test set
    print("\nEvaluating ensemble on separate test set...")
    test_metrics = ensemble_predict(ensemble_models, test_loader, config.device)
    print("Test Ensemble Performance:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print(f"AUC: {test_metrics['auc']:.4f}")
