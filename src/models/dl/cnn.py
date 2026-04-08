import os
import cv2
import time
import random
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ====================== Simple CNN Model ======================
class SimpleCNN(nn.Module):
    """
    A flexible CNN model for grayscale image patches.
    Adjusts depth according to patch size.
    """
    def __init__(self, patch_size: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # Dynamically build convolutional layers based on patch size
        if patch_size <= 16:
            self.convs = nn.ModuleList([nn.Conv2d(1, 16, 3, padding=1)])
            self.pools = nn.ModuleList([nn.MaxPool2d(2, 2)])
        elif patch_size <= 64:
            self.convs = nn.ModuleList([
                nn.Conv2d(1, 16, 3, padding=1),
                nn.Conv2d(16, 32, 3, padding=1)
            ])
            self.pools = nn.ModuleList([nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)])
        else:
            self.convs = nn.ModuleList([
                nn.Conv2d(1, 16, 3, padding=1),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.Conv2d(32, 64, 3, padding=1)
            ])
            self.pools = nn.ModuleList([
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2),
                nn.MaxPool2d(2, 2)
            ])

        # Calculate flattened feature size for fully connected layer
        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size)
            for conv, pool in zip(self.convs, self.pools):
                dummy = pool(F.relu(conv(dummy)))
            fc_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Linear(fc_size, num_classes)

    def forward(self, x):
        for conv, pool in zip(self.convs, self.pools):
            x = pool(F.relu(conv(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ====================== Data Preprocessing ======================
def convert(img_path: str, patch_size: int) -> torch.Tensor:
    """
    Read and preprocess a grayscale image.

    Args:
        img_path (str): Path to image file.
        patch_size (int): Desired patch size.

    Returns:
        torch.Tensor: Normalized image tensor with shape [1, H, W].
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    return torch.tensor(img, dtype=torch.float32)


class PatchDataset(Dataset):
    """
    Custom PyTorch Dataset for image patches and labels.
    """
    def __init__(self, img_paths, labels, patch_size):
        self.img_paths = img_paths
        self.labels = labels
        self.patch_size = patch_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = convert(self.img_paths[idx], self.patch_size)
        label = self.labels[idx]
        return img, label


# ====================== Device & Training Parameters ======================
def auto_device_and_params(batch_base: int = 64):
    """
    Automatically detect device and determine training parameters.

    Returns:
        device (torch.device): Selected device.
        amp (bool): Whether automatic mixed precision is enabled.
        batch_size (int): Calculated batch size.
        num_workers (int): Number of data loader workers.
    """
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        amp = False
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        amp = True
    else:
        device = torch.device("cpu")
        amp = False

    cpu_count = os.cpu_count() or 4
    num_workers = min(8, max(1, cpu_count // 2))
    batch_size = batch_base

    device_info = ""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1024**3
        batch_size = min(batch_base * int(total_mem_gb / 4), 1024)
        device_info = f"CUDA Device: {props.name}, Total Memory: {total_mem_gb:.2f} GB"
    elif device.type == "mps":
        device_info = "MPS Device detected"
    else:
        device_info = f"CPU detected, {cpu_count} cores"
        batch_size = batch_base // 2

    # Print configuration
    print("==== Device & Training Configuration ====")
    print(device_info)
    print(f"Using device: {device}, AMP enabled: {amp}")
    print(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print("========================================")

    return device, amp, batch_size, num_workers


# ====================== CNN Training Function ======================
def train_cnn(img_paths, y, patch_size, epochs: int = 20, batch_base: int = 64):
    """
    Train a CNN on given image paths and labels.

    Args:
        img_paths (list[str]): Paths to images.
        y (list[int]): Corresponding labels.
        patch_size (int): Patch size for input images.
        epochs (int): Number of training epochs.
        batch_base (int): Base batch size.
    """
    # Fix random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get device and training parameters
    device, use_amp, batch_size, num_workers = auto_device_and_params(batch_base)

    # Dataset and DataLoader
    dataset = PatchDataset(img_paths, y, patch_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    # Initialize model
    model = SimpleCNN(patch_size).to(device)

    # Compute class weights to handle imbalance
    counter = Counter(y)
    w0 = 1.0
    w1 = counter[0] / counter[1] if counter[1] > 0 else 1.0
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Automatic Mixed Precision (AMP) scaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    print("[CNN] Start training ...")
    start_time = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

    end_time = time.perf_counter()
    print(f"[CNN] Training time: {end_time - start_time:.2f} seconds")
    return model