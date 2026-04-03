import time
import random
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# ===== CNN模型 =====
class SimpleCNN(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

        self.num_classes = 2
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, patch_size, patch_size)
            dummy = self.pool(self.conv1(dummy))
            dummy = self.pool(self.conv2(dummy))
            fc_size = dummy.view(1, -1).shape[1]

        self.fc = nn.Linear(fc_size, self.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ===== 图像处理 =====
def convert(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Can not read image: {img_path}")

    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return torch.tensor(img, dtype=torch.float32)


# ===== 训练函数（完整版）=====
def train_cnn(img_paths, y, patch_size):
    # ===== 固定随机种子（保证稳定）=====
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ===== 设备 =====
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\nUsing device: {device}")

    # ===== 数据加载 =====
    X = [convert(p, patch_size) for p in img_paths]
    X = torch.stack(X)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # ===== 模型 =====
    model = SimpleCNN(patch_size).to(device)

    # ===== class weight（处理类别不平衡）=====
    counter = Counter(y.numpy())
    w0 = 1.0
    w1 = counter[0] / counter[1] if counter[1] > 0 else 1.0

    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ===== 训练 =====
    print("[CNN] Start training ...")
    start_time = time.perf_counter()

    model.train()
    epochs = 20

    for epoch in range(epochs):
        total_loss = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    end_time = time.perf_counter()
    print(f"[CNN] Training time: {end_time - start_time:.2f} seconds")

    return model