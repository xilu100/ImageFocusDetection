import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===== CNN模型 =====
class SimpleCNN(nn.Module):
    def __init__(self, patch_size):
        super().__init__()

        self.num_classes = 2  # ✅ 二分类固定
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


# ===== 单张图处理（替代 flatten 版本）=====
def convert(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    if img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)

    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # (1, H, W)

    return torch.tensor(img)


# ===== 主函数（接口保持一致）=====
def train_cnn(img_paths, y, patch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    X = [convert(p, patch_size) for p in img_paths]
    X = torch.stack(X).to(device)
    y = torch.tensor(y).to(device)

    model = SimpleCNN(patch_size).to(device)  # 二分类固定
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    return model
