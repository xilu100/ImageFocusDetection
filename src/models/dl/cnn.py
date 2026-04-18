import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Optional, cast

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tools.log import print_and_save,save


class SimpleCNN(nn.Module):
    def __init__(self, patch_size: int, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

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


def convert(img_path: str, patch_size: int, assume_fixed_size: bool = True) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    if not assume_fixed_size and img.shape != (patch_size, patch_size):
        img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
    if assume_fixed_size and img.shape != (patch_size, patch_size):
        raise ValueError(f"Unexpected patch shape {img.shape}, expected {(patch_size, patch_size)}: {img_path}")
    return torch.from_numpy(img).unsqueeze(0).to(torch.float32).div_(255.0)


class PatchDataset(Dataset):
    def __init__(self, img_paths, labels, patch_size, assume_fixed_size: bool = True):
        self.img_paths = img_paths
        self.labels = labels
        self.patch_size = patch_size
        self.assume_fixed_size = assume_fixed_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = convert(self.img_paths[idx], self.patch_size, assume_fixed_size=self.assume_fixed_size)
        label = self.labels[idx]
        return img, label


def build_lmdb_from_paths(
        img_paths,
        labels,
        patch_size: int,
        lmdb_path: Path,
        assume_fixed_size: bool = True,
        commit_interval: int = 10000
):
    lmdb_path = Path(lmdb_path)
    lmdb_path.parent.mkdir(parents=True, exist_ok=True)

    bytes_per_sample = patch_size * patch_size + 8
    map_size = int(max(1 << 30, len(img_paths) * bytes_per_sample * 1.5))
    env = lmdb.open(str(lmdb_path), map_size=map_size, subdir=False, lock=True)

    print(f"[LMDB] Building cache: {lmdb_path}")
    start = time.perf_counter()
    txn = env.begin(write=True)

    for idx, (img_path, label) in enumerate(zip(img_paths, labels)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")
        if not assume_fixed_size and img.shape != (patch_size, patch_size):
            img = cv2.resize(img, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        if assume_fixed_size and img.shape != (patch_size, patch_size):
            raise ValueError(f"Unexpected patch shape {img.shape}, expected {(patch_size, patch_size)}: {img_path}")

        key_img = f"img-{idx:09d}".encode("ascii")
        key_lbl = f"lbl-{idx:09d}".encode("ascii")
        txn.put(key_img, img.tobytes())
        txn.put(key_lbl, int(label).to_bytes(4, byteorder="little", signed=True))

        if (idx + 1) % commit_interval == 0:
            txn.commit()
            txn = env.begin(write=True)
            print(f"[LMDB] Processed {idx + 1}/{len(img_paths)}")

    txn.put(b"__len__", str(len(img_paths)).encode("ascii"))
    txn.put(b"__patch_size__", str(patch_size).encode("ascii"))
    txn.commit()
    env.sync()
    env.close()

    elapsed = time.perf_counter() - start
    print_and_save(f"[LMDB] Build done in {elapsed:.2f}s")


class LmdbPatchDataset(Dataset):
    def __init__(self, lmdb_path: Path):
        self.lmdb_path = str(lmdb_path)
        self.env = None
        self.txn = None
        with lmdb.open(self.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False) as env:
            with env.begin(write=False) as txn:
                len_raw = txn.get(b"__len__")
                patch_raw = txn.get(b"__patch_size__")
                if len_raw is None or patch_raw is None:
                    raise ValueError(f"LMDB metadata missing in {self.lmdb_path}")
                self.length = int(bytes(len_raw).decode("ascii"))
                self.patch_size = int(bytes(patch_raw).decode("ascii"))

    def _lazy_init(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256
            )
            self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._lazy_init()
        key_img = f"img-{idx:09d}".encode("ascii")
        key_lbl = f"lbl-{idx:09d}".encode("ascii")
        img_raw = self.txn.get(key_img)
        lbl_raw = self.txn.get(key_lbl)

        if img_raw is None or lbl_raw is None:
            raise IndexError(f"Missing sample at index {idx} in LMDB")

        img = np.frombuffer(img_raw, dtype=np.uint8).reshape(self.patch_size, self.patch_size).copy()
        img = torch.from_numpy(img).unsqueeze(0).to(torch.float32).div_(255.0)
        label = int.from_bytes(lbl_raw, byteorder="little", signed=True)
        return img, label


def auto_device_and_params(batch_base: int = 64):
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

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1024 ** 3
        batch_size = min(batch_base * int(total_mem_gb / 4), 1024)
        device_info = f"CUDA Device: {props.name}, Total Memory: {total_mem_gb:.2f} GB"
    elif device.type == "mps":
        device_info = "MPS Device detected"
    else:
        device_info = f"CPU detected, {cpu_count} cores"
        batch_size = batch_base // 2

    print_and_save("==== Device & Training Configuration ====")
    print_and_save(device_info)
    print_and_save(f"Using device: {device}, AMP enabled: {amp}")
    print_and_save(f"Batch size: {batch_size}, num_workers: {num_workers}")
    print_and_save("========================================")

    return device, amp, batch_size, num_workers


def train_cnn(
        img_paths,
        y,
        patch_size,
        epochs: int = 5,
        batch_base: int = 64,
        lmdb_path: Optional[str] = None,
        build_lmdb_if_missing: bool = True,
        assume_fixed_size: bool = True
):
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device, use_amp, batch_size, num_workers = auto_device_and_params(batch_base)

    dataset = None
    if lmdb_path is not None:
        lmdb_file = Path(lmdb_path)
        if build_lmdb_if_missing and not lmdb_file.exists():
            build_lmdb_from_paths(
                img_paths=img_paths,
                labels=y,
                patch_size=patch_size,
                lmdb_path=lmdb_file,
                assume_fixed_size=assume_fixed_size
            )
        if lmdb_file.exists():
            dataset = LmdbPatchDataset(lmdb_file)
            print_and_save(f"[CNN] Using LMDB dataset: {lmdb_file}")

    if dataset is None:
        dataset = PatchDataset(img_paths, y, patch_size, assume_fixed_size=assume_fixed_size)
        print("[CNN] Using file-path dataset")

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(**loader_kwargs)

    model = SimpleCNN(patch_size).to(device)

    counter = Counter(y)
    w0 = 1.0
    w1 = counter[0] / counter[1] if counter[1] > 0 else 1.0
    weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print("[CNN] Start training ...")
    start_time = time.perf_counter()
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=(device.type == "cuda"))
            batch_y = batch_y.to(device, non_blocking=(device.type == "cuda"))
            optimizer.zero_grad()

            if use_amp:
                assert scaler is not None
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                scaled_loss = cast(torch.Tensor, scaler.scale(loss))
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        print_and_save(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    end_time = time.perf_counter()
    print_and_save(f"[CNN] Training time: {end_time - start_time:.2f} seconds")
    save("\n")
    return model
