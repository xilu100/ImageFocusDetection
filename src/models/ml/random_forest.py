import numpy as np
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

from src.models import img_to_X


def train_random_forest(img_paths, y, patch_size, n_estimators=100, max_depth=None):
    X = np.array([
        img_to_X.convert(p, patch_size)
        for p in tqdm(img_paths, desc="Processing images")
    ])
    y = np.array(y)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    return model
