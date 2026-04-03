import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from src.models import img_to_X


def train_tree(img_paths, y, patch_size):
    X = []
    print("Start <Decision Tree> images processing...")
    for p in tqdm(img_paths, desc="Processing images"):
        X.append(img_to_X.convert(p, patch_size))

    X = np.array(X)
    y = np.array(y)

    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )
    print("Start <Decision Tree> training...")
    model.fit(X, y)

    return model
