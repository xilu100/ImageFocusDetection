import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from src.models import img_to_X


def train_tree(img_paths, y, patch_size):

    X = []
    for p in img_paths:
        X.append(img_to_X.convert(p, patch_size))

    X = np.array(X)
    y = np.array(y)

    model = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    model.fit(X, y)

    return model