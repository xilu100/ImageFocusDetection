import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

from src.models import img_to_X


def train_svm(img_paths, y, patch_size):
    X = []
    for p in tqdm(img_paths, desc="Processing images"):
        X.append(img_to_X.convert(p, patch_size))

    X = np.array(X)
    y = np.array(y)

    model = SVC(kernel='linear')
    model.fit(X, y)

    return model
