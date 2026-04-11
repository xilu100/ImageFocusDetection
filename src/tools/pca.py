from sklearn.decomposition import PCA


def reduce_dimensions(X, n_components):
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
