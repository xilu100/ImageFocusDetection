from sklearn.decomposition import PCA


def reduce_dimensions(X, n_components):
    original_dim = X.shape[1]

    if n_components == -1:
        print(f"[PCA] Skip PCA: n_components={n_components}, use original features (dim={original_dim}).")
        return X, None

    if n_components >= original_dim:
        print(
            f"[PCA] Skip PCA: n_components={n_components} >= original_dim={original_dim}, "
            "use original features."
        )
        return X, None

    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca
