import matplotlib.pyplot as plt

from algorithms_pca.matrix import Matrix, DimensionError
from matplotlib.figure import Figure


def plot_pca_projection(X_proj: Matrix) -> Figure:
    if not isinstance(X_proj, Matrix):
        raise DimensionError("X_proj must be a Matrix instance.")
    if X_proj.cols != 2:
        raise DimensionError("Projection must have exactly 2 columns.")
    data = X_proj._data
    xs = [row[0] for row in data]
    ys = [row[1] for row in data]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(xs, ys)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Projection')
    return fig
