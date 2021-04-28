
import numpy as np # linear algebra
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
X = iris.data[:, :2]
y = iris.target
h= .2 # feiner Borders ?
# Erklärung dazu noch googeln !!!!!!!!!
i = [1, 3, 15]
split = [0.3, 0.1]
# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
cmap_bold = ListedColormap(['#a30b0b', '#089e08', '#006ea6'])

for r in split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=r)
    for x in i:
        neigh = KNeighborsClassifier(n_neighbors=x, weights='distance')
        neigh.fit(X_train, y_train)

        # calculate min, max and limits
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # Put the result into a color plot
        pred = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
        pred = pred.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, pred, cmap=cmap_light, shading='auto')

        # Plot also the training points
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"3-Class classification (k = {x}), split = {r}")
        plt.show()