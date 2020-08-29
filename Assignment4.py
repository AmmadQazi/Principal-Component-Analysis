import pandas as pd
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
import seaborn as sns

df_train = pd.read_csv('CC GENERAL.csv')
print(df_train.head())

df_train.replace('null',np.NaN, inplace=True)
df_train.replace(r'^\s*$', np.NaN, regex=True, inplace=True)
df_train.fillna(value=df_train.median(), inplace=True)

X = df_train.iloc[:,1:18]

plt.figure(figsize=(12,10))
cor = X.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

correlated_features = set()

for i in range(len(cor.columns)):
    for j in range(i):
        if abs(cor.iloc[i, j]) > 0.8:
            colname = cor.columns[i]
            correlated_features.add(colname)

print(correlated_features)
X.drop(labels=correlated_features, axis=1, inplace=True)


pca = PCA(n_components = 2, random_state=1)
X_pca = pca.fit_transform(X)

print('Explained Variance Ratio : ' + str(pca.explained_variance_ratio_.cumsum()[1])) # Variability in data maintained


# variables for elbow method (for determining n_clusters)
distortions = []

K_to_try = range(1, 10)

# elbow method for finding optimal K
for i in K_to_try:
    model = KMeans(
            n_clusters=i,
            init='k-means++',   
            random_state=1)
    model.fit(X_pca)
    distortions.append(model.inertia_)

# plot elbow
plt.plot(K_to_try, distortions, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion')
plt.show()
plt.figure()

# use the best K from elbow method
model = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=1)

model = model.fit(X_pca)
y = model.predict(X_pca)

# plot k means cluster
plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], s = 50, c = 'yellow', label = 'Cluster 1')
plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], s = 50, c = 'green', label = 'Cluster 2')
plt.scatter(X_pca[y == 2, 0], X_pca[y == 2, 1], s = 50, c = 'red', label = 'Cluster 3')
plt.scatter(X_pca[y == 3, 0], X_pca[y == 3, 1], s = 50, c = 'purple', label = 'Cluster 4')
plt.scatter(X_pca[y == 4, 0], X_pca[y == 4, 1], s = 50, c = 'orange', label = 'Cluster 5')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 100, c = 'blue', label = 'Centroids') # centroids
plt.title('Clusters of Students')
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.legend()
plt.grid()
plt.show()
plt.figure()


tsne = TSNE()

tsne1 = TSNE(n_components = 2, random_state =0)
x_test_2d = tsne1.fit_transform(X)
target_ids = range(15)
target_names='BALANCE','BALANCE_FREQUENCY','PURCHASES','INSTALLMENTS_PURCHASES','CASH_ADVANCE','PURCHASES_FREQUENCY','ONEOFF_PURCHASES_FREQUENCY','CASH_ADVANCE_FREQUENCY','CASH_ADVANCE_TRX','PURCHASES_TRX','CREDIT_LIMIT','PAYMENTS','MINIMUM_PAYMENTS','PRC_FULL_PAYMENT','TENURE'

plt.figure(figsize=(6, 5))
colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'gold', 'orange', 'purple','grey','aqua','crimson','lime','peru'
for i, c, labels in zip(target_ids, colors,target_names):
    plt.scatter(x_test_2d[y == i, 0], x_test_2d[y == i, 1], c=c, label=labels)
plt.legend()
plt.show()
